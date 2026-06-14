# Grant 项目重构记忆 — 复用 swxy(DeepDOC + RAG) 改造打分系统

> 最后更新：2026-06-14
> 作用：记录把成熟项目 **swxy**（`E:\agent_projects\LLM_agent_course\S2_llm\swxy`，一个基于 RAGFlow/DeepDOC 的 RAG 问答系统）尽可能多地搬进本 grant 打分项目的**总体思路、已完成工作、架构决策、后续路线**。供后续随时参考。

---

## 0. 一句话目标

把 swxy 里成熟的 **DeepDOC 解析 + chunk 切分 + embedding 向量检索 + ES 库 + 问答** 模块，尽可能多地"整块复制 + 小改适配"进本项目，把一个"单篇 NIHR 申请规则式打分器"升级成 **语料级 RAG 评估系统**。允许为最大化复用而适度拓展目标。

两个项目位置：
- 本项目（grant）：`E:\agent_projects\nlp_grant_project`
- 参照源（swxy）：`E:\agent_projects\LLM_agent_course\S2_llm\swxy\backend\app\service\core`
- 运行环境：用 `D:\msc_AI\ai_venv`（Windows，Python 3.12；缺包直接 pip install）

---

## 1. 本项目现状（被改造对象）

NIHR 基金申请自动打分系统。核心链路（从 PDF 到分数）：

```
PDF → all_type_parser(规则式解析→具名section JSON) → build_pool(切chunk)
    → Stage1 belief accumulation(逐section扫描，rubric子标准→证据chunk映射)
    → Stage2 final scoring(逐rubric section打0-5分)
    → 加权聚合 + doc_type排除 → 0-100分 + 可审计证据
```

关键文件：
- 解析路由：[src/all_type_parser/all_type_parser.py](src/all_type_parser/all_type_parser.py)（特殊格式 parser `fellowships_parser.py`/`RfPB_parser.py`/`pdf_parser.py` **不可动**）
- 切块：[src/pool/build_pool.py](src/pool/build_pool.py)（当前字符级切，`MAX_CHARS=1200`，`_split_long_text` L41，`build_chunk_pool` L358）
- 打分管线：[src/scoring/pipeline.py](src/scoring/pipeline.py)
- LLM 客户端（依赖注入）：[qwen3_ollama.py](qwen3_ollama.py)（`_Scorer`，接口仅 `generate_json(messages, schema, max_tokens)` + `.model_name` + `.last_response_body`）

**重要发现（RAG 切入点）**：`pipeline.py` 的 Stage2 实际把**整篇全文**灌给每个 rubric section（[pipeline.py:1676](src/scoring/pipeline.py#L1676) `_build_full_application_text`），README 里说的 dynamic context selection（`_build_scoped_application_text` L408）**是死代码没被调用**。所以"检索"几乎不存在——这正是 RAG 升级最大的空档。

**评估体系**（`experiments/`）已有四类，且有 ground truth 思路：
- 可靠性/稳定性（同篇跑3次方差，pipeline vs 单次baseline）
- **证据命中 F1**（`ev_hit_f1_analysis.ipynb`）：比"本方法选的证据 chunk" vs "强参考模型(GPT-4.5/codex)选的 used_chunk_ids" → P/R/F1。注意 **ground truth 是模型当裁判(silver standard)，非人工 gold**。
- 与强参考模型分数吻合度
- 成功 vs 失败申请区分度（`compare_successful_unsuccessful.ipynb`）
- before/after 敏感性

---

## 2. 已完成：解析层重构（DeepDOC fallback + DOCX/PPT）

**已落地并验证通过。** 把 swxy 的 DeepDOC 解析栈整套搬进来作为 fallback。

### 新建包 `src/deepdoc_engine/`（swxy `service/core` 的自包含裁剪副本）
- 复制 + `sed` 改导入 `service.core.` → `deepdoc_engine.`
- 解析器：`deepdoc/parser/{pdf_parser, docx_parser, ppt_parser, utils, __init__(裁剪)}`
- 视觉栈：`deepdoc/vision/{ocr, recognizer, layout_recognizer, table_structure_recognizer, operators, postprocess, __init__}`（跳过 seeit/t_ocr/t_recognizer 等 CLI 工具）
- 切块/工具：`rag/nlp/__init__.py`（naive_merge 等）、`rag/utils/__init__.py`、`rag/app/naive.py`（裁剪到 pdf/docx，去掉 tika/markdown）
- 模型二进制（已拷入，约107MB，离线可用）：`rag/res/deepdoc/{det,rec,layout,tsr}.onnx + ocr.res + updown_concat_xgb.model`

### 两处适配（其余忠实照搬）
1. `rag/nlp/rag_tokenizer.py` → **改写成 NLTK 英文 shim**（原版是中文 huqie + 61MB 字典 + datrie；本项目英文，不需要）。保持同接口：`tokenize/fine_grained_tokenize/tag/is_chinese`。
2. `deepdoc/vision/{ocr,recognizer}.py` 的 `cuda_is_available()` → 增加 `"CUDAExecutionProvider" in ort.get_available_providers()` 判断（修 CPU 机器上误走 GPU 路径导致的 `gpu:0` arena 崩溃）。
3. `get_project_base_directory()` **无需改**（保留了相同相对目录布局，自动指向 `deepdoc_engine/`）。

### 胶水 + 路由
- 新增 [src/all_type_parser/deepdoc_fallback.py](src/all_type_parser/deepdoc_fallback.py)：`parse_pdf/parse_docx/parse_pptx`，输出沿用 `{"APPLICATION DETAILS": {"Raw Content": ...}}` 契约（优雅降级，build_pool 已能消费）。
- 路由 [all_type_parser.py](src/all_type_parser/all_type_parser.py)：PDF 顺序改为 **规则式 → DeepDOC → LLM**；DOCX 加 DeepDOC 中间级；新增 **PPTX/PPT** 分支。

### 依赖 + 兼容
- `requirements.txt` 新增：onnxruntime, opencv-python, **xgboost>=2.0,<3.1**（3.1+ 删了旧二进制模型格式）, pypdf, python-pptx, pyclipper, shapely, tiktoken, huggingface_hub, chardet, six。不需要 torch（被 guard）、fasttext（懒加载且不在路径上）、tika/markdown（已裁剪）。
- `start.sh` 加 NLTK `averaged_perceptron_tagger(_eng)`。

### 验证结果（全绿，ai_venv）
- 真实 PDF → DeepDOC fallback 18s 出 11KB 干净文本；DOCX 出 71KB；离线(`HF_HUB_OFFLINE=1`)模型全加载；
- **回归 OK**：已知 fellowship PDF 仍走规则式 parser、保留具名 section，DeepDOC 不触发；
- build_pool 能消费 fallback 输出（15 chunks + 派生特征）。

---

## 3. 核心架构决策（已和用户确认）

1. **保留具名 section 结构，不做纯 flat chunking。** 因为 NLP 派生特征（plain english 可读性、application form 分析）、`SECTION_TO_PARSER_SECTIONS` 规则先验、Stage1 逐 section 扫描、doc_type 适配全都依赖 section 身份。DeepDOC fallback 只对**未知格式**输出 flat `Raw Content`（优雅降级）。RAG 检索是 section **之上的叠加**，不是替换。

2. **rubric signal = query**：整个打分本质是 **multi-query RAG**（固定的 rubric 当 query 对单篇文档检索），不需要人来提问。

3. **双库架构**（解开"要不要 application_id"的关键）：
   | | A. 打分库(per-request) | B. 语料库(persistent ES) |
   |---|---|---|
   | 内容 | 仅当前 PDF 的 chunk | 所有**已知结果**申请的 chunk |
   | 生命周期 | 临时(内存/临时索引) | 长期 |
   | application_id 过滤 | **不需要**(只一篇) | 需要(+ success_label) |
   | 用途 | 打分检索 + 单篇问答 | 跨库范例/校准 + 语料问答 |
   - **打分路径只用 A，不碰 application_id**（每次只传一篇，库里只有它）。
   - 跨库特征才用 B。

4. **离线优先**：embedding/rerank 用**本地** sentence-transformers / bge（项目已装 sentence-transformers、hnswlib），不用 swxy 的 DashScope 云 API；与"本地 Ollama、可离线"定位一致。保持 `generate_embedding` 函数签名不变，让 swxy 的 `Dealer` 可直接用。

---

## 4. 后续路线（按优先级/风险）

### 步骤1：切块升级 ✅ 已完成（2026-06-14）
**实现**：改造了 [build_pool.py](src/pool/build_pool.py)，保留 section 外层分组，section 内切块换成 swxy 的 `naive_merge` 体系。
- 新增 `_split_sentences`（句子切分，piece 带尾空格避免 naive_merge 拼接时"粘连")、`_heading_flags`（用 swxy `bullets_category`+`BULLET_PATTERN`+`not_bullet` 检测标题）、`_chunk_text`（标题分组 + `naive_merge` token 级合并）、`_extract_positions`（剥离并捕获 DeepDOC `@@..##` 位置标记）。
- `add_leaf` 重写：HTML 表格→`tokenize_table` 单独成块（`is_table=True`，HTML 原样保留）；文本→`_chunk_text`+`tokenize`（产出 `content_ltks/content_sm_ltks/title_tks` BM25 字段）。
- `PoolChunk` + `pool_lookup` 每条新增字段：`content_ltks, content_sm_ltks, title_tks, token_count, is_table, position`。
- 扩展了 [deepdoc_fallback.py](src/all_type_parser/deepdoc_fallback.py)：`parse_pdf`/`parse_docx` 改用 `Pdf()`/`Docx()` 包装器直接拿 (sections, tables)，把 DeepDOC 还原的 HTML 表格写到 `APPLICATION DETAILS > Document Tables`，build_pool 据此产出表格 chunk。
- 常量：`CHUNK_TOKEN_NUM=256`、`DELIMITER="\n.!?;:"`（英文）；`MAX_CHARS` 保留仅为向后兼容 import。
- **验证**：真实 PDF → 22 chunks（3 表格块 + 17 带位置 + 全部带 BM25 字段）；合成样例 Detailed Research Plan 正确按 1./2./3. 标题切成 3 块；DOCX 71KB。测试 18 passed / 1 failed（`test_score_application_base...` 的 `pr.2` KeyError 是**改动前就存在**的 belief-shape drift，与切块无关）。
- **新依赖耦合**：build_pool 现在 import 期会拉起 `deepdoc_engine.rag.nlp`（nltk shim）+ `rag.utils`（tiktoken）+ chardet + PIL（不含 onnx/cv2，轻量）。

（原计划如下，供参考）保留 section 外层分组，section 内切块换成 swxy 的 `naive_merge` 体系：
- **token 级合并**：`naive_merge`(chunk_token_num + 句末 delimiter，中文标点换英文) 替换 `_split_long_text`，不腰斩句子。
- **结构感知**：`BULLET_PATTERN` 识别标题/编号，标题与正文绑定。
- **表格成块**：`tokenize_table` + DeepDOC 表格 HTML，每张表单独 chunk，喂 `budget.py`。
- **携带位置**：DeepDOC `@@page\tx0\tx1\ttop\tbottom##`(`_line_tag`) 写进 chunk 元数据，为 PDF 高亮做准备。
- **产出 BM25 字段**：`tokenize_chunks` 生成 `content_ltks/content_sm_ltks/title_tks`（承上启下：有分词字段，混合检索才能跑）。
- `PoolChunk` 新增字段：`token_count, content_ltks, content_sm_ltks, position, is_table`。

### 步骤2+3：embedding 入库 + 混合检索（双库）✅ 已完成（2026-06-14）
**实现**：复用 swxy 的 `Dealer`（混合检索 BM25+向量+融合+rerank）跑在两个库上。
- **本地向量后端**：重写 [model.py](src/deepdoc_engine/rag/nlp/model.py) → 本地 sentence-transformers（`generate_embedding` 用 `BAAI/bge-small-en-v1.5` 384维；`rerank_similarity` 用 `cross-encoder/ms-marco-MiniLM-L-6-v2`，sigmoid 归一；签名不变，Dealer 原样用）。env: `EMBED_MODEL`/`RERANK_MODEL`。
- **临时"当前PDF"库（内存，会话结束即消）**：新写 [inmem_conn.py](src/deepdoc_engine/rag/utils/inmem_conn.py) `InMemoryConnection(DocStoreConnection)`，用 numpy 复现 ES 的混合检索（过滤→BM25 over content_ltks→cosine→加权融合 0.05/0.95→topN），让 `Dealer` 原样跑、**打分不依赖 ES**。
- **持久语料库（ES，区分 successful/unsuccessful）**：复用 [es_conn.py](src/deepdoc_engine/rag/utils/es_conn.py)（改：dotenv 可选、auth 走 env、`request_timeout`）；vendor `conf/mapping.json` 并加 `*_384_vec` 模板；每个 chunk 带 `doc_id=application_id` + `success_label`。
- **建库/检索胶水**：[src/retrieval/indexer.py](src/retrieval/indexer.py)（`build_index_from_pool` 内存 / `build_corpus_es` 一次性建 ES 语料，含 CLI `python -m src.retrieval.indexer --recreate`）+ [src/retrieval/retriever.py](src/retrieval/retriever.py)（`evidence_for_section` 查当前PDF / `fewshot_for_section` 查 ES、**排除当前申请**、优先 successful、`section_query` 限 50 词）。
- **接入 pipeline**：[pipeline.py](src/scoring/pipeline.py) Stage2 把 `_build_full_application_text` 换成 `_retrieval_scope`（检索证据，失败/空→回退全文）；`build_final_scoring_messages` 加 `calibration_examples`（few-shot，仅校准 0-5 标尺、禁止当证据/抄袭）。新参数 `use_retrieval`/`evidence_top_k`/`corpus_index`/`fewshot_n`；[qwen3_ollama.py](qwen3_ollama.py) 经 env `GRANT_USE_RETRIEVAL`/`GRANT_CORPUS_INDEX` 暴露。`debug.retrieval_used`/`fewshot_used` 记录。
- **其他改动**：rag_tokenizer 加 `strQ2B`/`tradi2simp`（CJK no-op）；synonym WordNet 扩展默认关（`RAG_SYNONYM_WORDNET=1` 开，否则 ES 子句爆 maxClauseCount=1310）；search_v2 stray print → logging、success_label/parser_section 进 src+chunk dict。
- **新依赖**：sentence-transformers(已装)、elasticsearch/elasticsearch-dsl(`<9`)。NLTK 加 wordnet/omw-1.4。
- **验证（live ES 8.11 via Docker）**：corpus 建成 22 篇(11 succ/11 unsucc, 2220 chunks)；few-shot 正确排除当前申请、优先 successful、训练查询命中训练段(sim 0.74-0.76)；端到端 `retrieval_used=True`/`fewshot_used=True`，6/6 stage2 prompt 注入了 calibration_examples；测试 18 passed/1 pre-existing fail；ES 关闭时优雅降级（few-shot 跳过、打分照常）。
- **ES 容器**：`docker run -d --name grant-es -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false elasticsearch:8.11.3`；停 `docker stop grant-es`、起 `docker start grant-es`。建库：`ES_HOST=http://localhost:9200 python -m src.retrieval.indexer --recreate`。

### 检索质量调优 A+B ✅ 已完成（2026-06-14）
对真实申请实测后修两处检索噪声（均在检索层，不动 build_pool 契约/测试）：
- **A 过滤碎块**：[indexer.py](src/retrieval/indexer.py) `_indexable_items` 在建索引前丢弃 <5 token 的碎块（如 "2."、"3. Literature"）。实测 166→141（去 25 个）。
- **B 派生块按维度限定**：[retriever.py](src/retrieval/retriever.py) `_filter_derived` 把 3 个派生合成块限定到各自维度——`Application Context`→general、`Plain English NLP Analysis`→proposed_research(pr.1)、`Application Form Analysis`→application_form(af.*)，其余维度排除。evidence + fewshot 都过。实测后 training/wpcc 召回变干净（纯申请人原文），pr.1 去掉了乱入的 Application Form Analysis。
- 注意：派生块本质是 build_pool 用代码算出的"合成证据块"（非 DB metadata），块头自带 "evidence for pr.1 / af.*" 说明，B 正是按此意图限定。测试仍 18 passed/1 pre-existing fail。

### (原计划) 步骤2：embedding 落地
照搬 [file_parse.py `execute_insert_process`](../LLM_agent_course/S2_llm/swxy/backend/app/service/core/file_parse.py) 流水线：parse→build_pool→对每 chunk `generate_embedding`(本地ST)→先存内存。

### 步骤3：检索替换（核心）
整块搬 swxy [search_v2.py `Dealer.retrieval`](../LLM_agent_course/S2_llm/swxy/backend/app/service/core/rag/nlp/search_v2.py)（混合 BM25+向量、`FusionExpr` 融合、`rerank_by_model` 重排、rank feature）+ `query.py FulltextQueryer`。
- 把 [pipeline.py:1676](src/scoring/pipeline.py#L1676) 的 `_build_full_application_text` 换成：每个 rubric section/signal 当 query 取 top-k chunk。
- belief_state 作为先验叠加（召回 + belief 加权）。
- rerank 用本地 bge-reranker 替 gte-rerank。

### 步骤4：跑消融（出简历核心数据）
用现有 `ev_hit_f1_analysis.ipynb` 对比"全文灌入 vs 混合检索+rerank"的 **evidence-hit F1** 和 token 消耗。

### 步骤5（可选，最大复用）：上 ES 持久语料库 B
搬 `es_conn.py` + `doc_store_conn.py`。**一个总索引** `grant_chunks` 装所有 PDF chunk，doc 带 `application_id, doc_type, section, content_with_weight, content_ltks, q_<dim>_vec, position, success_label, gold_score?`。建库流水线遍历 `data/successful + data/unsuccessful` 批量灌入。

### 步骤6（可选，demo 杀手锏）：问答 + 引用
- **单篇问答(主)**：在打分库 A 上检索 → 评审 copilot（"为什么 feasibility 给 3 分"），= swxy `chat_on_docs`/`quick_parse` 搬过来，零串台。
- **语料问答(拓展)**：在 B 上检索。
- 引用：搬 `Dealer.insert_citations` → Stage2 pros/drawbacks 挂证据 chunk_id + DeepDOC 位置 → web UI 高亮。

---

## 5. 拓展目标：怎么用到"所有 PDF 的 chunk"（语料库 B 的价值）

打分本身只用单篇；要让全库发挥价值，加这些**跨库特征**（按价值排，都大量复用 swxy）：
1. **范例检索校准(few-shot anchoring)** ⭐：打某 signal 时从**成功**申请检索同维度相似 chunk，作为"满分长什么样"塞进 prompt。
2. **kNN 分数校准/百分位**：新申请某段在**已打分语料**找最近邻 + 其分数 → 相对定位。
3. **跨申请重复/样板检测**：全库找近重复 chunk → 喂 Application Form 原创性 signal。
4. **语料级问答/检索 UI**：swxy chat/检索 app 整体复用。
5. **成功 vs 失败分布对比**：强化 `compare_successful_unsuccessful` 实验。

### 为什么有意义 + 必守的诚实前提
- **语料库有标签(知道成功/失败)，新上传申请无标签(结果未知)**。价值在于"有标签→给无标签定位/校准"。正因为新申请未知才有意义。
- ⚠️ **防泄漏（否则方法学站不住）**：
  1. 打分当前申请时，参照集**排除它自己**（`application_id != 当前`，不能自检索）。
  2. 评估做 **leave-one-out**（给某篇打分时把它移出参照语料）。
  3. 结果标签(中标/落选)是"逐 rubric signal 质量"的**远端弱代理**，受文本外因素影响；表述成"用历史结果做弱监督校准/参照"，**不是真值训练**。

---

## 6. swxy 模块复用映射（速查）

| swxy 模块 | 作用 | grant 落点 | 状态 |
|---|---|---|---|
| `deepdoc/parser/*` + `deepdoc/vision/*` | DeepDOC 解析 | `src/deepdoc_engine/` | ✅ 已搬 |
| `rag/nlp/__init__` naive_merge/tokenize_chunks/tokenize_table/BULLET_PATTERN | token级+结构切块 | 改造 build_pool | 步骤1 |
| `rag/nlp/model.py` generate_embedding | 向量化(换本地ST) | 建向量 | 步骤2 |
| `rag/nlp/model.py` rerank_similarity | 重排(换本地bge) | 检索重排 | 步骤3 |
| `rag/nlp/search_v2.py` Dealer | 混合检索+融合+重排 | 替换全文灌入 | 步骤3 |
| `rag/nlp/query.py` FulltextQueryer | BM25 query 构造 | 检索文本侧 | 步骤3 |
| `rag/utils/es_conn.py`+`doc_store_conn.py` | ES 向量库 | 语料库 B | 步骤5 |
| `file_parse.py` execute_insert_process | 建库流水线 | 索引构建 | 步骤2/5 |
| `Dealer.insert_citations` | 答案→证据回引 | 引用标注 | 步骤6 |
| `term_weight.py`/`synonym.py` | 词权重/同义词 | query增强 | ⚠️中文词典，英文需替换 |
| `deepdoc/parser/{excel,html,json,md,txt}` | 多格式 | 扩展ingest | 备选 |

---

## 7. 待定决策
1. **向量库后端**：ES(最大复用，要起服务) vs 内存 hnswlib(轻量贴合单篇)。用户倾向"尽可能多复用"→ 偏 ES；稳妥可先内存跑通逻辑再切 ES。
2. **embedding/rerank 模型**：本地 sentence-transformers/bge(推荐，离线) vs 照搬 DashScope 云(要 key/联网)。

---

## 8. 杂项注意（坑）
- **tiktoken 缓存文件** `src/deepdoc_engine/9b5ad71b...`：是 `rag/utils/__init__.py` 把 `TIKTOKEN_CACHE_DIR` 设成包目录导致的 cl100k_base 缓存。tiktoken 库是切块数 token 必需(import 时建 encoder)；缓存文件可删(联网会自建)，但建议**保留 + 加 .gitignore + 把缓存目录改到 `rag/res/.tiktoken/`** 避免污染包根。删了且离线会导致 import 失败。
- **`✓`/`✗` 控制台字符**：原代码 print 里就有，在非 UTF-8 的 Windows GBK 控制台会 `UnicodeEncodeError`；Linux/Docker 目标无碍，本机测试加 `PYTHONUTF8=1`。
- **xgboost 必须 <3.1**（载入 swxy 旧二进制 `updown_concat_xgb.model`）。若想解锁版本：用兼容版把该模型转存 JSON 一次即可。
- **swxy 的 rag_tokenizer/term_weight/synonym 是中文(huqie)**：已用 NLTK 英文 shim 替 tokenizer；term_weight/synonym 词典是中文，英文需换资源或跳过。
- **API key 安全**：swxy 的 `retrieval2.py` 和某些 `.env` 里有硬编码真实 key；本项目做成公开/简历前务必吊销轮换 + 进 .gitignore + 用环境变量。

---

## 9. 评估口径提醒
- 若用 DeepSeek/云 API 加速开发迭代可以，但**最终评估必须在真正部署的模型(本地 Qwen3)上重跑**——方差/F1/分数都随模型变，云模型的数不能当本地系统的成绩。
- 证据 F1 的"真值"是强参考模型自报的 `used_chunk_ids`(silver standard)；表述用 "agreement with stronger reference model"，不是 "accuracy against ground truth"。
