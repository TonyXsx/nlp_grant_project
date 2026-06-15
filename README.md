# NIHR Grant Application Automated Scoring System

This project implements a structured scoring pipeline for NIHR grant applications, backed by a configurable large language model (a local model through Ollama, or the DeepSeek API). The input is a grant application document (PDF, DOCX, or PPT). The output is a structured JSON file containing rubric-level scores, evidence sources with page references, strengths, limitations, and an overall score.

Beyond scoring, the system now includes:

- A **vendored DeepDOC parsing engine** (layout recognition, OCR, table structure, block concatenation) as a robust fallback for documents the rule-based parsers cannot handle, and as the primary parser for DOCX/PPT.
- A **hybrid retrieval layer** (BM25 + dense vectors + reranking) over two stores — an ephemeral in-memory index of the current application (drives evidence selection during scoring) and a persistent Elasticsearch corpus of labelled applications (supplies few-shot calibration examples).
- A grounded **Question & Answer chat** on the result page, so a reviewer, applicant, or committee member can ask why a section scored as it did, what the application says, or how it compares to other funded applications — each answer cited back to the source chunks.

---

## Core Idea

### Why not send the full application directly to the model?

NIHR applications can contain tens of thousands of words, often exceeding or crowding the model context window. Even when the full text fits, the model may struggle to check each scoring signal against the right evidence because attention is spread across the entire document.

This system addresses that problem by first splitting the application into a traceable chunk pool, then using a two-stage pipeline for evidence discovery and final scoring. The model only sees the context most relevant to the current scoring dimension. For each rubric section, that context is selected by **hybrid retrieval** (lexical BM25 + dense vector similarity, fused and reranked) over an index of the current application's chunks, with a safe fallback to the full application text when retrieval is unavailable.

---

## Pipeline Architecture

```text
Document (PDF / DOCX / PPT)
 |
 v
[1] Parser
 |   Rule-based parsers (Fellowship / RfPB / generic PDF), then DeepDOC as the
 |   final fallback. Accept the first result that passes a content-quality gate.
 |   Extract text by application section and generate structured JSON.
 |
 v
[2] Chunk Pool Construction
 |   Token-based chunking (default: ~256 tokens, sentence-respecting, heading-aware).
 |   Tables are kept as their own chunks. Each chunk carries a unique chunk_id, its
 |   source section, BM25 token fields, and DeepDOC page positions when available.
 |
 v
[3] Stage 1 - Belief Accumulation
 |   Scan the application section by section.
 |   Ask the model to identify evidence chunks relevant to each rubric sub-criterion.
 |   Output good/bad evidence chunk IDs and short implications.
 |   Merge results into a global belief_state.
 |
 v
[4] Hybrid Evidence Retrieval (per rubric dimension)
 |   Index the current application's chunks in an ephemeral in-memory store.
 |   Retrieve the most relevant chunks for each rubric section (BM25 + dense
 |   vectors, fused and reranked). Fall back to full text if retrieval is empty.
 |   Optionally pull few-shot calibration examples from the Elasticsearch corpus.
 |
 v
[5] Stage 2 - Final Scoring
 |   Build a scoped version of the application for each rubric section.
 |   Ask the model to score each signal from 0 to 5.
 |   Return pros, drawbacks, and grounded evidence IDs.
 |
 v
[6] Aggregated Output
     signal -> sub_criterion -> section -> overall
     Compute weighted averages and apply doc_type-specific exclusions.
```

---

## Key Modules

### Parser Layer

Located in `src/all_type_parser/`. The parser automatically detects the document type and routes the file to the appropriate parser.

| File | Target format | doc_type |
|---|---|---|
| `fellowships_parser.py` | NIHR Fellowship applications (doctoral/postdoctoral) | `fellowship` |
| `RfPB_parser.py` | Research for Patient Benefit Stage 2 applications | `rfpb` |
| `pdf_parser.py` | Generic blue-box PDF parser | `unknown` |
| `deepdoc_fallback.py` | DeepDOC engine wrapper for PDF / DOCX / PPT | `unknown` |

The parser output is a structured JSON object. The top level includes a `doc_type` field, which is later used for scoring adaptation.

**Routing and the content-quality gate.** For PDFs, the rule-based parsers are tried in order, and the first result that passes a **content-quality gate** is accepted — not merely the first non-empty one. The gate (`_parse_is_good`) requires a minimum amount of real content (default: ≥ 1000 content characters across ≥ 1 section), which prevents a parser that only recovered a title block from silently "succeeding" and starving the scorer of evidence. If no rule-based parser passes the gate, the **DeepDOC engine** is used as the final fallback; if nothing passes, the richest non-empty result seen is returned. DOCX and PPT files are parsed directly by the DeepDOC engine.

Key layout differences between RfPB and Fellowship applications:

- Fellowship applications usually have fixed blue boxes near the top of each page, making section boundaries relatively clear.
- RfPB Stage 2 applications may place blue boxes at variable positions, and a single page can contain multiple sections, so line-level section detection is required.

#### DeepDOC parsing engine (`src/deepdoc_engine/`)

A self-contained, vendored copy of the DeepDOC document-understanding stack (adapted from a mature RAG codebase). It runs entirely offline from bundled ONNX/XGBoost model weights and provides:

- **Layout recognition** — a vision model labels each page region (title, text, table, figure, …).
- **OCR** — text detection + recognition for scanned or image-based pages.
- **Table structure recognition** — reconstructs table cells and emits HTML tables, which become standalone chunks.
- **Block concatenation** — an XGBoost model decides where consecutive blocks should be merged into continuous text, and records per-block page positions.

The engine is wrapped by `deepdoc_fallback.py`, which emits the same structured-JSON contract as the rule-based parsers (so the chunk pool consumes it unchanged). Its English tokenisation is provided by an NLTK-based shim that mirrors the original interface.

---

### Chunk Pool (`src/pool/build_pool.py`)

The chunk pool converts the parsed JSON into traceable evidence units. Chunking is **token-based** (default ~256 tokens), sentence-respecting, and heading-aware: it does not cut sentences mid-way, and it keeps numbered/bulleted headings attached to the text they introduce. Tables are restored as their own chunks (HTML kept intact) so budget and structural analysis can read them. Each chunk stores:

- `chunk_id`: unique ID, such as `secdrp__001_a`
- `parser_section`: source section, such as `Detailed Research Plan`
- `source_path`: original JSON path, such as `APPLICATION DETAILS > Detailed Research Plan`
- `content_ltks` / `content_sm_ltks` / `title_tks`: tokenised fields used for BM25 lexical retrieval
- `token_count`, `is_table`: chunk metadata
- `position`: DeepDOC page positions (page-level, used to show page references in the UI) when the document was parsed by DeepDOC; empty for rule-based parses

The pool also adds derived chunks:

- **Application Context**: application-level metadata such as title, applicant, organisation, and other contextual fields.
- **Plain English NLP Analysis**: readability and plain-English indicators, including sentence length, Flesch-Kincaid estimates, jargon proxy density, content coverage, and lexical overlap with the detailed research plan.
- **Application Form Analysis**: structural indicators derived from parser output, including word counts, list markers, table-like budget lines, repeated text, transition phrases, and cross-section lexical overlap. This chunk supports the application form quality dimension.

---

### Stage 1 - Belief Accumulation

**Motivation:** The same passage can be relevant to multiple scoring dimensions. For example, a CV section can support both research experience and leadership trajectory. Stage 1 scans the document first and builds a mapping from rubric sub-criteria to supporting or negative evidence chunks.

**Execution:**

- Iterate through parsed application sections, excluding the derived Application Form scoring chunk.
- For each section, send the current section chunks, the current global belief state, and the rubric to the model.
- The model returns findings in this shape:

  ```json
  {
    "sub_id": "g.4",
    "evidence": {
      "good_evidence_ids": ["secac__001_a"],
      "bad_evidence_ids": []
    },
    "implication": "The CV lists publications and prior funded projects, supporting research output quality."
  }
  ```

- Findings are merged by `sub_id` into `belief_state.subcriteria_beliefs`.
- Evidence accumulates across sections and is reused during final scoring.

**Flattened design:** Each finding maps directly to a parent `sub_id` rather than nesting at the signal level. This reduces token cost and allows Stage 1 to cover more rubric items within the context budget.

---

### Hybrid Evidence Retrieval

Final scoring does not reuse the same full-document prompt for every rubric section. Instead, the system retrieves a scoped context dynamically for each scoring dimension.

**Two stores, one retriever.** The retrieval layer (`src/retrieval/`, `src/deepdoc_engine/rag/`) reuses a single hybrid retriever (`Dealer`) over two backends:

| Store | Backend | Lifetime | Role |
|---|---|---|---|
| Current-application index | In-memory (`InMemoryConnection`) | Ephemeral (per request) | Evidence selection during scoring + single-document QA |
| Labelled corpus | Elasticsearch (`grant_corpus`) | Persistent | Few-shot calibration examples + cross-application QA |

Both run the same fusion pipeline: lexical **BM25** over the tokenised fields + **dense vector** cosine similarity (local `BAAI/bge-small-en-v1.5`, 384-dim), fused by a weighted sum, then reranked by a local cross-encoder. The in-memory store reproduces the Elasticsearch hybrid search in NumPy so the scoring path needs no external service.

**Per-dimension retrieval.** For each rubric section, a query is built from the section / sub-criterion / signal text, and the most relevant chunks of the current application are retrieved. Derived chunks are scoped to the dimension they were synthesised for (Application Context → `general`, Plain English NLP Analysis → `proposed_research`, Application Form Analysis → `application_form`). If retrieval returns nothing (or embeddings are unavailable), the system **falls back to the full application text**, preserving the original behaviour.

**Few-shot calibration (optional).** When an Elasticsearch corpus index is configured, Stage 2 also pulls a small number of same-dimension example chunks from **successful** applications (always excluding the current application) and injects them as calibration anchors — they only illustrate what a strong answer looks like for the 0–5 scale; they are never used as evidence for the current application. The corpus is built once with `python -m src.retrieval.indexer --recreate`.

---

### Stage 2 - Final Scoring

**Execution:**

- For each major rubric section, retrieve the most relevant chunks of the current application via hybrid retrieval (with full-text fallback).
- Send the scoped application text, the target rubric section, the final belief state, and any few-shot calibration examples to the model.
- The model scores every signal from 0 to 5 and returns:
  - `used_chunk_ids`
  - `pros`
  - `drawbacks`
  - signal-level scores

The model output is constrained by JSON schemas so that all required signals are scored and all score values are valid integers from 0 to 5.

`run_info.retrieval_method` records whether hybrid retrieval or the full-text fallback was used, and `debug.retrieval_used` / `debug.fewshot_used` flag whether each was active for the run.

---

### Scoring Rubric (`criteria_points.json`)

The rubric contains six major scoring sections. Each section contains sub-criteria, and each sub-criterion contains signal-level scoring items.

| Section | key | Description |
|---|---|---|
| General | `general` | Applicant experience, outputs, leadership potential, and career trajectory |
| Proposed Research | `proposed_research` | Research question, design, methodological rigour, feasibility, impact, and resources |
| Training and Development | `training_development` | Training plan, mentorship, career development, and skill acquisition |
| Sites and Support | `sites_support` | Institutional capability, supervision, infrastructure, and research culture |
| Working with People and Communities | `wpcc` | PPI/WPCC design, representativeness, depth of involvement, and feedback mechanisms |
| Application Form | `application_form` | Structural completeness, logical flow, formatting, repetition, and coherence |

Score aggregation path:

```text
signal score (0-5)
 -> weighted sub_criterion average
 -> sub_criterion score (0-10)
 -> weighted section average
 -> overall score
```

---

### `doc_type` Adaptation

Different NIHR application types do not share exactly the same scoring expectations. The pipeline uses `doc_type` to exclude criteria that are not applicable to a given application format.

#### RfPB (Research for Patient Benefit)

RfPB is a project grant rather than an individual career-development fellowship. Compared with Fellowship applications:

- It does not require a personal career-development narrative.
- It does not contain a training and development section in the Fellowship sense.
- Training-related Fellowship criteria should not reduce the score of an RfPB application.

For `doc_type=rfpb`, the pipeline applies these exclusions:

| Excluded item | Exclusion scope | Reason |
|---|---|---|
| `g.1` Common Characteristics of Good Applications | Excluded from the General section average | Contains Fellowship-oriented signals such as training plan quality |
| `g.2` Tell Us Why You Need This Award | Excluded from the General section average | Asks about personal career-development need, which is not applicable to a project grant |
| `training_development` | Excluded from the overall score | RfPB applications do not have a matching Fellowship-style training section |

Excluded sub-criteria can still appear in the output with `excluded_reason: "not_applicable_for_doc_type"`. They are visible for review but do not affect the averaged scores.

---

## Scoring Backends

The scorer is decoupled from the LLM client behind a small interface (`generate_json(messages, schema, max_tokens)`), so the backend is selectable at runtime via the `SCORER_BACKEND` environment variable:

| `SCORER_BACKEND` | Backend | Configuration |
|---|---|---|
| `ollama` (default) | Local model through Ollama | `OLLAMA_HOST`, `OLLAMA_MODEL` |
| `deepseek` | DeepSeek (OpenAI-compatible API) | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL` |

Configuration is read from a project-local `.env` file (loaded automatically when `python-dotenv` is installed). `.env` is gitignored — never commit API keys. The local Ollama path keeps the system fully offline; the DeepSeek path is convenient for fast iteration and for hosts without a GPU.

> Evaluation note: numbers reported for the deployed system should be produced on the model it actually runs (local Qwen3). Cloud-API runs are useful for development but their scores/variance do not transfer verbatim to the local model.

---

## Question Answering (QA) Chat

The result page includes a grounded chat assistant (`src/qa/qa_service.py`, exposed via the web server's `/ask` endpoint). It reuses the same hybrid retriever and stores as the scorer, so answers are cited back to source chunks (with section and page references).

**Intent routing (one LLM call).** Each question is routed into a mode, with a standalone retrieval query rewritten from the conversation, and — for scoring questions — the rubric section(s) the question targets:

| Mode | Trigger | Context fed to the model | Retrieval |
|---|---|---|---|
| `single_doc` | About the application's content | Top chunks of the current application | In-memory store |
| `scoring` | About the assessment (why a score, strengths/weaknesses, evidence) | The scored result, scoped to the targeted section(s) | None (uses stored assessment) |
| `corpus` | Comparison across other applications | Chunks from other **successful** applications | Elasticsearch corpus |

**Section-scoped scoring answers.** For scoring questions, the router names the section(s) involved (one, several, or none for whole-assessment questions). Only those sections of the assessment are serialised into the prompt, so a question about, say, *Sites and Support* sees that section's scores, strengths, weaknesses, and evidence — instead of a truncated dump of the whole assessment. Citations are deduplicated (the same chunk reuses one `[n]`) and capped.

**Role gating.** Cross-application (`corpus`) questions require a committee/admin role; the corpus retrieval always excludes the current application. The corpus modes require a configured Elasticsearch index (see below); the `single_doc` and `scoring` modes work without it.

**UI.** The chat panel is draggable and resizable. Citations and evidence chunks show a sentence-boundary preview and expand to the full chunk on click.

---

## Directory Structure

```text
.
|-- criteria_points.json          # Rubric definition
|-- qwen3_ollama.py               # Main scoring entry point (Ollama / DeepSeek)
|-- score_experiments.ipynb       # Score stability experiments
|-- start.sh                      # One-shot web server launcher (venv + deps + run)
|-- .env                          # Backend / API config (gitignored)
|-- src/
|   |-- all_type_parser/
|   |   |-- all_type_parser.py    # Parser router (+ content-quality gate)
|   |   |-- fellowships_parser.py # Fellowship PDF parser
|   |   |-- RfPB_parser.py        # RfPB Stage 2 PDF parser
|   |   |-- pdf_parser.py         # Generic PDF parser
|   |   |-- pdf_utils.py          # PDF utility functions
|   |   |-- deepdoc_fallback.py   # DeepDOC engine wrapper (PDF/DOCX/PPT)
|   |-- deepdoc_engine/           # Vendored DeepDOC stack (parsers, vision, rag/nlp)
|   |-- pool/
|   |   |-- build_pool.py         # Token-based chunk pool construction
|   |-- retrieval/
|   |   |-- indexer.py            # Build in-memory / ES indexes from the chunk pool
|   |   |-- retriever.py          # Per-section evidence + few-shot retrieval
|   |-- scoring/
|   |   |-- pipeline.py           # Two-stage scoring pipeline
|   |   |-- api_scorer.py         # DeepSeek (OpenAI-compatible) scorer
|   |-- qa/
|       |-- qa_service.py         # Grounded QA: intent router + 3 answer modes
|-- web/
|   |-- server.py                 # Flask server (upload, scoring, /ask, results)
|   |-- public/                   # UI (upload + result pages, QA chat widget)
|-- data/
|   |-- successful/               # Example successful application PDFs
|   |-- unsuccessful/             # Example unsuccessful application PDFs
|   |-- experiments/              # Experiment datasets
```

---

## Usage

The recommended way to run the full pipeline is the **end-to-end web server on a Linux/GPU host** (RunPod, EC2, university cluster, or any Ubuntu/Debian machine). The steps below mirror `Runpod_Instruction.ipynb` exactly and take you from a fresh pod to a browser-accessible scoring service.

### A. Run the web pipeline on a fresh Linux/RunPod host

Run each block in a shell on the host (RunPod web terminal, SSH, or local Linux). Pick a pod template that exposes HTTP port `8000`.

**Step 1 — Install OS packages**

```bash
sed -i 's|http://archive.ubuntu.com/ubuntu|https://mirrors.aliyun.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources
apt-get update && apt-get install -y git python3 python3-venv python3-pip curl
```

(The `sed` line swaps to a faster apt mirror; remove it if you are outside mainland China or on a non-Ubuntu base image.)

**Step 2 — Install Ollama and pull the model**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
sleep 3
ollama pull qwen3.5:27b
```

`ollama serve &` starts the model server in the background. The pull is ~17 GB and takes a few minutes on a fast network. Switch the model tag if you want a smaller footprint (e.g. `qwen3:4b`).

**Step 3 — Clone the project**

```bash
git clone https://github.com/ZLF329/nlp_grant_coursework.git
cd nlp_grant_coursework
```

**Step 4 — Launch the web server**

```bash
chmod +x start.sh
PORT=8000 ./start.sh
```

`start.sh` creates a virtual environment, installs Python dependencies, downloads the spaCy `en_core_web_sm` model and NLTK `punkt` data, then starts the Flask server on port 8000.

**Step 5 — Open the UI**

- Locally: navigate to `http://localhost:8000`.
- On RunPod: in the pod's **Connect** panel, pick the HTTP service exposed on port 8000 to get a public proxy URL such as `https://<pod-id>-8000.proxy.runpod.net`.

Upload a PDF, watch the progress bar through the three pipeline stages, and view the scored result.

**`start.sh` configurable environment variables**

| Variable      | Default                       | Purpose                                       |
|---------------|-------------------------------|-----------------------------------------------|
| `PORT`        | `8000`                        | HTTP port for the Flask server                |
| `PYTHON`      | `python3`                     | Python interpreter used to create the venv    |
| `VENV_DIR`    | `.venv`                       | Path of the virtual environment               |
| `OLLAMA_HOST` | `http://127.0.0.1:11434`      | URL of the Ollama service (read by the app)   |
| `OLLAMA_MODEL`| `qwen3.5:27b`                 | Model name passed to Ollama                   |

The script is idempotent: subsequent runs reuse the venv and skip dependency installation (a `.deps_installed` marker is written inside the venv).

---

### B. Optional — direct CLI usage of the parser and scorer

If you want to call the underlying components without the web server (e.g. for batch processing or experiments), they are exposed as plain Python modules.

**Parse a PDF**

```bash
python -m src.all_type_parser.all_type_parser path/to/application.pdf
```

The parser auto-detects the document type where possible and writes a structured JSON file under a `json_data/` directory next to the input PDF.

**Score a parsed application**

Requires a running Ollama server with the selected model available locally.

```bash
OLLAMA_MODEL=qwen3.5:27b python qwen3_ollama.py \
  data/successful/json_data/IC00029_RfPB.json \
  --criteria criteria_points.json \
  --out data/successful/json_data/IC00029_RfPB_scored.json
```

**Run experiments**

Use `score_experiments.ipynb` for scoring experiments, including:

- repeated scoring of the same PDF to inspect variance
- A/B group comparisons between application sets
- score distribution and hypothesis-test exploration

---

### C. Optional — DeepSeek backend and the Elasticsearch corpus

**Use the DeepSeek API instead of local Ollama.** Create a `.env` in the project root (it is gitignored):

```bash
SCORER_BACKEND=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

With this set, both scoring and the QA chat use DeepSeek; no Ollama is required. Remove the file (or set `SCORER_BACKEND=ollama`) to go back to the local model.

**Enable the labelled corpus** (powers few-shot calibration during scoring and committee/cross-application QA). Start an Elasticsearch instance and build the index once:

```bash
# 1. Elasticsearch 8.x (single node, security off for local use)
docker run -d --name grant-es -p 9200:9200 \
  -e discovery.type=single-node -e xpack.security.enabled=false \
  elasticsearch:8.11.3

# 2. Build the corpus from the labelled example sets
ES_HOST=http://localhost:9200 python -m src.retrieval.indexer --recreate
```

Then point the app at it (in `.env` or the environment):

```bash
GRANT_CORPUS_INDEX=grant_corpus
ES_HOST=http://localhost:9200
```

The corpus is entirely optional: with it absent, scoring still runs (without few-shot) and the QA chat still answers `single_doc` and `scoring` questions; only `corpus` (committee) questions are disabled.

**Relevant environment variables**

| Variable | Default | Purpose |
|---|---|---|
| `SCORER_BACKEND` | `ollama` | Scoring/QA backend: `ollama` or `deepseek` |
| `GRANT_USE_RETRIEVAL` | `1` | Set `0` to disable hybrid retrieval (full-text scoring) |
| `GRANT_CORPUS_INDEX` | _(unset)_ | Elasticsearch index name to enable the corpus features |
| `ES_HOST` | `http://localhost:9200` | Elasticsearch endpoint |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local sentence-transformers embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Local cross-encoder reranker |

> On Windows, run the server/CLI with `PYTHONUTF8=1` so console glyphs in log output do not raise `UnicodeEncodeError`.

---

## macOS — Native Ollama (recommended on Mac, much faster than Docker)

> **Why not Docker on Mac?** Docker Desktop on macOS runs containers inside a Linux VM, which **cannot access the Mac's Metal GPU**. LLM inference is forced onto CPU and ends up 5–10× slower than the native Ollama app. On Apple Silicon a single PDF can take 30–60 minutes through Docker; through native Ollama it typically finishes in 2–10 minutes (depending on model size).

Use this path when running on a MacBook (Intel or Apple Silicon). The web server still runs locally as a Python process — only Ollama is moved out of the container.

### Step 1 — Install native Ollama

Either route works:

```bash
# Homebrew
brew install ollama

# or download the .dmg from https://ollama.com/download and drag to Applications
```

### Step 2 — Start Ollama and pull the model

```bash
ollama serve &
ollama pull qwen3.5:27b      # paper configuration, ~17 GB
# or, for faster local testing:
ollama pull qwen3:4b         # ~2.6 GB
```

You can verify the model is loaded with:

```bash
ollama list
curl http://localhost:11434/api/tags
```

### Step 3 — Install OS-level dependencies for the parser

The parser uses `pdf2image` (poppler) and `pytesseract` (tesseract):

```bash
brew install poppler tesseract
```

### Step 4 — Launch the web server

```bash
cd /path/to/nlp_grant_coursework
chmod +x start.sh
./start.sh                                  # uses qwen3.5:27b by default
# or pick a different model:
OLLAMA_MODEL=qwen3:4b ./start.sh
```

`start.sh` creates a virtualenv on first run, installs dependencies, then serves at `http://localhost:8000`.

### Step 5 — Use it

Open `http://localhost:8000` in a browser, upload a PDF, and watch the three pipeline stages complete. Inference runs on Metal GPU automatically — no flags needed.

### Stopping

- Web server: `Ctrl+C` in its terminal
- Ollama background process: `pkill ollama` (or close the menu-bar Ollama app if installed via .dmg)

---

## Docker Deployment (All-in-One)

> **Note for macOS users:** prefer the **macOS — Native Ollama** section above. Docker on Mac forces CPU-only inference, which is several times slower than native Ollama on the same hardware. Use Docker on Mac only to verify the image builds correctly before delivering it to a Linux/GPU host.

The image bundles the Flask web server, all Python dependencies, the Ollama runtime, and the LLM weights, so a single `docker run` is enough to start the full pipeline.

### Files

- `Dockerfile` — image definition
- `docker-entrypoint.sh` — starts Ollama in the background, waits for it, pulls the model on first start if needed, then launches the web server
- `.dockerignore` — keeps notebooks, datasets, and dev artefacts out of the image

### Build arguments

| Build arg     | Default        | Purpose                                                      |
|---------------|----------------|--------------------------------------------------------------|
| `OLLAMA_MODEL`| `qwen3.5:27b`  | Ollama model tag baked into the image and used at runtime    |
| `BAKE_MODEL`  | `1`            | If `1`, pre-download the model during build (offline-ready). If `0`, pull on first container start instead. |

---

### Path A — Production image with `qwen3.5:27b` (paper configuration)

This is the configuration used in the paper and the recommended build to deliver to the marker. The image is fully self-contained (no network needed at runtime) but is large.

**Requirements**
- Disk: ~25 GB during build, final image ~20 GB
- Network: must reach `ollama.com` (model download) and Docker Hub during build
- For inference: a GPU host is strongly recommended; CPU-only is far too slow for `27b`

**Build**

```bash
docker build -t grant-ai .
```

(Default args, equivalent to `--build-arg OLLAMA_MODEL=qwen3.5:27b --build-arg BAKE_MODEL=1`. Expect 30–60 min depending on network.)

**Run (CPU)**

```bash
docker run --rm -p 8000:8000 grant-ai
```

**Run (NVIDIA GPU)**

```bash
docker run --rm --gpus all -p 8000:8000 grant-ai
```

Requires the NVIDIA Container Toolkit on the host. For maximum throughput, rebuild from a CUDA base image (e.g. `nvidia/cuda:12.4.0-runtime-ubuntu22.04`) instead of `python:3.11-slim`.

**Persist uploads and results across restarts**

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/data:/app/data" grant-ai
```

**Distribute the image as a single offline file**

```bash
# on the build host
docker save grant-ai | gzip > grant-ai.tar.gz
# on the target machine
gunzip -c grant-ai.tar.gz | docker load
docker run --rm -p 8000:8000 grant-ai
```

Open `http://localhost:8000` after the container reports `Grant AI pipeline listening on http://0.0.0.0:8000`.

---

### Path B — Lightweight Mac local test with `qwen3:4b`

Use this to verify the Docker pipeline end-to-end on a laptop (especially Apple Silicon, where the `27b` model is too slow for interactive testing).

> Note: there is no `qwen3.5:4b` in the Ollama registry. The 4-billion-parameter option in the Qwen3 family is `qwen3:4b` (~2.6 GB).

**Requirements**
- Disk: ~10 GB during build, final image ~8 GB
- Docker Desktop running
- Network access during build

**Build**

```bash
docker build -t grant-ai-test --build-arg OLLAMA_MODEL=qwen3:4b .
```

(Takes roughly 5–15 min; most of the time is `pip install` and the 2.6 GB model pull. The pip layer is cached on rebuilds.)

**Run**

```bash
docker run --rm -p 8000:8000 grant-ai-test
```

Then open `http://localhost:8000`.

**Override the model at runtime (optional)**

```bash
docker run --rm -p 8000:8000 -e OLLAMA_MODEL=qwen3:8b grant-ai-test
```

If the requested model is not already inside the image, the entrypoint pulls it on first start (requires network).

---

### Common operations

**Stop a running container**

```bash
docker stop grant-ai            # by name (only if you used --name)
# or find it:
docker ps                       # list running containers
docker stop <container_id>
```

If you started the container with `--rm` and **without** `-d`, it runs in the foreground — just press `Ctrl+C` to stop it.

**Remove the image**

```bash
docker rmi grant-ai-test
```

**Inspect the container interactively (for debugging)**

```bash
docker run --rm -it grant-ai-test bash
# inside:
ollama list      # confirm the baked model is present
ls /app          # confirm project files
```

---

## Output Structure

The top-level scored JSON has the following shape:

```json
{
  "doc_id": "IC00029_RfPB",
  "run_info": {
    "scorer_model": "qwen3.5:27b",
    "retrieval_method": "belief_then_inmem_hybrid_retrieval_scoring",
    "ran_at_utc": "..."
  },
  "pool_lookup": {
    "chunk_id": {
      "text": "...",
      "parser_section": "..."
    }
  },
  "belief_state": {
    "subcriteria_beliefs": {
      "pr.1": {
        "good_evidence_ids": ["..."],
        "bad_evidence_ids": []
      }
    }
  },
  "features": {
    "general": {
      "score_10": 7.33,
      "sub_criteria": [
        {
          "sub_id": "g.3",
          "score_10": 8.0,
          "counts_toward_section_average": true,
          "signals": [
            {
              "sid": "g.3.a",
              "score": 4
            }
          ],
          "pros": "...",
          "drawbacks": "...",
          "evidence": [
            {
              "id": "secac__001_a",
              "text": "...",
              "section": "Applicant CV",
              "pages": [3]
            }
          ]
        }
      ]
    }
  },
  "overall": {
    "score_10": 8.44,
    "final_score_0to100": 84.4
  },
  "debug": {
    "doc_type": "rfpb",
    "excluded_sections": ["training_development"],
    "excluded_sub_ids": ["g.1", "g.2"],
    "retrieval_used": true,
    "fewshot_used": false
  }
}
```

The output is designed to be auditable: each score can be inspected together with the evidence chunks and model rationale used to produce it.
