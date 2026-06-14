"""In-memory implementation of swxy's ``DocStoreConnection``.

Lets ``search_v2.Dealer`` run its full hybrid-retrieval pipeline (BM25 + dense +
weighted fusion, then rerank) over an in-process corpus — no Elasticsearch.
Used for the ephemeral "current-PDF" store: build it per request, throw it away
on disconnect.

It implements only what ``Dealer.search`` / ``Dealer.retrieval`` actually call:
``createIdx/deleteIdx/indexExist/insert/get`` + ``search`` + the result helpers
``getTotal/getChunkIds/getFields/getHighlight/getAggregation``. ``search``
reproduces, in numpy, what ``ESConnection.search`` delegates to Elasticsearch:
filter → BM25 text score → cosine dense score → weighted-sum fusion → top-N.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from deepdoc_engine.rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchTextExpr,
    MatchDenseExpr,
    FusionExpr,
    OrderByExpr,
)

# Extract candidate query terms from an ES query_string (e.g. "strong^0.18
# \"firm\"^0.04 (applicant^0.18)"): keep alphabetic words, drop the ^weights,
# operators and punctuation.
_TERM_RE = re.compile(r"[a-z][a-z0-9_]+")
_STOP_OPS = {"and", "or", "not"}


def _query_terms(matching_text: str) -> list[str]:
    if not matching_text:
        return []
    terms = [t for t in _TERM_RE.findall(matching_text.lower()) if t not in _STOP_OPS]
    # dedupe, preserve order
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _bm25(query_terms: list[str], docs_tokens: list[list[str]],
          k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    n = len(docs_tokens)
    if n == 0 or not query_terms:
        return np.zeros(n, dtype=float)
    qset = set(query_terms)
    df: dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks) & qset:
            df[t] = df.get(t, 0) + 1
    avgdl = max(1e-9, sum(len(t) for t in docs_tokens) / n)
    scores = np.zeros(n, dtype=float)
    for i, toks in enumerate(docs_tokens):
        if not toks:
            continue
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for t in query_terms:
            f = tf.get(t, 0)
            if not f:
                continue
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


class _Result:
    """Lightweight search result the getX helpers read from."""

    def __init__(self, ids, docs, total, query_vector=None):
        self.ids = ids
        self.docs = docs                # {chunk_id: doc_dict}
        self.total = total
        self.query_vector = query_vector


class InMemoryConnection(DocStoreConnection):
    def __init__(self):
        self._idx: dict[str, dict[str, dict]] = {}

    # ── db ────────────────────────────────────────────────────────────────
    def dbType(self) -> str:
        return "inmemory"

    def health(self) -> dict:
        return {"type": "inmemory", "status": "green",
                "indices": list(self._idx.keys())}

    # ── index lifecycle ───────────────────────────────────────────────────
    def createIdx(self, indexName: str, knowledgebaseId: str = None, vectorSize: int = 0):
        self._idx.setdefault(indexName, {})

    def deleteIdx(self, indexName: str, knowledgebaseId: str = None):
        self._idx.pop(indexName, None)

    def indexExist(self, indexName: str, knowledgebaseId: str = None) -> bool:
        return indexName in self._idx

    # ── crud ──────────────────────────────────────────────────────────────
    def insert(self, rows: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        bucket = self._idx.setdefault(indexName, {})
        for row in rows:
            cid = row.get("id")
            if cid is None:
                continue
            bucket[cid] = row
        return []

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str] = None) -> dict | None:
        return self._idx.get(indexName, {}).get(chunkId)

    def update(self, condition, newValue, indexName, knowledgebaseId=None) -> bool:
        return False  # not needed by Dealer.retrieval

    def delete(self, condition, indexName, knowledgebaseId=None) -> int:
        return 0  # not needed by Dealer.retrieval

    def sql(self, sql, fetch_size=128, format="json"):
        raise NotImplementedError("InMemoryConnection.sql not supported")

    # ── search (reproduces ES hybrid in numpy) ────────────────────────────
    def _gather(self, indexNames, knowledgebaseIds, condition) -> list[dict]:
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")
        docs: list[dict] = []
        for nm in indexNames:
            docs.extend(self._idx.get(nm, {}).values())

        kb_ids = knowledgebaseIds or []

        def keep(doc: dict) -> bool:
            if kb_ids and doc.get("kb_id") not in kb_ids:
                return False
            for k, v in (condition or {}).items():
                if k == "kb_id":
                    continue  # handled above
                if k == "available_int":
                    if v and int(doc.get("available_int", 1)) < 1:
                        return False
                    continue
                if not v:
                    continue
                dv = doc.get(k)
                if isinstance(v, list):
                    if dv not in v:
                        return False
                elif dv != v:
                    return False
            return True

        return [d for d in docs if keep(d)]

    def search(self, selectFields, highlightFields, condition, matchExprs,
               orderBy, offset, limit, indexNames, knowledgebaseIds,
               aggFields=[], rank_feature=None):
        docs = self._gather(indexNames, knowledgebaseIds, condition)
        if not docs:
            return _Result([], {}, 0)

        match_text = next((m for m in matchExprs if isinstance(m, MatchTextExpr)), None)
        match_dense = next((m for m in matchExprs if isinstance(m, MatchDenseExpr)), None)
        fusion = next((m for m in matchExprs if isinstance(m, FusionExpr)), None)

        n = len(docs)

        # text side (BM25 over content_ltks + title_tks)
        if match_text is not None:
            terms = _query_terms(match_text.matching_text)
            toks = [
                (d.get("content_ltks", "") + " " + d.get("title_tks", "")).split()
                for d in docs
            ]
            text_scores = _bm25(terms, toks)
            mx = float(text_scores.max()) if text_scores.size else 0.0
            text_norm = text_scores / mx if mx > 0 else text_scores
        else:
            text_norm = np.zeros(n, dtype=float)

        # dense side (cosine over the q_<dim>_vec column)
        q_vec = None
        if match_dense is not None:
            q_vec = np.asarray(match_dense.embedding_data, dtype=float)
            col = match_dense.vector_column_name
            dense = np.zeros(n, dtype=float)
            for i, d in enumerate(docs):
                dv = d.get(col)
                if dv is None:
                    continue
                dv = np.asarray(dv, dtype=float)
                if dv.shape != q_vec.shape:
                    continue
                dense[i] = max(0.0, float(np.dot(q_vec, dv)))  # vectors are L2-normalised
        else:
            dense = np.zeros(n, dtype=float)

        # weighted-sum fusion (ES default weights "0.05, 0.95")
        text_w, vec_w = 0.05, 0.95
        if fusion is not None and fusion.fusion_params:
            w = fusion.fusion_params.get("weights")
            if isinstance(w, str) and "," in w:
                try:
                    parts = [float(x) for x in w.split(",")]
                    text_w, vec_w = parts[0], parts[1]
                except ValueError:
                    pass
        if match_dense is None:
            fused = text_norm
        elif match_text is None:
            fused = dense
        else:
            fused = text_w * text_norm + vec_w * dense

        order = np.argsort(-fused)
        total = int((fused > 0).sum()) or n
        sliced = order[offset: offset + limit]
        ids = [docs[i]["id"] for i in sliced]
        out_docs = {docs[i]["id"]: docs[i] for i in sliced}
        return _Result(ids, out_docs, total, query_vector=q_vec)

    # ── result helpers ────────────────────────────────────────────────────
    def getTotal(self, res) -> int:
        return res.total

    def getChunkIds(self, res) -> list[str]:
        return list(res.ids)

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for cid in res.ids:
            doc = res.docs.get(cid, {})
            out[cid] = {f: doc[f] for f in fields if f in doc}
        return out

    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        return {}  # highlighting not supported in-memory

    def getAggregation(self, res, fieldnm: str):
        return []  # aggregations not needed by Dealer.retrieval
