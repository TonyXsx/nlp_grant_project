"""Retrieval helpers built on swxy's ``Dealer``.

`evidence_for_section` runs hybrid retrieval over the current-PDF in-memory
store to find the chunks most relevant to a rubric section (the "rubric signal =
query" idea). `fewshot_for_section` (Phase B) runs the same Dealer over the ES
corpus, excluding the current application.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent  # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deepdoc_engine.rag.nlp.search_v2 import Dealer  # noqa: E402

# Problem B: the synthetic "derived" chunks build_pool adds (must match the
# section names in pool/build_pool.py) are dimension-specific evidence — they
# paraphrase rubric vocabulary, so they over-match rubric-style queries for
# EVERY section. Restrict each to the rubric dimension it was built for; exclude
# it elsewhere.
_DERIVED_APPLICATION_CONTEXT = "Application Context"
_DERIVED_PLAIN_ENGLISH = "Plain English NLP Analysis"
_DERIVED_APPLICATION_FORM = "Application Form Analysis"
_DERIVED_SECTIONS = {
    _DERIVED_APPLICATION_CONTEXT, _DERIVED_PLAIN_ENGLISH, _DERIVED_APPLICATION_FORM,
}
# Which derived chunk(s) each rubric section_key is allowed to retrieve.
_ALLOWED_DERIVED_BY_SECTION_KEY = {
    "general": {_DERIVED_APPLICATION_CONTEXT},
    "proposed_research": {_DERIVED_PLAIN_ENGLISH},
    "application_form": {_DERIVED_APPLICATION_FORM},
    # training_development / sites_support / wpcc: no derived chunks
}


def _filter_derived(chunks: list[dict], section_key: str) -> list[dict]:
    """Drop derived meta-chunks that don't belong to this rubric dimension."""
    allowed = _ALLOWED_DERIVED_BY_SECTION_KEY.get(section_key, set())
    out = []
    for c in chunks:
        sec = c.get("docnm_kwd") or c.get("parser_section", "")
        if sec in _DERIVED_SECTIONS and sec not in allowed:
            continue
        out.append(c)
    return out


# Cap the query length: a long query bloats the ES query_string clause count
# (terms x fields) past ES's maxClauseCount, and adds little retrieval value.
_QUERY_MAX_WORDS = 50


def section_query(rubric_section: dict[str, Any]) -> str:
    """Build a concise retrieval query from a rubric section: its name + every
    sub-criterion name + signal text (definitions are omitted — they are long
    and dilute the query). Capped to a bounded number of words."""
    parts: list[str] = [rubric_section.get("human_name", "")]
    for sub in rubric_section.get("sub_criteria", []):
        parts.append(sub.get("name", ""))
        for sig in sub.get("signals", []):
            if sig.get("text"):
                parts.append(sig["text"])
    words = " ".join(p for p in parts if p).split()
    return " ".join(words[:_QUERY_MAX_WORDS]).strip()


def evidence_for_section(
    conn,
    index_name: str,
    rubric_section: dict[str, Any],
    *,
    top_k: int = 8,
    vector_similarity_weight: float = 0.7,
) -> list[dict]:
    """Top-k chunks from the current-PDF store for this rubric section.

    Returns Dealer chunk dicts (content_with_weight, doc_id, docnm_kwd,
    chunk_id, similarity, positions, ...).
    """
    query = section_query(rubric_section)
    if not query:
        return []
    dealer = Dealer(dataStore=conn)
    ranks = dealer.retrieval(
        question=query,
        embd_mdl=None,
        tenant_ids=index_name,
        kb_ids=None,
        page=1,
        page_size=top_k + len(_DERIVED_SECTIONS),  # over-fetch to absorb filtered derived chunks
        similarity_threshold=0.0,
        vector_similarity_weight=vector_similarity_weight,
        rank_feature=None,
    )
    chunks = _filter_derived(ranks.get("chunks", []), rubric_section.get("section_key", ""))
    return chunks[:top_k]


def fewshot_for_section(
    conn,
    index_name: str,
    rubric_section: dict[str, Any],
    *,
    current_app_id: str,
    n: int = 2,
    success_label: str = "successful",
    vector_similarity_weight: float = 0.7,
) -> list[dict]:
    """A few exemplar chunks from the ES corpus for this rubric section,
    EXCLUDING the current application (no label/self leakage), preferring the
    given success_label. Phase B (requires a populated ES corpus).
    """
    query = section_query(rubric_section)
    if not query:
        return []
    dealer = Dealer(dataStore=conn)
    ranks = dealer.retrieval(
        question=query,
        embd_mdl=None,
        tenant_ids=index_name,
        kb_ids=None,
        page=1,
        page_size=max(n * 4, 8),  # over-fetch, then filter
        similarity_threshold=0.0,
        vector_similarity_weight=vector_similarity_weight,
        rank_feature=None,
    )
    candidates = _filter_derived(ranks.get("chunks", []), rubric_section.get("section_key", ""))
    out: list[dict] = []
    for ch in candidates:
        if ch.get("doc_id") == current_app_id:
            continue  # exclude the application being scored
        if success_label and ch.get("success_label") not in (None, success_label):
            continue
        out.append(ch)
        if len(out) >= n:
            break
    return out
