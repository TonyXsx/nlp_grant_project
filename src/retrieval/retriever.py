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


def section_query(rubric_section: dict[str, Any]) -> str:
    """Build a retrieval query from a rubric section: its name + every
    sub-criterion name/definition + signal text. This is the section's
    information need expressed as one query."""
    parts: list[str] = [rubric_section.get("human_name", "")]
    for sub in rubric_section.get("sub_criteria", []):
        parts.append(sub.get("name", ""))
        if sub.get("definition"):
            parts.append(sub["definition"])
        for sig in sub.get("signals", []):
            if sig.get("text"):
                parts.append(sig["text"])
    return " ".join(p for p in parts if p).strip()


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
        page_size=top_k,
        similarity_threshold=0.0,
        vector_similarity_weight=vector_similarity_weight,
        rank_feature=None,
    )
    return ranks.get("chunks", [])


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
    out: list[dict] = []
    for ch in ranks.get("chunks", []):
        if ch.get("doc_id") == current_app_id:
            continue  # exclude the application being scored
        if success_label and ch.get("success_label") not in (None, success_label):
            continue
        out.append(ch)
        if len(out) >= n:
            break
    return out
