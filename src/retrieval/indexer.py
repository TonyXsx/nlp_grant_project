"""Build searchable indexes from a parsed application.

`build_current_index` produces the ephemeral in-memory store for the current
PDF (used for evidence retrieval during scoring). It mirrors swxy's
`file_parse.execute_insert_process` flow — chunk → embed → assemble ES-style
docs → insert — but targets `InMemoryConnection` and reuses our enriched
`build_chunk_pool` output (which already carries `content_ltks` etc.).
"""
from __future__ import annotations

import sys
import xxhash
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent  # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deepdoc_engine.rag.nlp.model import generate_embedding, embedding_dim  # noqa: E402
from deepdoc_engine.rag.utils.inmem_conn import InMemoryConnection  # noqa: E402

CURRENT_INDEX = "current_pdf"


def _assemble_docs(pool_lookup: dict[str, dict], embeddings: list, dim: int,
                   doc_id: str) -> list[dict]:
    """Turn pool chunks + their vectors into swxy-style ES/in-mem docs."""
    docs: list[dict] = []
    vec_field = f"q_{dim}_vec"
    for (chunk_id, meta), vec in zip(pool_lookup.items(), embeddings):
        if vec is None:
            continue
        docs.append({
            "id": chunk_id,
            "content_with_weight": meta["text"],
            "content_ltks": meta.get("content_ltks", ""),
            "content_sm_ltks": meta.get("content_sm_ltks", ""),
            "title_tks": meta.get("title_tks", ""),
            "important_kwd": [],
            "docnm_kwd": meta.get("parser_section", ""),
            "doc_id": doc_id,
            "kb_id": CURRENT_INDEX,
            "available_int": 1,
            "position_int": meta.get("position") or [],
            "parser_section": meta.get("parser_section", ""),
            "is_table": meta.get("is_table", False),
            vec_field: vec,
        })
    return docs


def build_index_from_pool(pool_lookup: dict[str, dict], doc_id: str = "current",
                          index_name: str = CURRENT_INDEX
                          ) -> tuple[InMemoryConnection, str, int]:
    """Embed an already-built chunk pool into a fresh in-memory store.

    Lets callers that already ran ``build_chunk_pool`` (e.g. the scoring
    pipeline) avoid re-chunking. Returns (connection, index_name, vector_dim).
    """
    conn = InMemoryConnection()
    if not pool_lookup:
        conn.createIdx(index_name)
        return conn, index_name, embedding_dim()

    texts = [meta["text"] for meta in pool_lookup.values()]
    embeddings = generate_embedding(texts)
    dim = len(embeddings[0]) if embeddings and embeddings[0] is not None else embedding_dim()

    docs = _assemble_docs(pool_lookup, embeddings, dim, doc_id)
    conn.createIdx(index_name, vectorSize=dim)
    conn.insert(docs, index_name)
    return conn, index_name, dim


def build_current_index(application: dict[str, Any], doc_id: str = "current"
                        ) -> tuple[InMemoryConnection, str, int]:
    """Chunk + embed the current application into a fresh in-memory store.

    Returns (connection, index_name, vector_dim). The connection is ephemeral —
    drop the reference and it's gone.
    """
    from pool.build_pool import build_chunk_pool  # lazy: avoids re-loading build_pool when only build_index_from_pool is needed
    pool = build_chunk_pool(application)
    return build_index_from_pool(pool["pool_lookup"], doc_id=doc_id)
