"""Build searchable indexes from a parsed application.

`build_current_index` produces the ephemeral in-memory store for the current
PDF (used for evidence retrieval during scoring). It mirrors swxy's
`file_parse.execute_insert_process` flow — chunk → embed → assemble ES-style
docs → insert — but targets `InMemoryConnection` and reuses our enriched
`build_chunk_pool` output (which already carries `content_ltks` etc.).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent  # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from deepdoc_engine.rag.nlp.model import generate_embedding, embedding_dim  # noqa: E402
from deepdoc_engine.rag.utils.inmem_conn import InMemoryConnection  # noqa: E402

logger = logging.getLogger(__name__)

CURRENT_INDEX = "current_pdf"
CORPUS_INDEX = "grant_corpus"


def _assemble_docs(pool_lookup: dict[str, dict], embeddings: list, dim: int,
                   doc_id: str, kb_id: str = CURRENT_INDEX,
                   extra: dict | None = None, id_prefix: str = "") -> list[dict]:
    """Turn pool chunks + their vectors into swxy-style ES/in-mem docs.

    ``id_prefix`` keeps chunk ids unique across a multi-document corpus (chunk
    ids repeat per application); ``extra`` adds custom fields (e.g.
    ``success_label``).
    """
    docs: list[dict] = []
    vec_field = f"q_{dim}_vec"
    for (chunk_id, meta), vec in zip(pool_lookup.items(), embeddings):
        if vec is None:
            continue
        doc = {
            "id": f"{id_prefix}{chunk_id}",
            "content_with_weight": meta["text"],
            "content_ltks": meta.get("content_ltks", ""),
            "content_sm_ltks": meta.get("content_sm_ltks", ""),
            "title_tks": meta.get("title_tks", ""),
            "important_kwd": [],
            "docnm_kwd": meta.get("parser_section", ""),
            "doc_id": doc_id,
            "kb_id": kb_id,
            "available_int": 1,
            "position_int": meta.get("position") or [],
            "parser_section": meta.get("parser_section", ""),
            "is_table": meta.get("is_table", False),
            vec_field: vec,
        }
        if extra:
            doc.update(extra)
        docs.append(doc)
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


# ── persistent ES corpus (Phase B) ───────────────────────────────────────────

def _index_labeled_pdfs(es, index_name, directory, success_label):
    """Parse + chunk + embed every PDF in `directory` and bulk-insert into ES,
    tagged with `doc_id=<stem>` and the given `success_label`."""
    from pool.build_pool import build_chunk_pool
    from all_type_parser.all_type_parser import parse

    pdfs = sorted(Path(directory).glob("*.pdf"))
    logger.info("[corpus] %s: %d PDFs from %s", success_label, len(pdfs), directory)
    n_docs = 0
    for pdf in pdfs:
        app_id = pdf.stem
        try:
            application = parse(str(pdf))
            pool = build_chunk_pool(application)
            pool_lookup = pool["pool_lookup"]
            if not pool_lookup:
                logger.warning("[corpus] %s: empty pool, skipped", app_id)
                continue
            texts = [m["text"] for m in pool_lookup.values()]
            embeddings = generate_embedding(texts)
            dim = len(embeddings[0]) if embeddings and embeddings[0] is not None else embedding_dim()
            docs = _assemble_docs(
                pool_lookup, embeddings, dim, doc_id=app_id, kb_id=index_name,
                extra={"success_label": success_label}, id_prefix=f"{app_id}__",
            )
            errs = es.insert(docs, index_name)
            if errs:
                logger.warning("[corpus] %s: %d insert errors (e.g. %s)", app_id, len(errs), errs[0])
            n_docs += 1
            logger.info("[corpus] indexed %s (%d chunks)", app_id, len(docs))
        except Exception as exc:  # noqa: BLE001
            logger.error("[corpus] failed on %s: %s", app_id, exc)
    return n_docs


def build_corpus_es(successful_dir: str, unsuccessful_dir: str,
                    index_name: str = CORPUS_INDEX, recreate: bool = False) -> dict:
    """One-time build of the persistent ES corpus from labelled applications.

    Each chunk carries ``doc_id`` (application id) and ``success_label``
    (``successful``/``unsuccessful``) so few-shot retrieval can exclude the
    current application and prefer successful exemplars. Requires a running ES.
    """
    from deepdoc_engine.rag.utils.es_conn import ESConnection

    es = ESConnection()
    client = es.es
    if recreate and client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
        logger.info("[corpus] dropped existing index %s", index_name)
    if not client.indices.exists(index=index_name):
        client.indices.create(
            index=index_name,
            settings=es.mapping["settings"],
            mappings=es.mapping["mappings"],
        )
        logger.info("[corpus] created index %s", index_name)

    n_ok = _index_labeled_pdfs(es, index_name, successful_dir, "successful")
    n_un = _index_labeled_pdfs(es, index_name, unsuccessful_dir, "unsuccessful")
    client.indices.refresh(index=index_name)
    summary = {"index": index_name, "successful_docs": n_ok, "unsuccessful_docs": n_un}
    logger.info("[corpus] done: %s", summary)
    return summary


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Build the persistent ES corpus.")
    ap.add_argument("--successful", default="data/successful")
    ap.add_argument("--unsuccessful", default="data/unsuccessful")
    ap.add_argument("--index", default=CORPUS_INDEX)
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()
    build_corpus_es(args.successful, args.unsuccessful, args.index, args.recreate)
