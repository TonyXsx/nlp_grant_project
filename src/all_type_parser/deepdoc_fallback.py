"""
DeepDOC parsing fallback.

Thin adapter around the vendored DeepDOC engine (``src/deepdoc_engine/``, copied
from the swxy project) that produces the grant project's unified JSON contract.

It is used by ``all_type_parser`` only when the rule-based, format-specific
parsers fail to recognise a document:

  * PDF  — full DeepDOC pipeline (CV layout detection + OCR + table-structure
           recognition + XGBoost block-concatenation), mirroring the upstream
           ``naive.chunk`` architecture.
  * DOCX — DeepDOC ``RAGFlowDocxParser`` (paragraphs + composed table content).
  * PPTX — DeepDOC ``RAGFlowPptParser`` (per-slide text + tables).

Because DeepDOC produces layout blocks with no knowledge of the NIHR section
names, the extracted text is returned flat under
``{"APPLICATION DETAILS": {"Raw Content": ...}}`` — the same graceful-degradation
contract the router already uses for python-docx output. ``build_pool`` re-chunks
this content downstream.

Every entry point returns ``{}`` on failure or empty extraction so the router
can fall through to the next stage (the LLM fallback).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the vendored engine package (src/deepdoc_engine) is importable
# regardless of the caller's working directory.
_SRC = Path(__file__).resolve().parent.parent  # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

logger = logging.getLogger(__name__)


# Token budget / delimiter used only to assemble the parser's section pieces into
# readable Raw Content; build_pool re-chunks this downstream, so exact values are
# not critical. Kept consistent with build_pool's English settings.
_CHUNK_TOKEN_NUM = 256
_DELIMITER = "\n.!?;:"


def _dummy_callback(prog=None, msg: str = "") -> None:
    """No-op progress callback expected by the DeepDOC parsers."""
    return None


def _table_htmls(tables) -> list:
    """Pull restored <table> HTML strings out of a DeepDOC parser's tables list.

    Each table item is ``((img, content), positions)``; for tables ``content``
    is an HTML string, for figures it is a list of caption texts — we keep only
    the HTML tables (e.g. budget / methods tables)."""
    out: list = []
    for item in (tables or []):
        content = None
        try:
            (_img, content), _poss = item
        except Exception:
            try:
                _img, content = item
            except Exception:
                continue
        if isinstance(content, str) and "<table" in content.lower():
            out.append(content)
    return out


def _as_application_details(text: str, table_htmls: list | None = None) -> dict:
    text = (text or "").strip()
    details: dict = {}
    if text:
        details["Raw Content"] = text
    if table_htmls:
        # Each table becomes its own leaf → build_pool turns it into a table chunk.
        details["Document Tables"] = {
            f"Table {i + 1}": html for i, html in enumerate(table_htmls)
        }
    if not details:
        return {}
    return {"APPLICATION DETAILS": details}


def parse_pdf(pdf_path: str) -> dict:
    """Run the full DeepDOC PDF pipeline and return the unified JSON dict.

    Uses the ``Pdf`` wrapper directly (one parse) so we get both the text
    sections and the restored HTML tables; tables are emitted under
    ``APPLICATION DETAILS > Document Tables`` for build_pool to chunk separately.
    """
    try:
        from deepdoc_engine.rag.app.naive import Pdf
        from deepdoc_engine.rag.nlp import naive_merge

        sections, tables = Pdf()(
            pdf_path, from_page=0, to_page=100000, callback=_dummy_callback
        )
        chunks = naive_merge(sections, _CHUNK_TOKEN_NUM, _DELIMITER)
        text = "\n\n".join(c for c in (chunks or []) if isinstance(c, str) and c.strip())
        return _as_application_details(text, _table_htmls(tables))
    except Exception as e:  # noqa: BLE001 - fallback must never raise
        logger.warning("[deepdoc_fallback] PDF parse failed: %s", e)
        return {}


def parse_docx(docx_path: str) -> dict:
    """Run the DeepDOC DOCX parser (paragraphs + composed table HTML)."""
    try:
        from deepdoc_engine.rag.app.naive import Docx
        from deepdoc_engine.rag.nlp import naive_merge_docx

        sections, tables = Docx()(docx_path, None)
        chunks, _images = naive_merge_docx(sections, _CHUNK_TOKEN_NUM, _DELIMITER)
        text = "\n\n".join(c for c in (chunks or []) if isinstance(c, str) and c.strip())
        return _as_application_details(text, _table_htmls(tables))
    except Exception as e:  # noqa: BLE001
        logger.warning("[deepdoc_fallback] DOCX parse failed: %s", e)
        return {}


def parse_pptx(pptx_path: str) -> dict:
    """Run the DeepDOC PPT parser and return the unified JSON dict.

    ``naive.chunk`` has no pptx branch upstream, so the slide parser is called
    directly (mirroring how RAGFlow's presentation app uses it).
    """
    try:
        from deepdoc_engine.deepdoc.parser import PptParser

        slides = PptParser()(pptx_path, 0, 100000, _dummy_callback)
        text = "\n\n".join(s for s in (slides or []) if isinstance(s, str) and s.strip())
        return _as_application_details(text)
    except Exception as e:  # noqa: BLE001
        logger.warning("[deepdoc_fallback] PPTX parse failed: %s", e)
        return {}
