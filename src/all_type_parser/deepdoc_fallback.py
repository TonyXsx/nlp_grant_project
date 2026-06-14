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


def _dummy_callback(prog=None, msg: str = "") -> None:
    """No-op progress callback expected by the DeepDOC parsers."""
    return None


def _as_application_details(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    return {"APPLICATION DETAILS": {"Raw Content": text}}


def parse_pdf(pdf_path: str) -> dict:
    """Run the full DeepDOC PDF pipeline and return the unified JSON dict."""
    try:
        from deepdoc_engine.rag.app.naive import chunk

        # section_only=True returns the merged section strings (post layout +
        # table + block-concatenation) without the embedding/tokenize step.
        sections = chunk(
            pdf_path,
            from_page=0,
            to_page=100000,
            lang="English",
            callback=_dummy_callback,
            section_only=True,
        )
        text = "\n\n".join(s for s in (sections or []) if isinstance(s, str) and s.strip())
        return _as_application_details(text)
    except Exception as e:  # noqa: BLE001 - fallback must never raise
        logger.warning("[deepdoc_fallback] PDF parse failed: %s", e)
        return {}


def parse_docx(docx_path: str) -> dict:
    """Run the DeepDOC DOCX parser and return the unified JSON dict."""
    try:
        from deepdoc_engine.rag.app.naive import chunk

        sections = chunk(
            docx_path,
            lang="English",
            callback=_dummy_callback,
            section_only=True,
        )
        text = "\n\n".join(s for s in (sections or []) if isinstance(s, str) and s.strip())
        return _as_application_details(text)
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
