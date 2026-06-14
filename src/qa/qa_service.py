"""Grounded QA over a scored grant application (reuses the swxy Dealer + the
existing stores + the active LLM backend).

Public entry point: ``answer_question(result, question, history, role)``.

Pipeline:
  1. route()  — ONE LLM call classifies the question into {single_doc, scoring,
                corpus} and rewrites it into a standalone retrieval query
                (built-in query rewrite — resolves follow-ups like "why is it
                low?" against the conversation).
  2. answer   — per mode:
       single_doc : Dealer hybrid-retrieve the current PDF's chunks → cite text+page
       scoring    : feed the scored features (scores / pros / drawbacks / evidence) → explain
       corpus     : Dealer over the ES corpus, EXCLUDING the current application
                    (committee/admin only) → compare
  All answers are grounded ("answer only from the provided material; cite [n] or
  say not found"). Returns {answer, mode, citations:[{n,text,section,pages}]}.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

_SRC = Path(__file__).resolve().parent.parent  # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Load project .env (LLM backend / keys) if available.
try:
    from dotenv import load_dotenv
    load_dotenv(_SRC.parent / ".env")
except Exception:
    pass

from deepdoc_engine.rag.nlp.search_v2 import Dealer  # noqa: E402

# Cache of per-application in-memory stores so we don't re-embed on every
# question. Keyed by job/doc id → (conn, index_name).
_STORE_CACHE: dict[str, tuple] = {}

CORPUS_ROLES = {"committee", "admin"}  # who may ask cross-application questions


# ── LLM backend (mirrors qwen3_ollama: SCORER_BACKEND = ollama | deepseek) ────
def _chat(messages: list[dict], *, json_mode: bool = False, max_tokens: int = 1500) -> str:
    backend = os.environ.get("SCORER_BACKEND", "ollama").lower()
    if backend in ("deepseek", "api", "openai"):
        base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        payload: dict[str, Any] = {
            "model": model, "messages": messages,
            "temperature": 0.2, "max_tokens": max_tokens, "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = requests.post(f"{base}/chat/completions",
                          headers={"Authorization": f"Bearer {key}",
                                   "Content-Type": "application/json"},
                          json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    # Ollama
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "qwen3.5:27b")
    payload = {"model": model, "messages": messages, "stream": False,
               "options": {"temperature": 0.2, "num_predict": max_tokens}}
    if json_mode:
        payload["format"] = "json"
    r = requests.post(f"{host}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "")


def _extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b != -1 and b > a:
        text = text[a:b + 1]
    try:
        return json.loads(text)
    except Exception:
        return {}


# ── 1. intent router (+ query rewrite) ────────────────────────────────────────
def route(question: str, history: list[dict] | None = None, allow_corpus: bool = True) -> dict:
    """Classify the question and rewrite it into a standalone retrieval query.

    Returns {"mode": single_doc|scoring|corpus, "search_query": str}.
    """
    convo = ""
    for turn in (history or [])[-4:]:
        role = turn.get("role", "user")
        convo += f"{role}: {turn.get('content', '')}\n"
    modes = "single_doc, scoring" + (", corpus" if allow_corpus else "")
    system = (
        "You route questions about a scored grant application. Return JSON only.\n"
        "Pick one `mode`:\n"
        "  - single_doc: about the application's CONTENT (what it says/proposes/budgets).\n"
        "  - scoring: about the AI ASSESSMENT (why a score, strengths/weaknesses, evidence behind a score).\n"
        + ("  - corpus: COMPARISON across other applications / what strong applications do.\n" if allow_corpus else "")
        + f"Allowed modes: {modes}.\n"
        "Also return `search_query`: a standalone retrieval query for the question, "
        "resolving any pronouns/follow-ups using the conversation. Keep it concise.\n"
        'Output: {"mode": "...", "search_query": "..."}'
    )
    user = (f"Conversation so far:\n{convo}\n" if convo else "") + f"Question: {question}"
    try:
        raw = _chat([{"role": "system", "content": system},
                     {"role": "user", "content": user}], json_mode=True, max_tokens=300)
        parsed = _extract_json(raw)
    except Exception:
        parsed = {}
    mode = parsed.get("mode", "single_doc")
    if mode not in ("single_doc", "scoring", "corpus"):
        mode = "single_doc"
    if mode == "corpus" and not allow_corpus:
        mode = "single_doc"
    query = (parsed.get("search_query") or question).strip() or question
    return {"mode": mode, "search_query": query}


# ── retrieval helpers ─────────────────────────────────────────────────────────
def _current_store(job_id: str, pool_lookup: dict):
    """Build (once, cached) the in-memory store for the current application."""
    if job_id in _STORE_CACHE:
        return _STORE_CACHE[job_id]
    from retrieval.indexer import build_index_from_pool
    conn, index_name, _dim = build_index_from_pool(pool_lookup, doc_id=job_id)
    _STORE_CACHE[job_id] = (conn, index_name)
    return conn, index_name


def _retrieve(conn, index_name: str, query: str, *, top_k: int, doc_exclude: str | None = None,
              success_label: str | None = None) -> list[dict]:
    dealer = Dealer(dataStore=conn)
    ranks = dealer.retrieval(
        question=query, embd_mdl=None, tenant_ids=index_name, kb_ids=None,
        page=1, page_size=top_k + (4 if (doc_exclude or success_label) else 0),
        similarity_threshold=0.0, vector_similarity_weight=0.7, rank_feature=None,
    )
    chunks = ranks.get("chunks", [])
    out = []
    for c in chunks:
        if doc_exclude and c.get("doc_id") == doc_exclude:
            continue
        if success_label and c.get("success_label") not in (None, "", success_label):
            continue
        out.append(c)
        if len(out) >= top_k:
            break
    return out


def _pages_of(chunk: dict, pool_lookup: dict | None = None) -> list[int]:
    pos = chunk.get("positions") or chunk.get("position_int") or []
    if not pos and pool_lookup:
        meta = pool_lookup.get(chunk.get("chunk_id") or chunk.get("id"))
        pos = (meta or {}).get("position") or []
    return sorted({int(p[0]) for p in pos if p}) if pos else []


def _references_block(chunks: list[dict], pool_lookup: dict | None = None) -> tuple[str, list[dict]]:
    """Numbered reference text for the prompt + citation list for the UI."""
    lines, cites = [], []
    for i, c in enumerate(chunks, 1):
        text = (c.get("content_with_weight") or c.get("text") or "").strip()
        section = c.get("docnm_kwd") or c.get("section") or ""
        pages = _pages_of(c, pool_lookup)
        meta = f"Section: {section}" + (f", p.{', '.join(map(str, pages))}" if pages else "")
        lines.append(f"[{i}] ({meta})\n{text}")
        cites.append({"n": i, "text": text[:400], "section": section, "pages": pages})
    return "\n\n".join(lines), cites


_GROUNDING = (
    "Answer ONLY from the material above. Cite the sources you use inline as [n]. "
    "If the answer is not in the material, say you could not find it in the application. "
    "Be concise and specific."
)


# ── 2. answer modes ───────────────────────────────────────────────────────────
def _answer_single_doc(question, search_query, job_id, result, role):
    pool_lookup = result.get("pool_lookup", {})
    conn, index_name = _current_store(job_id, pool_lookup)
    chunks = _retrieve(conn, index_name, search_query, top_k=6)
    refs, cites = _references_block(chunks, pool_lookup)
    system = (
        f"You are a grant-application assistant answering for a {role}. "
        "Use the retrieved excerpts from THIS application to answer.\n" + _GROUNDING
    )
    user = f"Application excerpts:\n{refs}\n\nQuestion: {question}"
    answer = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return {"answer": answer, "mode": "single_doc", "citations": cites}


def _format_assessment(result: dict, max_chars: int = 9000) -> tuple[str, list[dict]]:
    """Compact view of the scored result for scoring-QA, with evidence citations."""
    feats = result.get("features", {}) or {}
    overall = result.get("overall", {}) or {}
    lines = [f"OVERALL score: {overall.get('final_score_0to100')}/100"]
    cites, n = [], 0
    for skey, sec in feats.items():
        if skey == "orcid":
            continue
        lines.append(f"\n## {skey} — section score {sec.get('score_10')}/10")
        for sub in sec.get("sub_criteria", []) or []:
            lines.append(f"- {sub.get('sub_id')} {sub.get('name')}: {sub.get('score_10')}/10")
            if sub.get("pros"):
                lines.append(f"    strengths: {sub['pros']}")
            if sub.get("drawbacks"):
                lines.append(f"    weaknesses: {sub['drawbacks']}")
            for ev in (sub.get("evidence") or [])[:2]:
                n += 1
                pages = ev.get("pages") or []
                meta = (ev.get("section") or "") + (f", p.{', '.join(map(str, pages))}" if pages else "")
                lines.append(f"    [{n}] evidence ({meta}): {(ev.get('text') or '')[:200]}")
                cites.append({"n": n, "text": (ev.get("text") or "")[:400],
                              "section": ev.get("section") or "", "pages": pages})
    text = "\n".join(lines)
    return text[:max_chars], cites


def _answer_scoring(question, result, role):
    assessment, cites = _format_assessment(result)
    system = (
        f"You explain an AI grant assessment to a {role}. Use the assessment below "
        "(section/sub-criterion scores, strengths, weaknesses, and cited evidence) "
        "to answer why something scored as it did and what the evidence is.\n" + _GROUNDING
    )
    user = f"AI assessment:\n{assessment}\n\nQuestion: {question}"
    answer = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return {"answer": answer, "mode": "scoring", "citations": cites}


def _answer_corpus(question, search_query, result, role):
    corpus_index = os.environ.get("GRANT_CORPUS_INDEX")
    if not corpus_index:
        return {"answer": "Cross-application comparison is unavailable (no corpus index configured).",
                "mode": "corpus", "citations": []}
    try:
        from deepdoc_engine.rag.utils.es_conn import ESConnection
        es = ESConnection()
        if not es.es.ping():
            raise RuntimeError("ES not reachable")
    except Exception as exc:
        return {"answer": f"Cross-application comparison is unavailable ({exc}).",
                "mode": "corpus", "citations": []}
    current_id = result.get("doc_id")
    chunks = _retrieve(es, corpus_index, search_query, top_k=6,
                       doc_exclude=current_id, success_label="successful")
    refs, cites = _references_block(chunks)
    # add which application each came from
    for c, cite in zip(chunks, cites):
        cite["application"] = c.get("doc_id", "")
    system = (
        f"You are a grant portfolio assistant for a {role}. Compare using excerpts "
        "from OTHER (successful) applications in the corpus.\n" + _GROUNDING +
        " Make clear these excerpts are from other applications, not the current one."
    )
    user = f"Excerpts from successful applications:\n{refs}\n\nQuestion: {question}"
    answer = _chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return {"answer": answer, "mode": "corpus", "citations": cites}


# ── public entry point ────────────────────────────────────────────────────────
def answer_question(result: dict, question: str, *, job_id: str,
                    history: list[dict] | None = None, role: str = "reviewer") -> dict:
    """Route + answer a question about a scored application.

    `result` is the scored JSON (must include pool_lookup + features).
    Returns {answer, mode, citations}.
    """
    allow_corpus = role.lower() in CORPUS_ROLES
    routed = route(question, history, allow_corpus=allow_corpus)
    mode = routed["mode"]
    sq = routed["search_query"]
    if mode == "scoring":
        return _answer_scoring(question, result, role)
    if mode == "corpus":
        return _answer_corpus(question, sq, result, role)
    return _answer_single_doc(question, sq, job_id, result, role)
