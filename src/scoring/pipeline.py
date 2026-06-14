from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from src.pool.build_pool import (
    APPLICATION_CONTEXT_SECTION,
    APPLICATION_FORM_ANALYSIS_SECTION,
    MAX_CHARS,
    PLAIN_ENGLISH_ANALYSIS_SECTION,
    build_chunk_pool,
    write_pool_artifacts,
)

SECTION_KEY_MAP = {
    "General": "general",
    "Proposed research": "proposed_research",
    "Training and development": "training_development",
    "Sites and support": "sites_support",
    "Working with people and communities": "wpcc",
    "Application Form": "application_form",
}
SECTION_ID_PREFIX = {
    "general": "g",
    "proposed_research": "pr",
    "training_development": "td",
    "sites_support": "ss",
    "wpcc": "wp",
    "application_form": "af",
}
STAGE1_EXCLUDED_SECTION_KEYS = {"application_form"}
SCORER_ALLOWED_SCORES = (0, 1, 2, 3, 4, 5)

# ── Doc-type exclusions ────────────────────────────────────────────────────────
# RfPB (Research for Patient Benefit) is a project grant, not a fellowship.
# It has no training/career-development section, so `training_development` is
# meaningless and should not drag down the overall score.
# g.1 ("Common Characteristics of Good Applications") and g.2 ("Tell us why
# you need this award") are fellowship-oriented sub-criteria that ask about
# personal career trajectory and need for the award — neither applies when
# scoring a project grant where the research merit stands on its own.
OVERALL_EXCLUDED_SECTIONS_BY_DOC_TYPE: dict[str, set[str]] = {
    "rfpb": {"training_development"},
}
SECTION_EXCLUDED_SUB_IDS_BY_DOC_TYPE: dict[str, set[str]] = {
    "rfpb": {"g.1", "g.2"},
}
# Maximum score (0–10) for specific sub-criteria by doc_type.
# llm_fallback PDFs lose structural formatting on extraction, so the
# "Use of Formatting, Headings, and Subheadings" criterion (af.1) cannot
# reliably score above 6/10 — cap it to avoid rewarding format we cannot see.
SUB_ID_SCORE_CAPS_BY_DOC_TYPE: dict[str, dict[str, float]] = {
    "llm_fallback": {"af.1": 6.0},
}
# ──────────────────────────────────────────────────────────────────────────────
SCORER_MAX_SCORE = 5
USED_CHUNK_MAX = 5
SECTION_EVIDENCE_MAX = 3
STAGE1_MAX_TOKENS = int(os.environ.get("STAGE1_MAX_TOKENS", "8192"))
STAGE2_MAX_TOKENS = int(os.environ.get("STAGE2_MAX_TOKENS", "80000"))
STAGE1_MAX_SIGNALS_PER_FINDING = 3
STAGE1_IMPLICATION_MAX_CHARS = 220
STAGE1_HISTORY_IMPLICATIONS_PER_SIGNAL = 2
JSON_PARSE_MAX_RETRIES = 1

SECTION_TO_PARSER_SECTIONS: dict[str, list[str] | None] = {
    "general": None,
    "proposed_research": [
        APPLICATION_CONTEXT_SECTION,
        PLAIN_ENGLISH_ANALYSIS_SECTION,
        "Plain English Summary of Research",
        "Plain English Summary",
        "Scientific Abstract",
        "Detailed Research Plan",
        "Changes from Previous Stage",
        "Patient & Public Involvement",
        "Working with People and Communities Summary",
        "SUMMARY BUDGET",
    ],
    "training_development": [
        APPLICATION_CONTEXT_SECTION,
        "Training & Development and Research Support",
        "SUPPORT AND MENTORSHIP",
    ],
    "sites_support": [
        APPLICATION_CONTEXT_SECTION,
        "Training & Development and Research Support",
        "SUPPORT AND MENTORSHIP",
    ],
    "wpcc": [
        APPLICATION_CONTEXT_SECTION,
        "Patient & Public Involvement",
        "Working with People and Communities Summary",
        "Detailed Research Plan",
        "SUMMARY BUDGET",
    ],
    "application_form": [
        APPLICATION_FORM_ANALYSIS_SECTION,
    ],
}
CONFIDENCE_TO_SCORE = {
    "low_confidence": 0,
    "medium_confidence": 1,
    "high_confidence": 2,
}


def load_rubric(criteria_path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(criteria_path).read_text(encoding="utf-8"))
    sections: list[dict[str, Any]] = []

    def build_signal_objects(prefix: str, signal_texts: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "sid": f"{prefix}.{chr(97 + idx)}",
                "text": signal_text,
                "weight": 1.0,
            }
            for idx, signal_text in enumerate(signal_texts)
        ]

    def build_sub(
        *,
        sub_id: str,
        name: str,
        definition: str,
        signal_texts: list[str],
        group_name: str | None = None,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        sub = {
            "sub_id": sub_id,
            "name": name,
            "definition": definition,
            "weight": float(weight),
            "signals": build_signal_objects(sub_id, signal_texts),
        }
        if group_name:
            sub["group_name"] = group_name
        return sub

    for human_name, payload in raw.items():
        if human_name == "meta":
            continue
        section_key = SECTION_KEY_MAP.get(human_name)
        if not section_key:
            continue
        sub_criteria = []
        if isinstance(payload, dict) and isinstance(payload.get("sub_criteria"), list):
            for sub in payload.get("sub_criteria", []):
                signals = []
                for signal in sub.get("signals", []):
                    signals.append({
                        "sid": signal["sid"],
                        "text": signal["text"],
                        "weight": float(signal.get("weight", 1.0)),
                    })
                item = {
                    "sub_id": sub["sub_id"],
                    "name": sub["name"],
                    "definition": sub.get("definition", ""),
                    "weight": float(sub.get("weight", 1.0)),
                    "signals": signals,
                }
                if sub.get("group_name"):
                    item["group_name"] = sub["group_name"]
                sub_criteria.append(item)
        elif human_name == "General" and isinstance(payload, dict):
            idx = 1
            for group_key in ("common_characteristics_of_good_applications", "tell_us_why_you_need_this_award"):
                group_val = payload.get(group_key)
                if isinstance(group_val, dict):
                    sub_criteria.append(build_sub(
                        sub_id=f"g.{idx}",
                        name=group_val["name"],
                        definition=group_val.get("definition", ""),
                        signal_texts=list(group_val.get("signals", [])),
                        group_name=group_val["name"],
                    ))
                    idx += 1
            for applicant_item in payload.get("Applicant", []):
                sub_criteria.append(build_sub(
                    sub_id=f"g.{idx}",
                    name=applicant_item["name"],
                    definition=applicant_item.get("definition", ""),
                    signal_texts=list(applicant_item.get("signals", [])),
                    group_name="Applicant",
                ))
                idx += 1
        elif isinstance(payload, list):
            prefix = SECTION_ID_PREFIX[section_key]
            for idx, sub in enumerate(payload, start=1):
                sub_criteria.append(build_sub(
                    sub_id=f"{prefix}.{idx}",
                    name=sub["name"],
                    definition=sub.get("definition", ""),
                    signal_texts=list(sub.get("signals", [])),
                ))
        else:
            continue
        sections.append({
            "human_name": human_name,
            "section_key": section_key,
            "weight": float(payload.get("weight", 1.0)) if isinstance(payload, dict) else 1.0,
            "sub_criteria": sub_criteria,
        })
    return sections


def load_raw_criteria(criteria_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(criteria_path).read_text(encoding="utf-8"))


def _safe_json_loads(text: str) -> dict[str, Any]:
    clean = (text or "").strip()
    if clean.startswith("```"):
        clean = clean[3:]
        if clean.startswith("json"):
            clean = clean[4:]
        if "\n" in clean:
            clean = clean.split("\n", 1)[1]
        if clean.endswith("```"):
            clean = clean[:-3]
    clean = clean.strip()
    if clean:
        first_brace = clean.find("{")
        last_brace = clean.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            clean = clean[first_brace:last_brace + 1]
    return json.loads(clean)


class JsonRetryError(ValueError):
    def __init__(self, *, raw_response: str, attempts: int, original: Exception):
        super().__init__(str(original))
        self.raw_response = raw_response
        self.attempts = attempts
        self.original = original


def _generate_json_with_parse_retry(
    scorer_client: Any,
    messages: list[dict[str, str]],
    *,
    schema: dict[str, Any],
    max_tokens: int,
    max_retries: int = JSON_PARSE_MAX_RETRIES,
) -> tuple[str, dict[str, Any], int]:
    raw_response = ""
    last_exc: Exception | None = None
    for attempt_idx in range(max_retries + 1):
        raw_response = scorer_client.generate_json(messages, schema=schema, max_tokens=max_tokens)
        try:
            return raw_response, _safe_json_loads(raw_response), attempt_idx
        except Exception as exc:  # Retry only malformed model JSON, not transport/model errors.
            last_exc = exc

    raise JsonRetryError(
        raw_response=raw_response,
        attempts=max_retries + 1,
        original=last_exc or ValueError("unknown JSON parse error"),
    )


def _response_preview(text: str, limit: int = 500) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _chunk_order_map(pool_lookup: dict[str, dict[str, str]]) -> dict[str, int]:
    return {chunk_id: idx for idx, chunk_id in enumerate(pool_lookup)}


def _sort_chunk_ids(chunk_ids: list[str], chunk_order: dict[str, int]) -> list[str]:
    return sorted(chunk_ids, key=lambda chunk_id: chunk_order.get(chunk_id, 10**9))


def _confidence_label(avg_gap: float) -> str:
    if avg_gap < 0.5:
        return "high_confidence"
    if avg_gap < 1.5:
        return "medium_confidence"
    return "low_confidence"


def _compat_plausibility(avg_confidence_0to2: float) -> float:
    return round((avg_confidence_0to2 / 2) * 5, 2)


def _compat_evidence_score(avg_confidence_0to2: float) -> float:
    return round((avg_confidence_0to2 / 2) * 100, 2)


def _is_if_applicable_subcriterion(sub: dict[str, Any]) -> bool:
    text = f"{sub.get('name', '')} {sub.get('definition', '')}".lower()
    return "if applicable" in text


def rule_based_retrieval(
    rubric_sections: list[dict[str, Any]],
    section_chunk_ids: dict[str, list[str]],
    pool_lookup: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    all_chunk_ids = list(pool_lookup)
    parser_section_to_chunks: dict[str, list[str]] = {}
    for chunk_id, meta in pool_lookup.items():
        ps = meta["parser_section"]
        parser_section_to_chunks.setdefault(ps, []).append(chunk_id)

    result: dict[str, list[str]] = {}
    for section in rubric_sections:
        mapping = SECTION_TO_PARSER_SECTIONS.get(section["section_key"])
        if mapping is None:
            result[section["section_key"]] = list(all_chunk_ids)
        else:
            chunks: list[str] = []
            for ps_name in mapping:
                chunks.extend(parser_section_to_chunks.get(ps_name, []))
            result[section["section_key"]] = _dedupe_preserve_order(chunks)
    return result


def build_evidence_text(
    chunk_ids: list[str],
    pool_lookup: dict[str, dict[str, str]],
    chunk_order: dict[str, int],
) -> str:
    if not chunk_ids:
        return "(no retrieved chunks)"

    ordered_ids = _sort_chunk_ids(_dedupe_preserve_order(chunk_ids), chunk_order)
    lines: list[str] = []
    last_position: int | None = None
    last_section: str | None = None

    for chunk_id in ordered_ids:
        position = chunk_order.get(chunk_id, 10**9)
        meta = pool_lookup[chunk_id]
        current_section = meta["parser_section"]

        if last_section != current_section:
            if lines:
                lines.append("")
            lines.append(f"===== {current_section} =====")
            lines.append("")

        if last_position is not None and position - last_position > 1:
            lines.extend(["", "...", ""])
        lines.extend([
            f"<{chunk_id}>",
            meta["text"],
            f"<{chunk_id}>",
            "",
        ])
        last_position = position
        last_section = current_section

    return "\n".join(lines).strip()


def _build_full_application_text(
    pool_lookup: dict[str, dict[str, str]],
    chunk_order: dict[str, int],
) -> str:
    return build_evidence_text(list(pool_lookup), pool_lookup, chunk_order)


def _parser_sections_from_belief(
    rubric_section: dict[str, Any],
    belief_state: dict[str, Any],
    pool_lookup: dict[str, dict[str, str]],
) -> set[str]:
    sub_ids = {sub["sub_id"] for sub in rubric_section["sub_criteria"]}
    subcriteria_beliefs = belief_state.get("subcriteria_beliefs", {})
    if not isinstance(subcriteria_beliefs, dict):
        return set()
    parser_sections: set[str] = set()
    for sub_id, sub_entry in subcriteria_beliefs.items():
        if sub_id not in sub_ids or not isinstance(sub_entry, dict):
            continue
        for chunk_id in (
            *sub_entry.get("good_evidence_ids", []),
            *sub_entry.get("bad_evidence_ids", []),
        ):
            meta = pool_lookup.get(chunk_id)
            if meta:
                parser_sections.add(meta["parser_section"])
    return parser_sections


def _build_scoped_application_text(
    rubric_section: dict[str, Any],
    pool_lookup: dict[str, dict[str, str]],
    chunk_order: dict[str, int],
    belief_state: dict[str, Any],
) -> tuple[str, list[str]]:
    predefined = SECTION_TO_PARSER_SECTIONS.get(rubric_section["section_key"])
    if predefined is None:
        return _build_full_application_text(pool_lookup, chunk_order), []

    belief_parser_sections = _parser_sections_from_belief(
        rubric_section, belief_state, pool_lookup
    )
    relevant = set(predefined) | belief_parser_sections
    scoped_chunk_ids: list[str] = []
    first_order_by_section: dict[str, int] = {}
    for chunk_id, meta in pool_lookup.items():
        parser_section = meta["parser_section"]
        if parser_section not in relevant:
            continue
        scoped_chunk_ids.append(chunk_id)
        if parser_section not in first_order_by_section:
            first_order_by_section[parser_section] = chunk_order[chunk_id]
    if not scoped_chunk_ids:
        return _build_full_application_text(pool_lookup, chunk_order), []
    ordered = sorted(first_order_by_section, key=first_order_by_section.get)
    return build_evidence_text(scoped_chunk_ids, pool_lookup, chunk_order), ordered


def _build_current_retriever(pool_lookup, doc_id):
    """Embed the current application's chunks into an ephemeral in-memory store
    for hybrid evidence retrieval. Returns (conn, index_name) or None on any
    failure (missing sentence-transformers, model download offline, etc.) so the
    caller falls back to full-text scoring."""
    try:
        from src.retrieval.indexer import build_index_from_pool
        conn, index_name, _dim = build_index_from_pool(pool_lookup, doc_id=doc_id)
        return conn, index_name
    except Exception as exc:  # noqa: BLE001
        logger.warning("retrieval index build failed (%s); using full text", exc)
        return None


def _build_corpus_retriever(corpus_index):
    """Connect to the persistent ES corpus for few-shot retrieval. Returns
    (es_conn, corpus_index) or None when no index is requested or ES is
    unreachable (few-shot is then silently skipped — scoring is unaffected)."""
    if not corpus_index:
        return None
    try:
        from deepdoc_engine.rag.utils.es_conn import ESConnection
        es = ESConnection()
        if not es.es.ping():
            logger.warning("ES not reachable; skipping few-shot examples")
            return None
        return es, corpus_index
    except Exception as exc:  # noqa: BLE001
        logger.warning("corpus retriever unavailable (%s); skipping few-shot", exc)
        return None


def _fewshot_examples(corpus_retriever, rubric_section, current_app_id, n):
    """Few exemplar texts from SUCCESSFUL corpus applications for this rubric
    section, excluding the current application. [] on any failure."""
    if not corpus_retriever or n <= 0:
        return []
    es_conn, corpus_index = corpus_retriever
    try:
        from src.retrieval.retriever import fewshot_for_section
        chunks = fewshot_for_section(
            es_conn, corpus_index, rubric_section,
            current_app_id=current_app_id, n=n, success_label="successful",
        )
        return [c["content_with_weight"] for c in chunks if c.get("content_with_weight")]
    except Exception as exc:  # noqa: BLE001
        logger.warning("few-shot retrieval failed (%s); skipping", exc)
        return []


def _retrieval_scope(retriever, rubric_section, pool_lookup, chunk_order, top_k):
    """(scoped_text, ordered_parser_sections) from hybrid retrieval over the
    current-PDF store, or (None, None) to signal a full-text fallback."""
    if not retriever:
        return None, None
    conn, index_name = retriever
    try:
        from src.retrieval.retriever import evidence_for_section
        chunks = evidence_for_section(conn, index_name, rubric_section, top_k=top_k)
        ids = [c["chunk_id"] for c in chunks if c.get("chunk_id") in pool_lookup]
        if not ids:
            return None, None
        text = build_evidence_text(ids, pool_lookup, chunk_order)
        sections = _dedupe_preserve_order(
            [pool_lookup[i]["parser_section"] for i in ids]
        )
        return text, sections
    except Exception as exc:  # noqa: BLE001
        logger.warning("evidence retrieval failed (%s); using full text", exc)
        return None, None


def _strip_rubric_for_prompt(rubric_sections: list[dict[str, Any]]) -> dict[str, Any]:
    stripped: dict[str, Any] = {}
    for section in rubric_sections:
        stripped[section["human_name"]] = {
            "sub_criteria": [
                {
                    "sub_id": sub["sub_id"],
                    "name": sub["name"],
                    "definition": sub["definition"],
                    "group_name": sub.get("group_name"),
                    "signals": [
                        {
                            "sid": signal["sid"],
                            "text": signal["text"],
                        }
                        for signal in sub["signals"]
                    ],
                }
                for sub in section["sub_criteria"]
            ]
        }
    return stripped



def _stage1_rubric_sections(rubric_sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        section
        for section in rubric_sections
        if section["section_key"] not in STAGE1_EXCLUDED_SECTION_KEYS
    ]


def _initial_belief_state(rubric_sections: list[dict[str, Any]]) -> dict[str, Any]:
    stage1_sections = _stage1_rubric_sections(rubric_sections)
    all_sub_ids = [
        sub["sub_id"]
        for section in stage1_sections
        for sub in section["sub_criteria"]
    ]
    return {
        "processed_sections": [],
        "subcriteria_beliefs": {},
        "missing_signals": all_sub_ids,   # tracks sub IDs (not signal IDs)
    }


def _section_inputs(
    section_chunk_ids: dict[str, list[str]],
    pool_lookup: dict[str, dict[str, str]],
    chunk_order: dict[str, int],
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for section_name, chunk_ids in section_chunk_ids.items():
        ordered_ids = _sort_chunk_ids(chunk_ids, chunk_order)
        sections.append({
            "section_name": section_name,
            "section_content": {
                chunk_id: pool_lookup[chunk_id]["text"]
                for chunk_id in ordered_ids
            },
        })
    return sections


def _signal_sub_map(rubric_sections: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for section in rubric_sections:
        for sub in section["sub_criteria"]:
            for signal in sub["signals"]:
                mapping[signal["sid"]] = {
                    "sub_id": sub["sub_id"],
                    "sub_name": sub["name"],
                    "section_key": section["section_key"],
                    "section_name": section["human_name"],
                    "signal_text": signal["text"],
                }
    return mapping


def _build_stage1_criteria_view(rubric_sections: list[dict[str, Any]]) -> dict[str, Any]:
    stage1_sections = _stage1_rubric_sections(rubric_sections)
    return {
        "signals": [
            {
                "sid": signal["sid"],
                "sub_id": sub["sub_id"],
                "sub_name": sub["name"],
                "section_name": section["human_name"],
                "text": signal["text"],
            }
            for section in stage1_sections
            for sub in section["sub_criteria"]
            for signal in sub["signals"]
        ]
    }


def _build_stage1_belief_state_view(current_belief_state: dict[str, Any]) -> dict[str, Any]:
    processed_sections = list(current_belief_state.get("processed_sections", []))
    missing_signals = list(current_belief_state.get("missing_signals", []))
    resolved_sub_ids: list[str] = []
    sub_implications: list[dict[str, Any]] = []
    subcriteria_beliefs = current_belief_state.get("subcriteria_beliefs", {})
    if isinstance(subcriteria_beliefs, dict):
        for sub_id, sub_entry in subcriteria_beliefs.items():
            if not isinstance(sub_entry, dict):
                continue
            good_ids = sub_entry.get("good_evidence_ids") or []
            bad_ids = sub_entry.get("bad_evidence_ids") or []
            implications = sub_entry.get("implications") or []
            if good_ids or bad_ids:
                resolved_sub_ids.append(sub_id)
            if isinstance(implications, list):
                compact_implications = [
                    str(item).strip()[:STAGE1_IMPLICATION_MAX_CHARS]
                    for item in implications
                    if str(item).strip()
                ][:STAGE1_HISTORY_IMPLICATIONS_PER_SIGNAL]
                if compact_implications:
                    sub_implications.append({
                        "sub_id": sub_id,
                        "implications": compact_implications,
                    })
    return {
        "processed_sections": processed_sections[-10:],
        "processed_section_count": len(processed_sections),
        "missing_signals": missing_signals,
        "resolved_signals": _dedupe_preserve_order(resolved_sub_ids),
        "signal_implications": sub_implications,
    }


def _empty_section_update(section_name: str) -> dict[str, Any]:
    return {
        "section_name": section_name,
        "findings": [],
        "resolved_signals": [],
    }


def build_section_evidence_messages(
    *,
    application_section: dict[str, Any],
    stage1_criteria: dict[str, Any],
    current_belief_state: dict[str, Any],
) -> list[dict[str, str]]:
    section_name = application_section["section_name"]
    system = (
        "You review one parsed application section at a time.\n\n"
        "Return JSON only.\n"
        "Task: inspect the current section only, then return one finding per sub-criterion that is clearly supported or clearly challenged by the current section text.\n"
        "Do not summarize the full criteria. Do not explain why unrelated sub-criteria are irrelevant.\n"
        "Only use evidence IDs that appear in application_section.section_content.\n"
        "The current_belief_state may include prior implications for context, but it does not include prior evidence IDs.\n"
        "Never invent IDs.\n"
        "Identifier rules (STRICT):\n"
        "  - Each finding has exactly three fields: `sub_id`, `evidence`, `implication`.\n"
        "  - `sub_id` MUST be the parent subcriterion id (e.g. `g.4`, `pr.2`). NEVER a signal id.\n"
        "  - `evidence` is an object with `good_evidence_ids` and `bad_evidence_ids` (chunk ID lists).\n"
        "  - `implication` is a short string summarising what this section reveals about the sub-criterion.\n"
        f"Each evidence list holds at most {SECTION_EVIDENCE_MAX} IDs.\n"
        "Return one finding per relevant sub-criterion (no upper limit).\n"
        "A sub-criterion is relevant only if the section text gives concrete evidence for or against it.\n"
        "If the section only contains metadata or weak context, return no findings.\n"
        "When there is no relevant evidence, return the required empty object shape and stop.\n"
        "Never return a bare array like `[]`.\n"
        f"Keep implications short, specific to the current section, and under {STAGE1_IMPLICATION_MAX_CHARS} characters.\n"
        "Prefer the strongest, most decision-relevant evidence. Omit weaker or duplicative matches."
    )
    user_payload = {
        "application_section": application_section,
        "criteria": stage1_criteria,
        "current_belief_state": _build_stage1_belief_state_view(current_belief_state),
    }
    empty_output_example = {
        "section_name": section_name,
        "findings": [],
        "resolved_signals": [],
    }
    non_empty_output_example = {
        "section_name": section_name,
        "findings": [
            {
                "sub_id": "g.4",
                "evidence": {
                    "good_evidence_ids": ["secxx__001_a", "secxx__001_b"],
                    "bad_evidence_ids": [],
                },
                "implication": f"{section_name}: lists 7 publications and a PhD, directly supporting g.4.",
            },
            {
                "sub_id": "g.6",
                "evidence": {
                    "good_evidence_ids": ["secxx__002"],
                    "bad_evidence_ids": [],
                },
                "implication": f"{section_name}: applicant is a practitioner-researcher, supporting g.6.",
            },
        ],
        "resolved_signals": ["g.4", "g.6"],
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"Current section name: {section_name}\n"
                "Input JSON:\n"
                f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
                "Required behavior:\n"
                "1. Scan the current section text.\n"
                "2. Output only relevant subcriteria/signals.\n"
                "3. Output format example when there is relevant evidence:\n"
                f"{json.dumps(non_empty_output_example, ensure_ascii=False, indent=2)}\n\n"
                "4. If none are relevant, return exactly this object shape:\n"
                f"{json.dumps(empty_output_example, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


def _response_meta_summary(scorer_client: Any | None) -> str:
    body = getattr(scorer_client, "last_response_body", None)
    if not isinstance(body, dict):
        return ""
    done_reason = body.get("done_reason")
    message = body.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    thinking = message.get("thinking") if isinstance(message, dict) else None
    parts: list[str] = []
    if isinstance(done_reason, str) and done_reason:
        parts.append(f"done_reason={done_reason}")
    if isinstance(content, str):
        parts.append(f"content_len={len(content)}")
    if isinstance(thinking, str):
        parts.append(f"thinking_len={len(thinking)}")
    return ", ".join(parts)


def build_section_evidence_schema(
    rubric_sections: list[dict[str, Any]],
    section_chunk_ids: list[str],
) -> dict[str, Any]:
    stage1_sections = _stage1_rubric_sections(rubric_sections)
    sub_ids = [
        sub["sub_id"]
        for section in stage1_sections
        for sub in section["sub_criteria"]
    ]
    finding_schema = {
        "type": "object",
        "properties": {
            "sub_id": {"type": "string", "enum": sub_ids},
            "evidence": {
                "type": "object",
                "properties": {
                    "good_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string", "enum": section_chunk_ids},
                        "maxItems": SECTION_EVIDENCE_MAX,
                    },
                    "bad_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string", "enum": section_chunk_ids},
                        "maxItems": SECTION_EVIDENCE_MAX,
                    },
                },
                "required": ["good_evidence_ids", "bad_evidence_ids"],
                "additionalProperties": False,
            },
            "implication": {"type": "string"},
        },
        "required": ["sub_id", "evidence", "implication"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "section_name": {"type": "string"},
            "findings": {
                "type": "array",
                "items": finding_schema,
            },
            "resolved_signals": {
                "type": "array",
                "items": {"type": "string", "enum": sub_ids},
            },
        },
        "required": ["section_name", "findings", "resolved_signals"],
        "additionalProperties": False,
    }


def _ensure_implication_prefix(section_name: str, implication: str) -> str:
    clean = (implication or "").strip()
    prefix = f"{section_name}:"
    if not clean:
        return prefix
    return clean if clean.startswith(prefix) else f"{prefix} {clean}"


def _normalize_section_evidence_output(
    parsed: dict[str, Any],
    rubric_sections: list[dict[str, Any]],
    section_chunk_ids: list[str],
    section_name: str,
) -> dict[str, Any]:
    allowed_ids = set(section_chunk_ids)
    stage1_sections = _stage1_rubric_sections(rubric_sections)
    valid_sub_ids: set[str] = {
        sub["sub_id"]
        for section in stage1_sections
        for sub in section["sub_criteria"]
    }
    findings_by_sub: dict[str, dict[str, Any]] = {}
    resolved: list[str] = []
    raw_findings = parsed.get("findings", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_findings, list):
        raw_findings = []

    def _coerce_id_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            chunk_id for chunk_id in value
            if isinstance(chunk_id, str) and chunk_id in allowed_ids
        ]

    for raw_finding in raw_findings:
        if not isinstance(raw_finding, dict):
            continue
        sub_id = raw_finding.get("sub_id")
        if not isinstance(sub_id, str) or sub_id not in valid_sub_ids:
            continue

        raw_evidence = raw_finding.get("evidence", {})
        if not isinstance(raw_evidence, dict):
            raw_evidence = {}
        good_ids = _dedupe_preserve_order(
            _coerce_id_list(raw_evidence.get("good_evidence_ids"))
        )[:SECTION_EVIDENCE_MAX]
        bad_ids = _dedupe_preserve_order(
            _coerce_id_list(raw_evidence.get("bad_evidence_ids"))
        )[:SECTION_EVIDENCE_MAX]
        if not good_ids and not bad_ids:
            continue

        implication = raw_finding.get("implication", "")
        if not isinstance(implication, str):
            implication = ""

        if sub_id in findings_by_sub:
            existing = findings_by_sub[sub_id]
            existing["good_evidence_ids"] = _dedupe_preserve_order([
                *existing["good_evidence_ids"], *good_ids,
            ])[:SECTION_EVIDENCE_MAX]
            existing["bad_evidence_ids"] = _dedupe_preserve_order([
                *existing["bad_evidence_ids"], *bad_ids,
            ])[:SECTION_EVIDENCE_MAX]
            if implication.strip() and not existing["implication"].endswith(implication.strip()):
                existing["implication"] = _ensure_implication_prefix(
                    section_name, implication
                )[:STAGE1_IMPLICATION_MAX_CHARS]
        else:
            findings_by_sub[sub_id] = {
                "sub_id": sub_id,
                "good_evidence_ids": good_ids,
                "bad_evidence_ids": bad_ids,
                "implication": _ensure_implication_prefix(
                    section_name, implication
                )[:STAGE1_IMPLICATION_MAX_CHARS],
            }
            resolved.append(sub_id)

    return {
        "section_name": section_name,
        "findings": list(findings_by_sub.values()),
        "resolved_signals": _dedupe_preserve_order(resolved),
    }


def _merge_belief_state(
    current_belief_state: dict[str, Any],
    section_update: dict[str, Any],
) -> dict[str, Any]:
    next_state = json.loads(json.dumps(current_belief_state))
    processed_sections = list(next_state.get("processed_sections", []))
    section_name = section_update["section_name"]
    if section_name not in processed_sections:
        processed_sections.append(section_name)
    next_state["processed_sections"] = processed_sections

    subcriteria_beliefs = next_state.get("subcriteria_beliefs", {})
    if not isinstance(subcriteria_beliefs, dict):
        subcriteria_beliefs = {}

    for finding in section_update.get("findings", []):
        sub_id = finding["sub_id"]
        sub_entry = subcriteria_beliefs.setdefault(sub_id, {
            "good_evidence_ids": [],
            "bad_evidence_ids": [],
            "implications": [],
        })
        sub_entry["good_evidence_ids"] = _dedupe_preserve_order([
            *sub_entry.get("good_evidence_ids", []),
            *finding.get("good_evidence_ids", []),
        ])
        sub_entry["bad_evidence_ids"] = _dedupe_preserve_order([
            *sub_entry.get("bad_evidence_ids", []),
            *finding.get("bad_evidence_ids", []),
        ])
        implication = finding.get("implication", "")
        if isinstance(implication, str) and implication.strip():
            sub_entry["implications"] = _dedupe_preserve_order([
                *sub_entry.get("implications", []),
                implication.strip(),
            ])

    next_state["subcriteria_beliefs"] = subcriteria_beliefs
    missing_signals = list(next_state.get("missing_signals", []))
    resolved_signals = set(section_update.get("resolved_signals", []))
    next_state["missing_signals"] = [sid for sid in missing_signals if sid not in resolved_signals]
    return next_state


def _single_section_payload(section: dict[str, Any]) -> dict[str, Any]:
    return {
        "section_key": section["section_key"],
        "section_name": section["human_name"],
        "sub_criteria": [
            {
                "sub_id": sub["sub_id"],
                "name": sub["name"],
                "definition": sub["definition"],
                "group_name": sub.get("group_name"),
                "signals": [
                    {"sid": signal["sid"], "text": signal["text"]}
                    for signal in sub["signals"]
                ],
            }
            for sub in section["sub_criteria"]
        ],
    }


def build_final_scoring_messages(
    *,
    rubric_section: dict[str, Any],
    stripped_criteria: dict[str, Any],
    final_belief_state: dict[str, Any],
    scoped_application_text: str,
    scoped_parser_sections: list[str],
    examples: list[str] | None = None,
) -> list[dict[str, str]]:
    scope_note = (
        f"Application text is scoped to parser sections: {scoped_parser_sections}."
        if scoped_parser_sections
        else "Application text contains the full application (no scoping applied)."
    )
    target_sub_ids = [sub["sub_id"] for sub in rubric_section["sub_criteria"]]
    application_form_note = (
        "For Application Form scoring, the application text is a derived structural-analysis chunk. "
        "Use its metrics for formatting, duplication, and coherence signals; do not penalize missing "
        "bold/emphasis markers unless the analysis indicates they are absent rather than parser-lost.\n"
        if rubric_section["section_key"] == "application_form"
        else ""
    )
    proposed_research_note = (
        "For Proposed Research pr.1 Plain English Summary scoring, use any Plain English NLP Analysis metrics "
        "as supporting evidence, but still read the raw Plain English Summary and Detailed Research Plan in "
        "application_text. You must judge readability, jargon explanation, alignment, and sentence coherence "
        "yourself from the text; do not say these cannot be assessed merely because Stage 1 did not provide "
        "readability/coherence findings.\n"
        if rubric_section["section_key"] == "proposed_research"
        else ""
    )
    system = (
        "You are scoring one rubric section of a grant application.\n\n"
        "Return JSON only.\n"
        "Score only the target rubric section.\n"
        f"Target rubric section: section_key=`{rubric_section['section_key']}`, section_name=`{rubric_section['human_name']}`.\n"
        f"Allowed top-level keys (parent sub ids): {target_sub_ids}.\n"
        f"{scope_note}\n"
        "Use the final belief state as the primary evidence index.\n"
        "Use the application text only to verify, refine, or reject what the belief state suggests.\n"
        f"{application_form_note}"
        f"{proposed_research_note}"
        "Do not score subcriteria outside the target rubric section.\n"
        "Identifier rules (STRICT):\n"
        "  - Top-level JSON keys MUST be parent sub ids from the allowed list (e.g. `g.4`, `pr.2`).\n"
        "  - NEVER use a signal id (e.g. `g.4.a`, `pr.2.b`) as a top-level key.\n"
        "  - Each top-level sub id maps to an object with `signals`, `used_chunk_ids`, `pros`, `drawbacks`.\n"
        "  - Inside `signals`, use the full signal id (e.g. `g.4.a`) as the key and a 0-5 integer as the value.\n"
        "  - Every signal belonging to a sub must appear under that sub's object — never split a sub's signals "
        "across multiple top-level keys, and never place signals from other subs here.\n"
        "Each signal score must be an integer from 0 to 5 inclusive (0,1,2,3,4,5).\n"
        "Scoring guide: 0=no evidence, 1=very weak, 2=weak, 3=moderate, 4=strong with only minor gaps, "
        "5=perfectly met: complete, specific, directly evidenced, and with no material caveats or missing detail.\n"
        "If a signal is missing from final_belief_state, you MUST still inspect application_text and score it. "
        "Do not omit a signal just because Stage 1 did not identify it.\n"
        "If your drawbacks mention any caveat, inference, missing detail, weak support, or partial support for a signal, "
        "that signal MUST NOT receive 5; use 4 or lower.\n"
        "`used_chunk_ids` must list all grounded chunk IDs that support the scoring.\n"
        "If evidence is missing, give a low score (0-3), not a high one (4-5).\n"
        "Keep pros and drawbacks concise and evidence-based."
    )
    examples = [e for e in (examples or []) if e and e.strip()]
    if examples:
        system += (
            "\n`calibration_examples` contains excerpts from PREVIOUSLY SUCCESSFUL "
            "applications for the same rubric section, retrieved from a reference corpus. "
            "Use them ONLY to calibrate what strong (4-5) evidence looks like for this "
            "section. They are NOT this applicant's text — never cite them as evidence, "
            "never copy them, and do not let their presence inflate scores."
        )
    user_payload = {
        "application_text": scoped_application_text,
        "criteria": stripped_criteria,
        "final_belief_state": final_belief_state,
    }
    if examples:
        # keep each exemplar short to bound prompt size
        user_payload["calibration_examples"] = [e.strip()[:800] for e in examples]
    example_sub = rubric_section["sub_criteria"][0] if rubric_section["sub_criteria"] else {}
    example_sub_id = example_sub.get("sub_id") or (target_sub_ids[0] if target_sub_ids else "g.4")
    example_signal_scores = {
        signal["sid"]: 5 if idx == 0 else 3
        for idx, signal in enumerate(example_sub.get("signals", []))
    } or {
        f"{example_sub_id}.a": 5,
        f"{example_sub_id}.b": 3,
    }
    example_output = {
        example_sub_id: {
            "signals": example_signal_scores,
            "used_chunk_ids": ["secxx__001", "secxx__002"],
            "pros": "The first signal is clearly supported by specific, grounded evidence.",
            "drawbacks": (
                "Weaker signals have only partial evidence or lack feasibility detail."
            ),
        }
    }
    user = (
        "Input JSON:\n"
        f"{json.dumps(user_payload, ensure_ascii=False, indent=2)}\n\n"
        "Scoring rules:\n"
        f"1. Output for all subcriteria under rubric section `{rubric_section['section_key']}`. "
        f"Top-level keys MUST come from: {target_sub_ids}.\n"
        "2. Prefer chunk IDs that already appear in the belief state when they support the score.\n"
        "3. Use only grounded chunk IDs from the application text / belief state.\n"
        "4. Each sub_id object must contain `signals`, `used_chunk_ids`, `pros`, and `drawbacks`.\n"
        "5. The `signals` object must include every signal id listed for that subcriterion, even when the score is 0.\n"
        "6. `pros` must describe the strongest directly evidenced strengths.\n"
        "7. `drawbacks` must describe missing evidence, weak support, caveats, or limitations. "
        "Mention strengths only when needed to explain why a non-zero score remains justified.\n"
        "8. Return JSON only.\n\n"
        "Output format example (shape only — use real sub/signal ids from the target section):\n"
        f"{json.dumps(example_output, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_scoring_schema(rubric_section: dict[str, Any], all_chunk_ids: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for sub in rubric_section["sub_criteria"]:
        signal_properties = {
            signal["sid"]: {
                "type": "integer",
                "enum": list(SCORER_ALLOWED_SCORES),
            }
            for signal in sub["signals"]
        }
        properties[sub["sub_id"]] = {
            "type": "object",
            "properties": {
                "signals": {
                    "type": "object",
                    "properties": signal_properties,
                    "required": list(signal_properties),
                    "additionalProperties": False,
                },
                "used_chunk_ids": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": all_chunk_ids,
                    },
                },
                "pros": {
                    "type": "string",
                },
                "drawbacks": {
                    "type": "string",
                },
            },
            "required": ["signals", "used_chunk_ids", "pros", "drawbacks"],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _normalize_score(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    normalized = int(value)
    return normalized if normalized in SCORER_ALLOWED_SCORES and normalized == value else 0


_BENIGN_NO_CAVEAT_PHRASES = (
    "no significant gaps",
    "no material gaps",
    "no gaps",
    "no significant caveats",
    "no material caveats",
    "no caveats",
    "no significant drawbacks",
    "no material drawbacks",
    "no significant limitations",
    "no material limitations",
    "no limitations",
    "no weaknesses",
)
_MATERIAL_CAVEAT_RE = re.compile(
    r"\b("
    r"not explicitly|not explicit|no explicit|not provided|not included|not detailed|"
    r"not fully|less detailed|missing|lacks?|lacking|limited|partial(?:ly)?|"
    r"inferred|implied|could be more|could improve|weak|unclear|caveats?|"
    r"gaps?|limitations?|drawbacks?|however|but"
    r")\b",
    flags=re.IGNORECASE,
)


def _has_material_caveat(text: str) -> bool:
    normalized = (text or "").lower()
    normalized = re.sub(
        r"\bno\s+(?:significant|material)?\s*(?:gaps?|caveats?|drawbacks?|limitations?|weaknesses)\b",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\bor\s+(?:significant|material)?\s*(?:gaps?|caveats?|drawbacks?|limitations?|weaknesses)\b",
        "",
        normalized,
    )
    for phrase in _BENIGN_NO_CAVEAT_PHRASES:
        normalized = normalized.replace(phrase, "")
    return bool(_MATERIAL_CAVEAT_RE.search(normalized))


def _sentence_mentions_signal_with_caveat(drawbacks: str, signal_id: str) -> bool:
    parts = re.split(r"(?<=[.!?])\s+|\n+", drawbacks or "")
    signal_re = re.compile(rf"\b{re.escape(signal_id)}\b", flags=re.IGNORECASE)
    return any(signal_re.search(part) and _has_material_caveat(part) for part in parts)


def _cap_perfect_scores_for_caveats(signals: dict[str, int], drawbacks: str) -> dict[str, int]:
    if not drawbacks or not _has_material_caveat(drawbacks):
        return signals

    capped = dict(signals)
    mentioned_signals = [
        sid
        for sid in capped
        if re.search(rf"\b{re.escape(sid)}\b", drawbacks, flags=re.IGNORECASE)
    ]
    if mentioned_signals:
        for sid in mentioned_signals:
            if capped.get(sid) == 5 and _sentence_mentions_signal_with_caveat(drawbacks, sid):
                capped[sid] = 4
        return capped

    return {sid: (4 if score == 5 else score) for sid, score in capped.items()}


def _collect_stage2_sub_sources(
    parsed: dict[str, Any],
    sub: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect all raw top-level entries in `parsed` that belong to this sub.

    Handles the canonical case (key == sub_id) as well as schema drift where
    the model keys entries by signal id (e.g. `g.4.a`) or nests multiple
    partial entries across keys for the same sub.
    """
    if not isinstance(parsed, dict):
        return []
    sources: list[dict[str, Any]] = []
    sub_id = sub["sub_id"]
    signal_ids = {signal["sid"] for signal in sub["signals"]}
    for key, value in parsed.items():
        if not isinstance(value, dict):
            continue
        if key == sub_id:
            sources.append(value)
            continue
        if key in signal_ids:
            sources.append(value)
            continue
        parent = key.rsplit(".", 1)[0] if "." in key else None
        if parent == sub_id:
            sources.append(value)
    return sources


def _normalize_model_section_output(
    parsed: dict[str, Any],
    rubric_section: dict[str, Any],
    all_chunk_ids: list[str],
) -> dict[str, dict[str, Any]]:
    allowed_ids = set(all_chunk_ids)
    normalized: dict[str, dict[str, Any]] = {}

    for sub in rubric_section["sub_criteria"]:
        sources = _collect_stage2_sub_sources(parsed, sub)

        merged_signals: dict[str, int] = {}
        used_ids_accum: list[str] = []
        pros_parts: list[str] = []
        drawbacks_parts: list[str] = []

        for raw_sub in sources:
            raw_signals = raw_sub.get("signals", {})
            if not isinstance(raw_signals, dict):
                raw_signals = {}
            for signal in sub["signals"]:
                sid = signal["sid"]
                if sid not in raw_signals:
                    continue
                score = _normalize_score(raw_signals.get(sid, 0))
                if score > merged_signals.get(sid, 0):
                    merged_signals[sid] = score

            raw_used_ids = raw_sub.get("used_chunk_ids", [])
            if isinstance(raw_used_ids, list):
                used_ids_accum.extend(
                    chunk_id
                    for chunk_id in raw_used_ids
                    if isinstance(chunk_id, str) and chunk_id in allowed_ids
                )

            pros = raw_sub.get("pros", raw_sub.get("strengths", raw_sub.get("advantages", "")))
            if isinstance(pros, str) and pros.strip():
                pros_parts.append(pros.strip())

            drawbacks = raw_sub.get("drawbacks", raw_sub.get("rationale", ""))
            if isinstance(drawbacks, str) and drawbacks.strip():
                drawbacks_parts.append(drawbacks.strip())

        signals: dict[str, int] = {}
        has_positive = False
        for signal in sub["signals"]:
            score = merged_signals.get(signal["sid"], 0)
            signals[signal["sid"]] = score
            has_positive = has_positive or score > 0

        used_chunk_ids = _dedupe_preserve_order(used_ids_accum)

        evidence_status = "ok"
        if has_positive and not used_chunk_ids:
            evidence_status = "invalid_evidence"
        elif len(used_chunk_ids) == 1:
            evidence_status = "sparse_evidence"

        pros = " ".join(_dedupe_preserve_order(pros_parts))
        drawbacks = " ".join(_dedupe_preserve_order(drawbacks_parts))
        signals = _cap_perfect_scores_for_caveats(signals, drawbacks)

        normalized[sub["sub_id"]] = {
            "signals": signals,
            "used_chunk_ids": used_chunk_ids,
            "evidence_status": evidence_status,
            "pros": pros,
            "drawbacks": drawbacks,
            "rationale": drawbacks,
            "missing_evidence": has_positive and not used_chunk_ids,
        }
    return normalized


def _build_evidence(
    used_chunk_ids: list[str],
    pool_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    evidence = []
    for chunk_id in used_chunk_ids:
        meta = pool_lookup.get(chunk_id)
        if not meta:
            continue
        evidence.append({
            "id": chunk_id,
            "section_id": chunk_id,
            "text": meta["text"],
            "section": meta["parser_section"],
            "source_path": meta["source_path"],
        })
    return evidence


def _aggregate_sub_criterion(
    sub: dict[str, Any],
    pool_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total_weight = sum(signal["weight"] for signal in sub["signals"]) or 1.0
    weighted_sum = sum(signal["score_0to5_weighted"] * signal["weight"] for signal in sub["signals"])
    score_10 = round((weighted_sum / (SCORER_MAX_SCORE * total_weight)) * 10, 2)
    confidence_score = CONFIDENCE_TO_SCORE[sub["confidence_label"]]
    avg_plausibility = _compat_plausibility(confidence_score)
    evidence = _build_evidence(sub["used_chunk_ids"], pool_lookup)
    return {
        **sub,
        "score_10": score_10,
        "quality_score_0to10": score_10,
        "avg_confidence_0to2": float(confidence_score),
        "avg_plausibility_0to5": avg_plausibility,
        "weak_signal_count": 1 if sub["confidence_label"] == "low_confidence" else 0,
        "evidence": evidence,
        "quality": (
            "good" if score_10 >= 7.5 else
            "mixed" if score_10 >= 4 else
            "weak"
        ),
        "exists": "yes" if score_10 > 0 else "no",
    }


def _apply_score_caps(
    features: dict[str, dict[str, Any]],
    sub_id_caps: dict[str, float],
) -> None:
    """
    Cap specific sub-criteria scores in-place and recompute section aggregates.
    Called after _aggregate_section so all score_10 values are already on 0–10.
    """
    if not sub_id_caps:
        return
    for section_data in features.values():
        modified = False
        sub_criteria = section_data.get("sub_criteria", [])
        for sub in sub_criteria:
            cap = sub_id_caps.get(sub["sub_id"])
            if cap is not None and sub.get("score_10", 0) > cap:
                sub["score_10"] = cap
                sub["quality_score_0to10"] = cap
                sub["quality"] = (
                    "good" if cap >= 7.5 else
                    "mixed" if cap >= 4 else
                    "weak"
                )
                modified = True
        if not modified:
            continue
        # Recompute section-level aggregates from capped sub-criteria
        scored = [s for s in sub_criteria if s.get("counts_toward_section_average", True)]
        denom = scored or sub_criteria
        total_weight = sum(s["weight"] for s in denom) or 1.0
        new_score_10 = round(sum(s["score_10"] * s["weight"] for s in denom) / total_weight, 2)
        section_data["score_10"] = new_score_10
        ov = section_data.get("overall", {})
        ov["score_10"] = new_score_10
        ov["final_score_0to100"] = round(new_score_10 * 10, 2)
        ov["quality_score_0to100"] = round(new_score_10 * 10, 2)
        ov["quality_score_avg_0to10"] = new_score_10
        ov["good_items"] = sum(1 for s in denom if s["score_10"] >= 7.5)


def _aggregate_section(section: dict[str, Any], pool_lookup: dict[str, dict[str, str]]) -> dict[str, Any]:
    sub_criteria = [_aggregate_sub_criterion(sub, pool_lookup) for sub in section["sub_criteria"]]
    scored_sub_criteria = [
        sub for sub in sub_criteria
        if sub.get("counts_toward_section_average", True)
    ]
    scoring_denominator = scored_sub_criteria or sub_criteria
    total_weight = sum(sub["weight"] for sub in scoring_denominator) or 1.0
    weighted_score = sum(sub["score_10"] * sub["weight"] for sub in scoring_denominator)
    score_10 = round(weighted_score / total_weight, 2)
    total_items = len(sub_criteria)
    scored_items = len(scoring_denominator)
    signal_count = sum(len(sub["signals"]) for sub in sub_criteria)
    evidence_count = sum(sub["evidence_count"] for sub in sub_criteria)
    good_items = sum(1 for sub in scoring_denominator if sub["score_10"] >= 7.5)
    positive_items = sum(1 for sub in scoring_denominator if sub["score_10"] > 0)
    high_confidence_count = sum(1 for sub in scoring_denominator if sub["confidence_label"] == "high_confidence")
    medium_confidence_count = sum(1 for sub in scoring_denominator if sub["confidence_label"] == "medium_confidence")
    low_confidence_count = sum(1 for sub in scoring_denominator if sub["confidence_label"] == "low_confidence")
    avg_confidence_0to2 = round(
        sum(CONFIDENCE_TO_SCORE[sub["confidence_label"]] for sub in scoring_denominator) / max(1, scored_items),
        2,
    )
    avg_plausibility = _compat_plausibility(avg_confidence_0to2)
    overall = {
        "score_10": score_10,
        "final_score_0to100": round(score_10 * 10, 2),
        "coverage_score_0to100": round((positive_items / max(1, scored_items)) * 100, 2),
        "quality_score_0to100": round(score_10 * 10, 2),
        "evidence_score_0to100": _compat_evidence_score(avg_confidence_0to2),
        "quality_score_avg_0to10": score_10,
        "avg_confidence_0to2": avg_confidence_0to2,
        "avg_plausibility_0to5": avg_plausibility,
        "high_confidence_subcriterion_count": high_confidence_count,
        "medium_confidence_subcriterion_count": medium_confidence_count,
        "low_confidence_subcriterion_count": low_confidence_count,
        "weak_signal_count": low_confidence_count,
        "total_items": total_items,
        "scored_items": scored_items,
        "signal_count": signal_count,
        "good_items": good_items,
        "positive_items": positive_items,
        "expected_items": scored_items,
        "evidence_count": evidence_count,
        "target_evidence_per_item": USED_CHUNK_MAX,
    }
    return {
        "score_10": score_10,
        "avg_confidence_0to2": avg_confidence_0to2,
        "avg_plausibility_0to5": avg_plausibility,
        "high_confidence_subcriterion_count": high_confidence_count,
        "medium_confidence_subcriterion_count": medium_confidence_count,
        "low_confidence_subcriterion_count": low_confidence_count,
        "weak_signal_count": low_confidence_count,
        "scored_items": scored_items,
        "sub_criteria": sub_criteria,
        "criteria": sub_criteria,
        "overall": overall,
    }


def _aggregate_overall(
    features: dict[str, dict[str, Any]],
    section_weights: dict[str, float],
    *,
    excluded_sections: set[str] | None = None,
) -> dict[str, Any]:
    if not features:
        return {
            "score_10": 0,
            "final_score_0to100": 0,
            "quality_score_0to100": 0,
            "coverage_score_0to100": 0,
            "evidence_score_0to100": 0,
            "quality_score_avg_0to10": 0,
            "avg_confidence_0to2": 0,
            "avg_plausibility_0to5": 0,
            "high_confidence_subcriterion_count": 0,
            "medium_confidence_subcriterion_count": 0,
            "low_confidence_subcriterion_count": 0,
            "weak_signal_count": 0,
            "total_items": 0,
            "scored_items": 0,
            "signal_count": 0,
            "evidence_count": 0,
            "target_evidence_per_item": USED_CHUNK_MAX,
        }

    scoring_features = {
        k: v for k, v in features.items()
        if k not in (excluded_sections or set())
    }
    if not scoring_features:
        scoring_features = features

    total_weight = sum(section_weights.get(key, 1.0) for key in scoring_features) or 1.0
    weighted_score = sum(scoring_features[key]["score_10"] * section_weights.get(key, 1.0) for key in scoring_features)
    score_10 = round(weighted_score / total_weight, 2)
    total_items = sum(section["overall"]["total_items"] for section in features.values())
    scored_items = sum(section["overall"].get("scored_items", section["overall"]["total_items"]) for section in features.values())
    signal_count = sum(section["overall"]["signal_count"] for section in features.values())
    evidence_count = sum(section["overall"]["evidence_count"] for section in features.values())
    good_items = sum(section["overall"]["good_items"] for section in scoring_features.values())
    positive_items = sum(section["overall"]["positive_items"] for section in scoring_features.values())
    high_confidence_count = sum(section["overall"]["high_confidence_subcriterion_count"] for section in features.values())
    medium_confidence_count = sum(section["overall"]["medium_confidence_subcriterion_count"] for section in features.values())
    low_confidence_count = sum(section["overall"]["low_confidence_subcriterion_count"] for section in features.values())
    avg_confidence_0to2 = round(
        sum(
            section["overall"]["avg_confidence_0to2"]
            * section["overall"].get("scored_items", section["overall"]["total_items"])
            for section in features.values()
        )
        / max(1, scored_items),
        2,
    )
    return {
        "score_10": score_10,
        "final_score_0to100": round(score_10 * 10, 2),
        "quality_score_0to100": round(score_10 * 10, 2),
        "coverage_score_0to100": round((positive_items / max(1, scored_items)) * 100, 2),
        "evidence_score_0to100": _compat_evidence_score(avg_confidence_0to2),
        "quality_score_avg_0to10": score_10,
        "avg_confidence_0to2": avg_confidence_0to2,
        "avg_plausibility_0to5": _compat_plausibility(avg_confidence_0to2),
        "high_confidence_subcriterion_count": high_confidence_count,
        "medium_confidence_subcriterion_count": medium_confidence_count,
        "low_confidence_subcriterion_count": low_confidence_count,
        "weak_signal_count": low_confidence_count,
        "total_items": total_items,
        "scored_items": scored_items,
        "signal_count": signal_count,
        "evidence_count": evidence_count,
        "good_items": good_items,
        "positive_items": positive_items,
        "expected_items": scored_items,
        "target_evidence_per_item": USED_CHUNK_MAX,
    }


def _build_scored_section(
    rubric_section: dict[str, Any],
    normalized_section: dict[str, dict[str, Any]],
    chunk_order: dict[str, int],
    *,
    excluded_sub_ids: set[str] | None = None,
) -> dict[str, Any]:
    section_subs: list[dict[str, Any]] = []
    for sub in rubric_section["sub_criteria"]:
        normalized_sub = normalized_section.get(sub["sub_id"], {})
        used_chunk_ids = _sort_chunk_ids(normalized_sub.get("used_chunk_ids", []), chunk_order)
        evidence_status = normalized_sub.get("evidence_status", "ok")
        signals: list[dict[str, Any]] = []
        for signal in sub["signals"]:
            score = int(normalized_sub.get("signals", {}).get(signal["sid"], 0))
            signals.append({
                "sid": signal["sid"],
                "signal_text": signal["text"],
                "weight": signal["weight"],
                "score": score,
                "score_0to5_raw": float(score),
                "score_0to5_weighted": float(score),
            })
        pros = normalized_sub.get("pros", "")
        drawbacks = normalized_sub.get("drawbacks", normalized_sub.get("rationale", ""))
        missing_evidence = bool(normalized_sub.get("missing_evidence"))
        doc_type_excluded = bool(excluded_sub_ids and sub["sub_id"] in excluded_sub_ids)
        counts_toward_section_average = not _is_if_applicable_subcriterion(sub) and not doc_type_excluded
        entry: dict[str, Any] = {
            "sub_id": sub["sub_id"],
            "name": sub["name"],
            "definition": sub["definition"],
            "group_name": sub.get("group_name"),
            "weight": sub["weight"],
            "used_chunk_ids": used_chunk_ids,
            "evidence_count": len(used_chunk_ids),
            "evidence_status": evidence_status,
            "confidence_gap": 0.0,
            "confidence_label": _confidence_label(0.0),
            "signals": signals,
            "pros": pros,
            "drawbacks": drawbacks,
            "rationale": drawbacks,
            "missing_evidence": missing_evidence,
            "if_applicable": not counts_toward_section_average,
            "counts_toward_section_average": counts_toward_section_average,
        }
        if doc_type_excluded:
            entry["excluded_reason"] = "not_applicable_for_doc_type"
        section_subs.append(entry)

    return {
        "human_name": rubric_section["human_name"],
        "section_key": rubric_section["section_key"],
        "weight": rubric_section["weight"],
        "sub_criteria": section_subs,
    }


def _write_artifacts(
    *,
    artifacts_dir: str | Path | None,
    doc_id: str,
    pool_lookup: dict[str, dict[str, str]],
    pool_index_text: str,
    stage1_raw_by_section: dict[str, str],
    stage1_updates: list[dict[str, Any]],
    final_belief_state: dict[str, Any],
    stage2_raw_by_section: dict[str, str],
    normalized_sections: list[dict[str, Any]],
) -> dict[str, str]:
    if artifacts_dir is None:
        return {}

    artifacts = write_pool_artifacts(
        pool_lookup=pool_lookup,
        pool_index_text=pool_index_text,
        artifacts_dir=artifacts_dir,
        doc_id=doc_id,
    )
    artifacts_path = Path(artifacts_dir)
    stage1_raw_path = artifacts_path / f"{doc_id}_stage1_raw.json"
    stage1_updates_path = artifacts_path / f"{doc_id}_stage1_updates.json"
    belief_state_path = artifacts_path / f"{doc_id}_belief_state.json"
    stage2_raw_path = artifacts_path / f"{doc_id}_stage2_raw.json"
    scored_sections_path = artifacts_path / f"{doc_id}_scored_sections.json"

    stage1_raw_path.write_text(json.dumps(stage1_raw_by_section, ensure_ascii=False, indent=2), encoding="utf-8")
    stage1_updates_path.write_text(json.dumps(stage1_updates, ensure_ascii=False, indent=2), encoding="utf-8")
    belief_state_path.write_text(json.dumps(final_belief_state, ensure_ascii=False, indent=2), encoding="utf-8")
    stage2_raw_path.write_text(json.dumps(stage2_raw_by_section, ensure_ascii=False, indent=2), encoding="utf-8")
    scored_sections_path.write_text(json.dumps(normalized_sections, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        **artifacts,
        "stage1_raw_response": str(stage1_raw_path),
        "stage1_updates": str(stage1_updates_path),
        "belief_state": str(belief_state_path),
        "stage2_raw_response": str(stage2_raw_path),
        "scored_sections": str(scored_sections_path),
    }


def _write_failure_artifacts(
    *,
    artifacts_dir: str | Path | None,
    doc_id: str,
    pool_lookup: dict[str, dict[str, str]],
    pool_index_text: str,
    stage1_raw_by_section: dict[str, str],
    stage1_updates: list[dict[str, Any]],
    final_belief_state: dict[str, Any],
    stage2_raw_by_section: dict[str, str],
    normalized_sections: list[dict[str, Any]],
    failure_label: str,
    raw_response: str,
    scorer_client: Any | None = None,
) -> dict[str, str]:
    artifact_paths = _write_artifacts(
        artifacts_dir=artifacts_dir,
        doc_id=doc_id,
        pool_lookup=pool_lookup,
        pool_index_text=pool_index_text,
        stage1_raw_by_section=stage1_raw_by_section,
        stage1_updates=stage1_updates,
        final_belief_state=final_belief_state,
        stage2_raw_by_section=stage2_raw_by_section,
        normalized_sections=normalized_sections,
    )
    if artifacts_dir is None:
        return artifact_paths

    artifacts_path = Path(artifacts_dir)
    failure_raw_path = artifacts_path / f"{doc_id}_{failure_label}_raw.txt"
    failure_raw_path.write_text(raw_response, encoding="utf-8")
    artifact_paths["failed_raw_response"] = str(failure_raw_path)
    if getattr(scorer_client, "last_response_body", None) is not None:
        failure_body_path = artifacts_path / f"{doc_id}_{failure_label}_response_body.json"
        failure_body_path.write_text(
            json.dumps(scorer_client.last_response_body, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifact_paths["failed_response_body"] = str(failure_body_path)
    return artifact_paths


def score_application_base(
    *,
    application: dict[str, Any],
    criteria_path: str | Path,
    doc_id: str | None,
    scorer_client: Any,
    artifacts_dir: str | Path | None = None,
    use_retrieval: bool = True,
    evidence_top_k: int = 8,
    corpus_index: str | None = None,
    fewshot_n: int = 2,
) -> dict[str, Any]:
    doc_type = (application.get("doc_type") or "").lower()
    excluded_sections = OVERALL_EXCLUDED_SECTIONS_BY_DOC_TYPE.get(doc_type, set())
    excluded_sub_ids = SECTION_EXCLUDED_SUB_IDS_BY_DOC_TYPE.get(doc_type, set())

    rubric_sections = load_rubric(criteria_path)
    stripped_criteria = _strip_rubric_for_prompt(rubric_sections)
    stage1_criteria = _build_stage1_criteria_view(rubric_sections)
    pool_data = build_chunk_pool(application, max_chars=MAX_CHARS)
    pool_lookup = pool_data["pool_lookup"]
    pool_index_text = pool_data["pool_index_text"]
    chunk_order = _chunk_order_map(pool_lookup)
    all_chunk_ids = list(pool_lookup)

    belief_state = _initial_belief_state(rubric_sections)
    stage1_inputs = _section_inputs(pool_data["section_chunk_ids"], pool_lookup, chunk_order)
    stage1_raw_by_section: dict[str, str] = {}
    stage1_updates: list[dict[str, Any]] = []
    json_retry_events: list[dict[str, Any]] = []

    for application_section in stage1_inputs:
        section_name = application_section["section_name"]
        section_chunk_ids = list(application_section["section_content"])
        messages = build_section_evidence_messages(
            application_section=application_section,
            stage1_criteria=stage1_criteria,
            current_belief_state=belief_state,
        )
        schema = build_section_evidence_schema(rubric_sections, section_chunk_ids)
        try:
            raw_stage1, parsed_stage1, retry_count = _generate_json_with_parse_retry(
                scorer_client,
                messages,
                schema=schema,
                max_tokens=STAGE1_MAX_TOKENS,
            )
        except JsonRetryError as exc:
            raw_stage1 = exc.raw_response
            stage1_raw_by_section[section_name] = raw_stage1
            response_meta = _response_meta_summary(scorer_client)
            failure_artifacts = _write_failure_artifacts(
                artifacts_dir=artifacts_dir,
                doc_id=doc_id or "unknown",
                pool_lookup=pool_lookup,
                pool_index_text=pool_index_text,
                stage1_raw_by_section=stage1_raw_by_section,
                stage1_updates=stage1_updates,
                final_belief_state=belief_state,
                stage2_raw_by_section={},
                normalized_sections=[],
                failure_label=f"stage1_{section_name.lower().replace(' ', '_')}",
                raw_response=raw_stage1,
                scorer_client=scorer_client,
            )
            raise ValueError(
                f"Stage 1 returned invalid JSON for section {section_name} after {exc.attempts} attempts. "
                f"Response preview: {_response_preview(raw_stage1)!r}. "
                f"Response meta: {response_meta or 'unavailable'}. "
                f"Raw outputs written to: {failure_artifacts.get('failed_raw_response')}"
            ) from exc.original
        stage1_raw_by_section[section_name] = raw_stage1
        if retry_count:
            json_retry_events.append({
                "stage": "stage1",
                "section": section_name,
                "retry_count": retry_count,
            })
        normalized_update = _normalize_section_evidence_output(
            parsed_stage1,
            rubric_sections,
            section_chunk_ids,
            section_name,
        )
        belief_state = _merge_belief_state(belief_state, normalized_update)
        stage1_updates.append({
            **normalized_update,
            "missing_signals_after": list(belief_state["missing_signals"]),
        })

    # Hybrid evidence retrieval over an ephemeral in-memory index of the current
    # application (reuses swxy's Dealer). None → full-text fallback.
    retriever = _build_current_retriever(pool_lookup, doc_id or "current") if use_retrieval else None
    # Persistent ES corpus for few-shot exemplars (opt-in via corpus_index).
    corpus_retriever = _build_corpus_retriever(corpus_index)
    retrieval_used = False
    fewshot_used = False

    stage2_raw_by_section: dict[str, str] = {}
    stage2_scope_by_section: dict[str, list[str]] = {}
    sections: list[dict[str, Any]] = []
    for rubric_section in rubric_sections:
        section_key = rubric_section["section_key"]
        scoped_text, scoped_parser_sections = _retrieval_scope(
            retriever, rubric_section, pool_lookup, chunk_order, evidence_top_k
        )
        if scoped_text is None:
            scoped_text = _build_full_application_text(pool_lookup, chunk_order)
            scoped_parser_sections = []
        else:
            retrieval_used = True
        stage2_scope_by_section[section_key] = scoped_parser_sections
        examples = _fewshot_examples(
            corpus_retriever, rubric_section, doc_id or "current", fewshot_n
        )
        if examples:
            fewshot_used = True
        messages = build_final_scoring_messages(
            rubric_section=rubric_section,
            stripped_criteria=stripped_criteria,
            final_belief_state=belief_state,
            scoped_application_text=scoped_text,
            scoped_parser_sections=scoped_parser_sections,
            examples=examples,
        )
        schema = build_scoring_schema(rubric_section, all_chunk_ids)
        try:
            raw_stage2, parsed_stage2, retry_count = _generate_json_with_parse_retry(
                scorer_client,
                messages,
                schema=schema,
                max_tokens=STAGE2_MAX_TOKENS,
            )
        except JsonRetryError as exc:
            raw_stage2 = exc.raw_response
            stage2_raw_by_section[section_key] = raw_stage2
            response_meta = _response_meta_summary(scorer_client)
            failure_artifacts = _write_failure_artifacts(
                artifacts_dir=artifacts_dir,
                doc_id=doc_id or "unknown",
                pool_lookup=pool_lookup,
                pool_index_text=pool_index_text,
                stage1_raw_by_section=stage1_raw_by_section,
                stage1_updates=stage1_updates,
                final_belief_state=belief_state,
                stage2_raw_by_section=stage2_raw_by_section,
                normalized_sections=sections,
                failure_label=f"stage2_{section_key}",
                raw_response=raw_stage2,
                scorer_client=scorer_client,
            )
            raise ValueError(
                f"Stage 2 returned invalid JSON for section {section_key} after {exc.attempts} attempts. "
                f"Response preview: {_response_preview(raw_stage2)!r}. "
                f"Response meta: {response_meta or 'unavailable'}. "
                f"Raw outputs written to: {failure_artifacts.get('failed_raw_response')}"
            ) from exc.original
        stage2_raw_by_section[section_key] = raw_stage2
        if retry_count:
            json_retry_events.append({
                "stage": "stage2",
                "section": section_key,
                "retry_count": retry_count,
            })

        normalized_stage2 = _normalize_model_section_output(parsed_stage2, rubric_section, all_chunk_ids)
        sections.append(_build_scored_section(
            rubric_section,
            normalized_stage2,
            chunk_order,
            excluded_sub_ids=excluded_sub_ids,
        ))

    artifact_paths = _write_artifacts(
        artifacts_dir=artifacts_dir,
        doc_id=doc_id or "unknown",
        pool_lookup=pool_lookup,
        pool_index_text=pool_index_text,
        stage1_raw_by_section=stage1_raw_by_section,
        stage1_updates=stage1_updates,
        final_belief_state=belief_state,
        stage2_raw_by_section=stage2_raw_by_section,
        normalized_sections=sections,
    )

    features = {
        section["section_key"]: _aggregate_section(section, pool_lookup)
        for section in sections
    }
    sub_id_caps = SUB_ID_SCORE_CAPS_BY_DOC_TYPE.get(doc_type, {})
    _apply_score_caps(features, sub_id_caps)
    section_weights = {section["section_key"]: section["weight"] for section in sections}

    return {
        "doc_id": doc_id or "unknown",
        "run_info": {
            "ran_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
            "retrieval_method": (
                "belief_then_inmem_hybrid_retrieval_scoring"
                if retrieval_used else "all_sections_belief_then_fulltext_scoring"
            ),
            "scorer_model": getattr(scorer_client, "model_name", "unknown"),
        },
        "pool_size": len(pool_lookup),
        "pool_lookup": pool_lookup,
        "section_chunk_ids": pool_data["section_chunk_ids"],
        "belief_state": belief_state,
        "features": features,
        "overall": _aggregate_overall(features, section_weights, excluded_sections=excluded_sections),
        "debug": {
            "scoring_contract_version": "section_evidence_belief_single_model_v1",
            "doc_type": doc_type or None,
            "excluded_sections": sorted(excluded_sections),
            "excluded_sub_ids": sorted(excluded_sub_ids),
            "stage1_section_updates": stage1_updates,
            "json_retry_events": json_retry_events,
            "retrieval_used": retrieval_used,
            "fewshot_used": fewshot_used,
            "artifacts": artifact_paths,
        },
    }
