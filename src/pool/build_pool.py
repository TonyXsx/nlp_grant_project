from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── make the vendored DeepDOC engine (src/deepdoc_engine) importable ──────────
_SRC = Path(__file__).resolve().parent.parent  # …/src
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Reuse swxy's chunking primitives wholesale (token-budget merge, table
# tokenisation, BM25-field tokenisation, bullet/heading detection).
from deepdoc_engine.rag.nlp import (  # noqa: E402
    naive_merge,
    tokenize,
    tokenize_table,
    bullets_category,
    not_bullet,
    BULLET_PATTERN,
)
from deepdoc_engine.rag.nlp import rag_tokenizer  # noqa: E402
from deepdoc_engine.rag.utils import num_tokens_from_string  # noqa: E402

# Kept for backward-compatible imports (pipeline.py imports MAX_CHARS); no longer
# used for chunking — chunk size is now token-based (CHUNK_TOKEN_NUM).
MAX_CHARS = 1200
# Token budget per chunk. swxy default is 128; we use a larger budget because
# rubric-signal retrieval benefits from slightly bigger evidence units.
CHUNK_TOKEN_NUM = 256
# English sentence/clause delimiters (swxy default was CJK punctuation). Chunk
# boundaries fall after these so sentences are never cut mid-way.
DELIMITER = "\n.!?;:"
# Hard safety cap (characters) for a single delimiter-free run of text.
_HARD_CHAR_CAP = 4000

# Sentence splitter: split *after* any delimiter char, keeping it attached to
# the preceding sentence. (DELIMITER chars are all char-class-safe.)
_SENT_SPLIT_RE = re.compile(r"(?<=[\n.!?;:])")
# DeepDOC position tag format from RAGFlowPdfParser._line_tag:
#   @@<page(-page)>\t<x0>\t<x1>\t<top>\t<bottom>##
_POS_TAG_RE = re.compile(r"@@[0-9-]+\t[0-9.\t]+##")
# Detect an HTML table (DeepDOC restores budget/method tables as <table> HTML).
_TABLE_RE = re.compile(r"<table[\s>]", re.IGNORECASE)

APPLICATION_DETAILS_KEY = "APPLICATION DETAILS"
SUMMARY_BUDGET_KEY = "SUMMARY BUDGET"
APPLICATION_CONTEXT_SECTION = "Application Context"
APPLICATION_FORM_ANALYSIS_SECTION = "Application Form Analysis"
PLAIN_ENGLISH_ANALYSIS_SECTION = "Plain English NLP Analysis"


@dataclass(frozen=True)
class PoolChunk:
    chunk_id: str
    text: str
    parser_section: str
    source_path: str
    # ── swxy-style enrichment (used by hybrid BM25 + dense retrieval) ──
    content_ltks: str = ""          # coarse-grained tokens (BM25 main field)
    content_sm_ltks: str = ""       # fine-grained tokens (BM25 secondary)
    title_tks: str = ""             # tokenised section/title context
    token_count: int = 0
    is_table: bool = False
    # DeepDOC layout positions [(page, x0, x1, top, bottom), ...] when available
    # (groundwork for highlighting evidence on the source PDF).
    position: Optional[list] = None


def _slug_initials(name: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", name.lower())
    initials = "".join(part[0] for part in parts if part)
    return f"sec{initials or 'x'}"


def _stringify_leaf(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, indent=2).strip()


def _extract_positions(text: str) -> tuple[str, Optional[list]]:
    """Pull DeepDOC ``@@page\\tx0\\tx1\\ttop\\tbottom##`` tags out of ``text``.

    Returns ``(clean_text, positions)`` where positions is a list of
    ``(page, x0, x1, top, bottom)`` tuples, or ``None`` when no tags are present
    (the rule-based parsers emit no positions). Stripping the tags also keeps the
    text the LLM sees clean.
    """
    tags = _POS_TAG_RE.findall(text)
    if not tags:
        return text, None
    positions: list = []
    for tag in tags:
        body = tag.strip("@").strip("#")
        parts = body.split("\t")
        if len(parts) < 5:
            continue
        try:
            x0, x1, top, bottom = (float(parts[1]), float(parts[2]),
                                   float(parts[3]), float(parts[4]))
            for p in parts[0].split("-"):
                positions.append((int(p), x0, x1, top, bottom))
        except (ValueError, IndexError):
            continue
    clean = _POS_TAG_RE.sub("", text).strip()
    return clean, (positions or None)


def _protect_tags(text: str) -> tuple[str, dict]:
    """Replace DeepDOC @@..## position tags with split-safe placeholders so the
    sentence splitter (which breaks on '.') doesn't shatter the float coords
    inside them. Returns (protected_text, {placeholder: original_tag})."""
    holders: dict[str, str] = {}
    for i, tag in enumerate(_POS_TAG_RE.findall(text)):
        holder = f"\x00{i}\x00"
        text = text.replace(tag, holder, 1)
        holders[holder] = tag
    return text, holders


def _restore_tags(text: str, holders: dict) -> str:
    for holder, tag in holders.items():
        text = text.replace(holder, tag)
    return text


def _split_sentences(text: str, delimiter: str = DELIMITER) -> list[str]:
    """Split text into delimiter-bounded pieces, keeping the delimiter attached
    so a chunk boundary never cuts mid-sentence. Mirrors how swxy's parsers emit
    small section pieces that ``naive_merge`` then merges by token budget."""
    if not text:
        return []
    pieces: list[str] = []
    for part in _SENT_SPLIT_RE.split(text):
        part = part.strip()
        if not part:
            continue
        # Trailing space so naive_merge (which concatenates pieces with no
        # separator) keeps sentences readable instead of "fused.LikeThis".
        if len(part) > _HARD_CHAR_CAP:  # safety: a single delimiter-free run
            for i in range(0, len(part), _HARD_CHAR_CAP):
                seg = part[i:i + _HARD_CHAR_CAP].strip()
                if seg:
                    pieces.append(seg + " ")
        else:
            pieces.append(part + " ")
    return pieces


def _heading_flags(pieces: list[str]) -> list[bool]:
    """Mark which pieces are headings/numbered items, using swxy's BULLET_PATTERN
    + bullets_category (picks the dominant bullet style in this text)."""
    if not pieces:
        return []
    bull = bullets_category(pieces)
    if bull < 0:
        return [False] * len(pieces)
    pats = BULLET_PATTERN[bull]
    flags = []
    for piece in pieces:
        s = piece.strip()
        is_head = any(re.match(p, s) for p in pats) and not not_bullet(s)
        flags.append(is_head)
    return flags


def _chunk_text(text: str, chunk_token_num: int = CHUNK_TOKEN_NUM,
                delimiter: str = DELIMITER) -> list[str]:
    """Token-aware, sentence-respecting, heading-bound chunking.

    1. split into sentence pieces (no mid-sentence cuts);
    2. group so each heading starts a new group (binds a heading to its body and
       prevents the next heading from being swallowed into the previous body);
    3. ``naive_merge`` each group into <= chunk_token_num token chunks.
    """
    # Protect DeepDOC position tags so '.' in their coords doesn't get split,
    # so each resulting chunk keeps the tags of the lines it covers (→ per-chunk
    # page positions). Restored on each chunk at the end.
    protected, holders = _protect_tags(text)
    pieces = _split_sentences(protected, delimiter)
    if not pieces:
        return []
    flags = _heading_flags(pieces)
    groups: list[list[str]] = []
    current: list[str] = []
    for piece, is_head in zip(pieces, flags):
        if is_head and current:
            groups.append(current)
            current = []
        current.append(piece)
    if current:
        groups.append(current)

    chunks: list[str] = []
    for group in groups:
        merged = naive_merge([(s, "") for s in group], chunk_token_num, delimiter)
        chunks.extend(_restore_tags(c.strip(), holders) for c in merged if c and c.strip())
    return chunks


def _child_path(path: list[str], key: Any) -> list[str]:
    if isinstance(key, int):
        return [*path, f"[{key}]"]
    return [*path, str(key)]


def _iter_leaves(value: Any, path: list[str], parser_section: str) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        out: list[tuple[str, str]] = []
        for key, child in value.items():
            out.extend(_iter_leaves(child, _child_path(path, key), parser_section))
        return out
    if isinstance(value, list):
        out = []
        for idx, child in enumerate(value):
            out.extend(_iter_leaves(child, _child_path(path, idx), parser_section))
        return out

    text = _stringify_leaf(value)
    return [(text, " > ".join(path))] if text else []


def _format_combined_context(entries: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"{source_path}:\n{text}"
        for text, source_path in entries
        if text.strip()
    )


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _normalized_lines(text: str) -> list[str]:
    lines: list[str] = []
    for line in (text or "").splitlines():
        clean = re.sub(r"\s+", " ", line).strip().lower()
        if len(clean.split()) >= 4:
            lines.append(clean)
    return lines


def _sentence_tokens(text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        clean = re.sub(r"[^a-z0-9 ]+", "", sentence.lower()).strip()
        if len(clean.split()) >= 5:
            sentences.append(clean)
    return sentences


def _duplication_rate(items: list[str]) -> float:
    if not items:
        return 0.0
    return round(1 - (len(set(items)) / len(items)), 3)


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r"\b[a-z]{3,}\b", (text_a or "").lower()))
    words_b = set(re.findall(r"\b[a-z]{3,}\b", (text_b or "").lower()))
    if not words_a or not words_b:
        return 0.0
    return round(len(words_a & words_b) / len(words_a | words_b), 3)


def _sentences_for_readability(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or "")
        if sentence.strip()
    ]


def _words_for_readability(text: str) -> list[str]:
    return re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text or "")


def _syllable_count(word: str) -> int:
    clean = re.sub(r"[^a-z]", "", (word or "").lower())
    if not clean:
        return 0
    clean = re.sub(r"e$", "", clean)
    groups = re.findall(r"[aeiouy]+", clean)
    return max(1, len(groups))


def _flesch_kincaid_grade(text: str) -> float:
    sentences = _sentences_for_readability(text)
    words = _words_for_readability(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllable_count(word) for word in words)
    return round(
        0.39 * (len(words) / len(sentences))
        + 11.8 * (syllables / len(words))
        - 15.59,
        2,
    )


def _flesch_reading_ease(text: str) -> float:
    sentences = _sentences_for_readability(text)
    words = _words_for_readability(text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllable_count(word) for word in words)
    return round(
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6 * (syllables / len(words)),
        2,
    )


def _technical_terms(text: str) -> list[str]:
    stopwords = {
        "because", "between", "different", "important", "research", "summary",
        "treatment", "patients", "people", "project", "condition", "currently",
    }
    terms: list[str] = []
    for word in _words_for_readability(text):
        clean = word.lower().strip("'")
        if len(clean) < 11 or clean in stopwords:
            continue
        if clean not in terms:
            terms.append(clean)
    return terms


def _format_plain_english_analysis(
    section_chunk_ids: dict[str, list[str]],
    pool_lookup: dict[str, dict[str, str]],
) -> str:
    summary_section = None
    for candidate in ("Plain English Summary of Research", "Plain English Summary"):
        if candidate in section_chunk_ids:
            summary_section = candidate
            break
    if not summary_section:
        return ""

    summary_text = "\n\n".join(
        pool_lookup[chunk_id]["text"]
        for chunk_id in section_chunk_ids.get(summary_section, [])
    )
    if not summary_text.strip():
        return ""

    detailed_text = "\n\n".join(
        pool_lookup[chunk_id]["text"]
        for chunk_id in section_chunk_ids.get("Detailed Research Plan", [])
    )
    words = _words_for_readability(summary_text)
    sentences = _sentences_for_readability(summary_text)
    sentence_lengths = [
        len(_words_for_readability(sentence))
        for sentence in sentences
    ]
    avg_sentence_length = round(sum(sentence_lengths) / len(sentence_lengths), 2) if sentence_lengths else 0.0
    long_sentence_ratio = round(
        sum(1 for length in sentence_lengths if length >= 30) / len(sentence_lengths),
        3,
    ) if sentence_lengths else 0.0
    terms = _technical_terms(summary_text)
    jargon_density = round((len(terms) / len(words)) * 100, 2) if words else 0.0
    alignment = _jaccard_similarity(summary_text, detailed_text) if detailed_text else 0.0
    coverage_terms = {
        "problem": r"\b(problem|condition|burden|currently|uncertainty|need)\b",
        "objectives": r"\b(aim|objective|will|project|develop|evaluate|identify)\b",
        "methods": r"\b(method|data|dataset|model|interview|review|analysis|study)\b",
        "beneficiaries": r"\b(patient|people|clinician|public|service|nhs)\b",
        "impact": r"\b(benefit|improve|impact|personalised|save|reduce|support)\b",
    }
    coverage_hits = [
        label for label, pattern in coverage_terms.items()
        if re.search(pattern, summary_text, flags=re.IGNORECASE)
    ]

    return "\n".join([
        "Plain English Summary NLP analysis derived from parser output.",
        "Use these metrics as supporting evidence for pr.1, but also read the raw Plain English Summary and "
        "Detailed Research Plan in application_text. Do not claim readability, jargon, alignment, or sentence "
        "coherence evidence is missing solely because Stage 1 did not provide NLP/coherence findings.",
        "",
        "Readability and sentence structure metrics:",
        f"- source_section={summary_section}",
        f"- word_count={len(words)}",
        f"- sentence_count={len(sentences)}",
        f"- avg_sentence_length_words={avg_sentence_length}",
        f"- long_sentence_ratio_30_words={long_sentence_ratio}",
        f"- flesch_kincaid_grade_estimate={_flesch_kincaid_grade(summary_text)}",
        f"- flesch_reading_ease_estimate={_flesch_reading_ease(summary_text)}",
        "",
        "Jargon proxy metrics:",
        f"- unexplained_jargon_proxy_density_pct={jargon_density}",
        f"- technical_terms_sample={terms[:12]}",
        "- Jargon proxy is based on long/difficult-looking terms; final scoring must still judge whether terms "
        "are explained clearly in the summary text.",
        "",
        "Alignment and content coverage metrics:",
        f"- lexical_overlap_with_detailed_research_plan={alignment}",
        f"- lay_summary_coverage_hits={coverage_hits}",
        "- Alignment metric is lexical only; final scoring must compare the actual plain-English claims with "
        "the detailed proposal content.",
    ])


def _format_application_form_analysis(
    section_chunk_ids: dict[str, list[str]],
    pool_lookup: dict[str, dict[str, str]],
) -> str:
    section_rows: list[tuple[str, list[str], str]] = []
    for section_name, chunk_ids in section_chunk_ids.items():
        if section_name == APPLICATION_FORM_ANALYSIS_SECTION:
            continue
        text = "\n\n".join(pool_lookup[chunk_id]["text"] for chunk_id in chunk_ids)
        if text.strip():
            section_rows.append((section_name, chunk_ids, text))

    if not section_rows:
        return ""

    all_text = "\n\n".join(text for _, _, text in section_rows)
    non_budget_text = "\n\n".join(
        text
        for section_name, _, text in section_rows
        if "budget" not in section_name.lower()
    ) or all_text
    all_lines = _normalized_lines(all_text)
    non_budget_sentences = _sentence_tokens(non_budget_text)
    bullet_marker_count = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", all_text))
    numbered_heading_count = len(re.findall(r"(?m)^\s*\d+(?:\.\d+)*[.)]?\s+[A-Z][^\n]{3,120}$", all_text))
    table_like_line_count = len(re.findall(r"(?im)\b(year\s+1|year\s+2|year\s+3|total cost|total \(|£)\b", all_text))
    emphasis_marker_count = len(re.findall(r"(\*\*|__|<b>|</b>|\b[A-Z][A-Z /&-]{8,}\b)", all_text))
    transition_count = len(re.findall(
        r"\b(however|therefore|furthermore|moreover|in addition|to do this|for example|"
        r"as a result|this will|this project|aligns? with|building on|in phase|phase \d)\b",
        all_text,
        flags=re.IGNORECASE,
    ))
    objective_method_link_count = len(re.findall(
        r"\b(aims?|objectives?|research questions?|methods?|workstreams?|work packages?|"
        r"phase \d|project plan|data analysis|impact|dissemination|budget|justification)\b",
        all_text,
        flags=re.IGNORECASE,
    ))

    section_summary_lines = [
        f"- {section_name}: words={_word_count(text)}"
        for section_name, chunk_ids, text in section_rows
    ]

    overlap_rows: list[tuple[str, str, float]] = []
    for idx, (section_a, _, text_a) in enumerate(section_rows):
        for section_b, _, text_b in section_rows[idx + 1:]:
            score = _jaccard_similarity(text_a, text_b)
            if score >= 0.18:
                overlap_rows.append((section_a, section_b, score))
    overlap_rows = sorted(overlap_rows, key=lambda row: row[2], reverse=True)[:8]
    overlap_lines = [
        f"- {section_a} <-> {section_b}: lexical_overlap={score}"
        for section_a, section_b, score in overlap_rows
    ] or ["- No high cross-section lexical overlap detected at threshold 0.18."]

    return "\n".join([
        "Application form structural analysis derived from parser output.",
        "Use this single derived chunk as evidence for Application Form criteria af.*.",
        "",
        "Section coverage and hierarchy:",
        *section_summary_lines,
        "",
        "Formatting and structure indicators:",
        f"- parser_sections_detected={len(section_rows)}",
        f"- bullet_or_numbered_list_markers={bullet_marker_count}",
        f"- numbered_heading_like_lines={numbered_heading_count}",
        f"- table_like_budget_lines={table_like_line_count}",
        f"- extracted_emphasis_markers={emphasis_marker_count}",
        "- Parser limitation: bold/emphasis may be lost during text extraction; use extracted headings, "
        "section labels, list markers, and table structure as the available evidence.",
        "",
        "Duplication and repetition indicators:",
        f"- duplicate_sentence_rate_excluding_budget={_duplication_rate(non_budget_sentences)}",
        f"- repeated_line_rate={_duplication_rate(all_lines)}",
        *overlap_lines,
        "",
        "Logical flow and coherence indicators:",
        f"- transition_phrase_count={transition_count}",
        f"- objective_method_budget_link_terms={objective_method_link_count}",
        "- Section order moves from applicant/context, plain summary and abstract, research plan, PPI, "
        "training/support, and budget where those sections are present.",
    ])


def build_chunk_pool(application: dict[str, Any], max_chars: int = MAX_CHARS) -> dict[str, Any]:
    section_slug_map: dict[str, str] = {}
    used_slugs: dict[str, str] = {}
    section_counters: dict[str, int] = {}
    pool_lookup: dict[str, dict[str, str]] = {}
    section_chunk_ids: dict[str, list[str]] = {}

    def get_slug(section_name: str) -> str:
        base_slug = _slug_initials(section_name)
        existing_owner = used_slugs.get(base_slug)
        if existing_owner is None or existing_owner == section_name:
            used_slugs[base_slug] = section_name
            section_slug_map.setdefault(section_name, base_slug)
            return section_slug_map[section_name]

        suffix = 2
        while True:
            candidate = f"{base_slug}{suffix}"
            existing = used_slugs.get(candidate)
            if existing is None or existing == section_name:
                used_slugs[candidate] = section_name
                section_slug_map.setdefault(section_name, candidate)
                return section_slug_map[section_name]
            suffix += 1

    def _record(d: dict, *, is_table: bool, position) -> dict:
        return {
            "text": d["content_with_weight"],
            "parser_section": "",  # filled by caller
            "source_path": "",     # filled by caller
            "content_ltks": d.get("content_ltks", ""),
            "content_sm_ltks": d.get("content_sm_ltks", ""),
            "title_tks": d.get("title_tks", ""),
            "token_count": num_tokens_from_string(d["content_with_weight"]),
            "is_table": is_table,
            "position": position,
        }

    def add_leaf(parser_section: str, source_path: str, text: str, *, split: bool = True) -> None:
        if not text or not text.strip():
            return
        # Title/section context for BM25 title field + chunk doc template.
        doc_tpl = {
            "docnm_kwd": parser_section,
            "title_tks": rag_tokenizer.tokenize(parser_section),
        }

        records: list[dict] = []
        if split and _TABLE_RE.search(text):
            # Restore the table as its own chunk(s); HTML kept as the chunk text
            # (better for budget.py and for the scorer than flattened cells).
            for d in tokenize_table([((None, text), None)], dict(doc_tpl), True):
                records.append(_record(d, is_table=True, position=None))
        elif split:
            # Chunk the raw text (tags preserved), then per chunk strip + capture
            # its OWN @@..## tags → per-chunk page positions.
            for ck in _chunk_text(text):
                clean, positions = _extract_positions(ck)
                if not clean.strip():
                    continue
                d = dict(doc_tpl)
                tokenize(d, clean, True)  # → content_with_weight / content_ltks / content_sm_ltks
                records.append(_record(d, is_table=False, position=positions))
        else:
            # Derived single chunk (split=False): no chunking.
            clean, positions = _extract_positions(text)
            if clean.strip():
                d = dict(doc_tpl)
                tokenize(d, clean, True)
                records.append(_record(d, is_table=False, position=positions))

        if not records:
            return

        slug = get_slug(parser_section)
        section_counters[slug] = section_counters.get(slug, 0) + 1
        base_id = f"{slug}__{section_counters[slug]:03d}"
        ids = [base_id] if len(records) == 1 else [
            f"{base_id}_{chr(97 + idx)}" for idx in range(len(records))
        ]
        for chunk_id, rec in zip(ids, records):
            rec = dict(rec)
            rec["parser_section"] = parser_section
            rec["source_path"] = source_path
            pool_lookup[chunk_id] = rec
            section_chunk_ids.setdefault(parser_section, []).append(chunk_id)

    combined_context_entries: list[tuple[str, str]] = []
    for root_key, root_value in application.items():
        root_name = str(root_key)
        if root_name == APPLICATION_DETAILS_KEY and isinstance(root_value, dict):
            for child_key, child_value in root_value.items():
                parser_section = str(child_key)
                for leaf_text, source_path in _iter_leaves(
                    child_value,
                    [root_name, str(child_key)],
                    parser_section,
                ):
                    add_leaf(parser_section, source_path, leaf_text)
        elif root_name == SUMMARY_BUDGET_KEY:
            for leaf_text, source_path in _iter_leaves(root_value, [root_name], root_name):
                add_leaf(root_name, source_path, leaf_text)
        elif isinstance(root_value, dict):
            for child_key, child_value in root_value.items():
                child_name = str(child_key)
                if child_name == SUMMARY_BUDGET_KEY:
                    for leaf_text, source_path in _iter_leaves(
                        child_value,
                        [root_name, child_name],
                        child_name,
                    ):
                        add_leaf(child_name, source_path, leaf_text)
                    continue
                combined_context_entries.extend(
                    _iter_leaves(child_value, [root_name, child_name], APPLICATION_CONTEXT_SECTION)
                )
        else:
            combined_context_entries.extend(
                _iter_leaves(root_value, [root_name], APPLICATION_CONTEXT_SECTION)
            )

    combined_context = _format_combined_context(combined_context_entries)
    if combined_context:
        add_leaf(
            APPLICATION_CONTEXT_SECTION,
            APPLICATION_CONTEXT_SECTION,
            combined_context,
            split=False,
        )

    plain_english_analysis = _format_plain_english_analysis(section_chunk_ids, pool_lookup)
    if plain_english_analysis:
        add_leaf(
            PLAIN_ENGLISH_ANALYSIS_SECTION,
            PLAIN_ENGLISH_ANALYSIS_SECTION,
            plain_english_analysis,
            split=False,
        )

    application_form_analysis = _format_application_form_analysis(section_chunk_ids, pool_lookup)
    if application_form_analysis:
        add_leaf(
            APPLICATION_FORM_ANALYSIS_SECTION,
            APPLICATION_FORM_ANALYSIS_SECTION,
            application_form_analysis,
            split=False,
        )

    pool_index_lines = [
        f'{chunk_id}: {json.dumps(meta["text"], ensure_ascii=False)}'
        for chunk_id, meta in pool_lookup.items()
    ]

    return {
        "pool_lookup": pool_lookup,
        "pool_index_text": "\n".join(pool_index_lines),
        "section_chunk_ids": section_chunk_ids,
        "id_to_text": {chunk_id: meta["text"] for chunk_id, meta in pool_lookup.items()},
        "id_to_parser_section": {
            chunk_id: meta["parser_section"] for chunk_id, meta in pool_lookup.items()
        },
    }


def write_pool_artifacts(
    *,
    pool_lookup: dict[str, dict[str, str]],
    pool_index_text: str,
    artifacts_dir: str | Path,
    doc_id: str,
) -> dict[str, str]:
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    pool_json_path = artifacts_path / f"{doc_id}_pool.json"
    pool_index_path = artifacts_path / f"{doc_id}_pool_index.txt"
    pool_json_path.write_text(
        json.dumps(pool_lookup, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pool_index_path.write_text(pool_index_text, encoding="utf-8")
    return {
        "pool_json": str(pool_json_path),
        "pool_index": str(pool_index_path),
    }
