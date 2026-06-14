#
#  English drop-in replacement for the upstream (Chinese, huqie/datrie-based)
#  rag_tokenizer used by DeepDOC.
#
#  The grant applications processed by this project are English, so the heavy
#  61MB huqie dictionary + datrie trie are unnecessary. This shim mirrors the
#  exact public interface the copied DeepDOC code relies on
#  (``tokenize`` / ``fine_grained_tokenize`` / ``tag`` / ``is_chinese``) but is
#  backed by NLTK (already a project dependency), not hand-rolled regex rules.
#
#  Interface contract (matched to the upstream module-level API):
#    - tokenize(line: str)            -> str   space-joined, lowercased tokens
#    - fine_grained_tokenize(tks: str)-> str   (English no-op; upstream splits CJK sub-words)
#    - tag(token: str)                -> str   POS tag, lowercased (callers do ``.find("n")``)
#    - is_chinese(s: str)             -> bool  True if any CJK character present
#

import logging
import re

import nltk


def _ensure_nltk_data() -> None:
    """Best-effort download of the small NLTK resources we use."""
    wanted = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),               # NLTK >= 3.8.2
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),  # NLTK >= 3.9
    ]
    for path, pkg in wanted:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-_][A-Za-z0-9]+)*")


class RagTokenizer:
    """Lightweight English tokenizer mirroring the upstream interface."""

    def __init__(self):
        try:
            _ensure_nltk_data()
        except Exception:
            logging.warning("[rag_tokenizer] NLTK data preparation failed; "
                            "falling back to regex tokenization.")

    def tokenize(self, line):
        if not line:
            return ""
        line = str(line)
        try:
            from nltk.tokenize import word_tokenize
            toks = word_tokenize(line)
        except Exception:
            toks = _WORD_RE.findall(line)
        toks = [t.lower() for t in toks if t and not t.isspace()]
        return " ".join(toks)

    def fine_grained_tokenize(self, tks):
        # English has no sub-word segmentation step; return unchanged.
        return tks if isinstance(tks, str) else " ".join(tks or [])

    def tag(self, token):
        if not token:
            return ""
        try:
            from nltk import pos_tag
            tagged = pos_tag([str(token)])
            if tagged:
                return (tagged[0][1] or "").lower()
        except Exception:
            pass
        return ""

    def is_chinese(self, s):
        if not s:
            return False
        return bool(_CJK_RE.search(str(s)))

    # --- CJK normalizers used by query.FulltextQueryer; no-ops for English ---
    def tradi2simp(self, line):
        # traditional → simplified Chinese; irrelevant for English text
        return line if line is not None else ""

    def strQ2B(self, line):
        # full-width → half-width; convert if any full-width chars slip in
        if not line:
            return ""
        out = []
        for ch in str(line):
            code = ord(ch)
            if code == 0x3000:
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            out.append(chr(code))
        return "".join(out)

    # --- kept for interface compatibility (unused on the English path) ---
    def freq(self, tk):  # noqa: D401
        return 0

    def addUserDict(self, fnm):
        return None


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
is_chinese = tokenizer.is_chinese
freq = tokenizer.freq
addUserDict = tokenizer.addUserDict
tradi2simp = tokenizer.tradi2simp
strQ2B = tokenizer.strQ2B
