"""Local (offline) embedding + rerank backend for the vendored DeepDOC retriever.

Drop-in replacement for the upstream DashScope-based ``model.py``. Keeps the exact
public signatures that ``search_v2.Dealer`` (and the grant indexer) rely on:

  - ``generate_embedding(text | list[str]) -> list[float] | list[list[float]]``
  - ``rerank_similarity(query, texts) -> (np.ndarray scores, None)``

Backed by ``sentence-transformers`` (already a project dependency — see
src/feature_eng/coherence.py), so there is no API key and no network call at
inference time.
"""
import logging
import os
from typing import List

import numpy as np

# bge-small-en-v1.5 (384-d) is a strong small retrieval model; all-MiniLM-L6-v2
# (also 384-d, already cached by coherence.py) is a no-extra-download fallback.
_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
_RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_embedder = None
_reranker = None
_reranker_failed = False


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logging.info("[model] loading embedding model %s", _EMBED_MODEL_NAME)
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder


def embedding_dim() -> int:
    """Vector dimension of the active embedding model (e.g. 384)."""
    return int(_get_embedder().get_sentence_embedding_dimension())


def generate_embedding(text, api_key: str = None, base_url: str = None,
                       model_name: str = None, dimensions: int = None,
                       encoding_format: str = None, max_batch_size: int = 64):
    """Embed a string (→ 1-D vector) or a list of strings (→ list of vectors).

    Vectors are L2-normalised so cosine similarity reduces to a dot product.
    The extra keyword args exist only for signature-compatibility with the
    upstream DashScope version and are ignored.
    """
    model = _get_embedder()
    if isinstance(text, str):
        vec = model.encode([text], normalize_embeddings=True)[0]
        return [float(x) for x in vec]
    if isinstance(text, list):
        if not text:
            return []
        vecs = model.encode(list(text), batch_size=max_batch_size,
                            normalize_embeddings=True)
        return [[float(x) for x in v] for v in vecs]
    return None


def rerank_similarity(query, texts):
    """Cross-encoder rerank. Returns ``(scores aligned with texts, None)``.

    Scores are sigmoid-squashed to (0, 1) to match the scale the upstream
    gte-rerank returned (so Dealer's weighted blend stays balanced). Falls back
    to all-zeros if the cross-encoder cannot be loaded — Dealer then ranks on
    token + vector similarity only.
    """
    global _reranker, _reranker_failed
    texts = list(texts)
    if not texts:
        return np.zeros(0, dtype=float), None

    if _reranker is None and not _reranker_failed:
        try:
            from sentence_transformers import CrossEncoder
            logging.info("[rerank] loading cross-encoder %s", _RERANK_MODEL_NAME)
            _reranker = CrossEncoder(_RERANK_MODEL_NAME)
        except Exception as e:  # noqa: BLE001
            logging.warning("[rerank] CrossEncoder unavailable (%s); "
                            "falling back to zero rerank scores.", e)
            _reranker_failed = True

    if _reranker is None:
        return np.zeros(len(texts), dtype=float), None

    try:
        raw = np.asarray(_reranker.predict([(query, t) for t in texts]), dtype=float)
        scores = 1.0 / (1.0 + np.exp(-raw))  # sigmoid → (0, 1)
        return scores, None
    except Exception as e:  # noqa: BLE001
        logging.warning("[rerank] predict failed (%s); zero scores.", e)
        return np.zeros(len(texts), dtype=float), None
