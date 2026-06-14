"""
Qwen3 Ollama grant scorer.

Section scoring runs on configurable Ollama models with rule-based retrieval.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests

from src.scoring.pipeline import score_application_base

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:27b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "1200"))
OLLAMA_NUM_CTX = int(os.environ["OLLAMA_NUM_CTX"]) if "OLLAMA_NUM_CTX" in os.environ else None


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()


def _extract_json_object(text: str) -> str:
    clean = (text or "").strip()
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return clean[first_brace:last_brace + 1]
    return clean


def _extract_message_content(body: dict[str, Any]) -> str:
    message = body.get("message") or {}
    content = _strip_think_tags((message.get("content") or ""))
    return _extract_json_object(content)


class _Scorer:
    def __init__(self, model_name: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host.rstrip("/")
        self.last_response_body: dict[str, Any] | None = None
        print(f"[qwen3_ollama] using {self.host} model={self.model_name}", flush=True)

    def generate_json(self, messages: list[dict[str, str]], *, schema: dict[str, Any], max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": max_tokens,
                **({"num_ctx": OLLAMA_NUM_CTX} if OLLAMA_NUM_CTX else {}),
            },
            "think": False,
        }
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.host}. Start the Ollama server or set OLLAMA_HOST to the correct "
                "endpoint before running scoring."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                "Timed out waiting for Ollama at "
                f"{self.host}. Increase OLLAMA_TIMEOUT or check whether the model server is healthy."
            ) from exc
        response.raise_for_status()
        body = response.json()
        self.last_response_body = body
        return _extract_message_content(body)


def score_application(
    application: dict[str, Any],
    criteria_path: str | Path,
    *,
    doc_id: str | None = None,
    scorer: _Scorer | None = None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    scorer_client = scorer or _Scorer(model_name=OLLAMA_MODEL, host=OLLAMA_HOST)
    # Retrieval config (env-driven, opt-in for the ES few-shot corpus):
    #   GRANT_USE_RETRIEVAL=0  → disable hybrid evidence retrieval (full-text)
    #   GRANT_CORPUS_INDEX=grant_corpus → enable few-shot from the ES corpus
    use_retrieval = os.environ.get("GRANT_USE_RETRIEVAL", "1") != "0"
    corpus_index = os.environ.get("GRANT_CORPUS_INDEX") or None
    return score_application_base(
        application=application,
        criteria_path=criteria_path,
        doc_id=doc_id,
        scorer_client=scorer_client,
        artifacts_dir=artifacts_dir,
        use_retrieval=use_retrieval,
        corpus_index=corpus_index,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("application_json")
    parser.add_argument("--criteria", default=str(Path(__file__).parent / "criteria_points.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    in_path = Path(args.application_json)
    out_path = Path(args.out) if args.out else in_path.with_name(in_path.stem + "_scored.json")
    application = json.loads(in_path.read_text(encoding="utf-8"))
    result = score_application(
        application,
        args.criteria,
        doc_id=in_path.stem,
        artifacts_dir=out_path.parent,
    )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[qwen3_ollama] wrote {out_path}")
    sys.exit(0)
