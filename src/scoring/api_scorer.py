"""DeepSeek (OpenAI-compatible) API scorer.

A drop-in alternative to the local Ollama ``_Scorer`` in ``qwen3_ollama.py``.
Implements the only interface ``score_application_base`` needs:

    generate_json(messages, *, schema, max_tokens) -> str   (JSON text)
    .model_name           (recorded in run_info)
    .last_response_body   (for failure artifacts)

DeepSeek's API is OpenAI-compatible, so this just POSTs to
``<base_url>/chat/completions`` with ``response_format={"type":"json_object"}``.
It does NOT enforce an arbitrary JSON schema (DeepSeek can't), but the scoring
prompts already embed the exact format + strict ID rules, and the pipeline's
JSON-parse retry + output normalisation filter anything off-spec — so json_object
mode is sufficient.
"""
from __future__ import annotations

import os
import re

import requests


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()


def _extract_json_object(text: str) -> str:
    s = (text or "").strip()
    a, b = s.find("{"), s.rfind("}")
    return s[a:b + 1] if a != -1 and b != -1 and b > a else s


class DeepSeekScorer:
    """OpenAI-compatible chat client for DeepSeek (or any compatible endpoint)."""

    def __init__(self, model_name: str | None = None, api_key: str | None = None,
                 base_url: str | None = None):
        self.model_name = model_name or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.timeout = float(os.environ.get("DEEPSEEK_TIMEOUT", "300"))
        # DeepSeek caps output tokens (~8K); the pipeline may request far more
        # (those defaults target Ollama). Clamp to keep the API from rejecting.
        self.max_output_tokens = int(os.environ.get("DEEPSEEK_MAX_OUTPUT_TOKENS", "8192"))
        self.last_response_body: dict | None = None
        if not self.api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Put it in .env (SCORER_BACKEND=deepseek) "
                "or export it before scoring."
            )
        print(f"[deepseek] using {self.base_url} model={self.model_name}", flush=True)

    def generate_json(self, messages: list[dict[str, str]], *, schema: dict, max_tokens: int) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": min(int(max_tokens), self.max_output_tokens),
            "stream": False,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"DeepSeek API request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text[:600]}")

        body = resp.json()
        self.last_response_body = body
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected DeepSeek response shape: {str(body)[:400]}") from exc
        return _extract_json_object(_strip_think_tags(content))
