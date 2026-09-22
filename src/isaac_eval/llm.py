from __future__ import annotations
import os
import time
from google import genai
from google.genai import types
from google.genai import errors as genai_errors


class LLMClient:
    def __init__(self, cfg: dict):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Put it in isaac-eval/.env as GEMINI_API_KEY=... "
                "and call config.load_env() before constructing LLMClient."
            )
        self._client = genai.Client(api_key=api_key)
        self._cfg = cfg["llm"]
        # C-2 비용/지연 계측용 누적기 (비침습적, 기존 동작 불변)
        self.usage = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0, "latency_s": 0.0}

    def generate(self, system: str, user: str) -> str:
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                _t0 = time.monotonic()
                response = self._client.models.generate_content(
                    model=self._cfg["model"],
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        temperature=self._cfg["temperature"],
                        top_p=self._cfg["top_p"],
                        max_output_tokens=self._cfg["max_tokens"],
                    ),
                )
                self._record_usage(response, time.monotonic() - _t0)
                return response.text
            except genai_errors.ServerError as e:  # 503/500 과부하 — 백오프 재시도
                if attempt == max_retries:
                    raise
                time.sleep(min(8.0 * attempt, 40.0))
            except genai_errors.ClientError as e:
                if e.code != 429 or attempt == max_retries or _is_daily_quota_exhausted(e):
                    # Daily-quota exhaustion won't resolve within this process's
                    # lifetime, so fail fast instead of sleeping through retries.
                    raise
                delay = _extract_retry_delay_seconds(e) or 15.0
                time.sleep(delay + 1.0)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _record_usage(self, response, latency_s: float) -> None:
        self.usage["calls"] += 1
        self.usage["latency_s"] += latency_s
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            pt = getattr(um, "prompt_token_count", 0) or 0
            ot = getattr(um, "candidates_token_count", 0) or 0
            tt = getattr(um, "total_token_count", 0) or (pt + ot)
            self.usage["prompt_tokens"] += pt
            self.usage["output_tokens"] += ot
            self.usage["total_tokens"] += tt


def _is_daily_quota_exhausted(e: "genai_errors.ClientError") -> bool:
    details = (getattr(e, "details", None) or {}).get("error", {}).get("details", [])
    for d in details:
        for violation in d.get("violations", []):
            if "PerDay" in violation.get("quotaId", ""):
                return True
    return False


def _extract_retry_delay_seconds(e: "genai_errors.ClientError") -> float | None:
    details = (getattr(e, "details", None) or {}).get("error", {}).get("details", [])
    for d in details:
        if d.get("@type", "").endswith("RetryInfo"):
            raw = d.get("retryDelay", "")
            if raw.endswith("s"):
                try:
                    return float(raw[:-1])
                except ValueError:
                    return None
    return None
