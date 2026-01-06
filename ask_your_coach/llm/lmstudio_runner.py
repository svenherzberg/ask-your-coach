from __future__ import annotations
import asyncio
import time
import json
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

from .interfaces import LLMRunner, GenerationOptions, TokenEvent


class LMStudioRunner(LLMRunner):
    """Adapter for LMStudio local endpoint (OpenAI-compatible HTTP streaming).

    Expects environment or init params: base_url (e.g. http://localhost:1234), model id.
    """

    def __init__(self, base_url: str, model: str, api_key: Optional[str] = None, **kwargs):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._kwargs = kwargs

    async def generate_stream(self, prompt: str, options: GenerationOptions) -> AsyncIterator[TokenEvent]:
        import requests

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        start_ts = time.time()

        def _blocking_stream():
            url = f"{self.base_url}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": options.max_tokens,
                "temperature": options.temperature,
                "top_p": options.top_p,
                "stream": True,
            }

            try:
                with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        line = raw.strip()
                        # some servers prefix 'data: '
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()
                        if line == "[DONE]":
                            break
                        try:
                            obj = json.loads(line)
                        except Exception:
                            # fallback: push raw
                            loop.call_soon_threadsafe(queue.put_nowait, (line, time.time()))
                            continue

                        # extract token text in various payload shapes
                        token_text = None
                        try:
                            choices = obj.get("choices")
                            if choices and isinstance(choices, list):
                                d = choices[0]
                                if "delta" in d:
                                    token_text = d["delta"].get("content")
                                else:
                                    token_text = d.get("text") or (d.get("message") or {}).get("content")
                        except Exception:
                            token_text = None

                        if token_text:
                            loop.call_soon_threadsafe(queue.put_nowait, (token_text, time.time()))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("__ERROR__:" + str(e), time.time()))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, (None, time.time()))

        loop.run_in_executor(self._executor, _blocking_stream)

        seen_first = False
        while True:
            token_item, ts = await queue.get()
            if token_item is None:
                break
            if isinstance(token_item, str) and token_item.startswith("__ERROR__:"):
                raise RuntimeError(token_item[len("__ERROR__:"):])

            latency_ms = (ts - start_ts) * 1000.0
            ev = TokenEvent(token=token_item, is_first=not seen_first, latency_ms=latency_ms, meta={})
            seen_first = True
            yield ev

    async def generate(self, prompt: str, options: GenerationOptions) -> str:
        import requests

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "top_p": options.top_p,
            "stream": False,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        out = resp.json()
        try:
            choices = out.get("choices")
            if choices and isinstance(choices, list):
                return choices[0].get("text") or (choices[0].get("message") or {}).get("content") or str(out)
        except Exception:
            return str(out)
