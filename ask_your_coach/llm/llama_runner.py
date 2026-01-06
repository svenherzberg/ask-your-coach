from __future__ import annotations
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

from .interfaces import LLMRunner, GenerationOptions, TokenEvent


class LlamaCPPRunner(LLMRunner):
    """
    Adapter für llama.cpp / llama-cpp-python.
    Bietet eine blocking->async Brücke via ThreadPoolExecutor und asyncio.Queue.
    """

    def __init__(self, model_path: str, n_threads: int = 8, n_ctx: int = 2048, n_gpu_layers: int = 0, **kwargs):
        self.model_path = model_path
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._client = None  # lazy init of llama-cpp client
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._kwargs = kwargs

    def _init_client(self):
        if self._client is None:
            try:
                from llama_cpp import Llama

                self._client = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, n_gpu_layers=self.n_gpu_layers, **self._kwargs)
            except Exception as e:
                raise RuntimeError(
                    "Failed to initialize llama-cpp client. Ensure 'llama-cpp-python' is installed and model path is valid."
                ) from e

    async def generate_stream(self, prompt: str, options: GenerationOptions) -> AsyncIterator[TokenEvent]:
        """
        Async wrapper: ruft das blocking streaming API an und yieldet TokenEvent.
        Implementierungsdetails:
        - Startet in einem Thread die blocking create(stream=True) Aufrufe
        - Übergibt erhaltene Token per asyncio loop.call_soon_threadsafe -> queue.put_nowait
        - Yield tokens asynchron aus der Queue
        """
        self._init_client()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        start_ts = time.time()

        def _blocking_stream():
            try:
                # llama-cpp-python: client.create(..., stream=True) is an iterator/generator
                stream = self._client.create(prompt=prompt, max_tokens=options.max_tokens, temperature=options.temperature, top_p=options.top_p, stream=True)
                for part in stream:
                    token_text = None
                    # try several common payload shapes
                    if isinstance(part, dict):
                        # openai-like streaming chunks
                        try:
                            choices = part.get("choices")
                            if choices and isinstance(choices, list):
                                d = choices[0]
                                token_text = d.get("delta", {}).get("content") or d.get("text")
                        except Exception:
                            token_text = str(part)
                    else:
                        token_text = str(part)

                    if token_text:
                        loop.call_soon_threadsafe(queue.put_nowait, (token_text, time.time()))
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, ("__ERROR__:" + str(e), time.time()))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, (None, time.time()))

        # Start blocking streaming in executor
        loop.run_in_executor(self._executor, _blocking_stream)

        seen_first = False
        while True:
            token_item, ts = await queue.get()
            if token_item is None:
                break
            if token_item.startswith("__ERROR__:"):
                raise RuntimeError(token_item[len("__ERROR__:"):])

            latency_ms = (ts - start_ts) * 1000.0
            ev = TokenEvent(token=token_item, is_first=not seen_first, latency_ms=latency_ms, meta={})
            seen_first = True
            yield ev

    async def generate(self, prompt: str, options: GenerationOptions) -> str:
        """
        Sync wrapper using run_in_executor to call blocking create without streaming.
        """
        self._init_client()
        loop = asyncio.get_running_loop()

        def _blocking_complete():
            out = self._client.create(prompt=prompt, max_tokens=options.max_tokens, temperature=options.temperature, top_p=options.top_p, stream=False)
            # response shape may vary; try to normalize
            if isinstance(out, dict):
                try:
                    choices = out.get("choices")
                    if choices and isinstance(choices, list):
                        return choices[0].get("text") or choices[0].get("message", {}).get("content") or str(out)
                except Exception:
                    return str(out)
            return str(out)

        result = await loop.run_in_executor(self._executor, _blocking_complete)
        return result
