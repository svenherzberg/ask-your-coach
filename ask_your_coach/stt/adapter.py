from __future__ import annotations
import asyncio
import json
import time
from typing import AsyncIterator, Callable, Dict, Any, List, Optional


class ExternalProcessSTTAdapter:
    """Adapter that runs an external STT process and yields normalized transcript items.

    The external process is expected to write line-delimited JSON or plain text to stdout.
    Each yielded item is a dict: {"text": str, "is_final": bool, "timestamp": float}

    Example usage:
        adapter = ExternalProcessSTTAdapter(["/usr/local/bin/my_stt", "--stream-json"]) 
        async for item in adapter.stream():
            print(item)
    """

    def __init__(self, cmd: List[str], encoding: str = "utf-8") -> None:
        self.cmd = cmd
        self.encoding = encoding

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        proc = await asyncio.create_subprocess_exec(
            *self.cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            assert proc.stdout is not None
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    s = line.decode(self.encoding).strip()
                except Exception:
                    s = line.decode(errors="ignore").strip()

                item: Dict[str, Any]
                # try json first
                try:
                    obj = json.loads(s)
                    text = obj.get("text") or obj.get("transcript") or obj.get("result") or s
                    is_final = bool(obj.get("is_final", obj.get("final", True)))
                    item = {"text": text, "is_final": is_final, "timestamp": time.time()}
                except Exception:
                    # fallback: plain text lines are treated as final segments
                    if not s:
                        continue
                    item = {"text": s, "is_final": True, "timestamp": time.time()}

                yield item

            # wait for process to exit
            await proc.wait()
        finally:
            # try to terminate if still running
            if proc.returncode is None:
                try:
                    proc.kill()
                except Exception:
                    pass


async def normalize_async_iterator(it: AsyncIterator[Any]) -> AsyncIterator[Dict[str, Any]]:
    """Normalize various STT iterator item shapes into the expected dict shape.

    Accepts mapping-like objects, objects with `.text` attribute, or plain strings.
    """
    async for item in it:
        text = None
        is_final = False
        ts = time.time()

        if item is None:
            continue

        # mapping-like
        if hasattr(item, "get"):
            try:
                text = item.get("text") or item.get("transcript") or item.get("result")
                is_final = bool(item.get("is_final", item.get("final", False)))
            except Exception:
                pass

        # object with attribute
        if text is None and hasattr(item, "text"):
            try:
                text = getattr(item, "text")
                is_final = bool(getattr(item, "is_final", False) or getattr(item, "final", False))
            except Exception:
                pass

        # fallback to string
        if text is None:
            try:
                text = str(item)
                is_final = True
            except Exception:
                continue

        if not text:
            continue

        yield {"text": text, "is_final": bool(is_final), "timestamp": ts}
