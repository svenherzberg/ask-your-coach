from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Optional

from ..llm.interfaces import LLMRunner, GenerationOptions, TokenEvent
from ..prompts.manager import PromptManager


@dataclass
class OrchestratorConfig:
    stt_queue_maxsize: int = 8
    turn_final_timeout_s: float = 1.0
    llm_generation_options: GenerationOptions = GenerationOptions()
    # heuristics
    tts_start_on_first_sentence: bool = True


class Orchestrator:
    """
    Orchestrator skeleton that wires STT -> LLM -> TTS.

    Expected STT input: an AsyncIterator yielding objects with fields
      - text: str
      - is_final: bool
      - timestamp: float (optional)

    LLMRunner must implement the streaming contract from `llm.interfaces`.
    TTS callback is an async function accepting text chunks (or full text) and playing/saving audio.
    """

    def __init__(
        self,
        stt_stream: AsyncIterator[Any],
        llm_runner: LLMRunner,
        tts_callback: Callable[[str], Awaitable[None]],
        config: Optional[OrchestratorConfig] = None,
        mode: str = "default",
        user_profile: Optional[dict] = None,
    ) -> None:
        self.stt_stream = stt_stream
        self.llm = llm_runner
        self.tts_callback = tts_callback
        self.config = config or OrchestratorConfig()
        self.mode = mode
        self.user_profile = user_profile or {}
        self.prompt_manager = PromptManager()
        self._prompt_watch_task: Optional[asyncio.Task] = None

        self._stt_queue: asyncio.Queue = asyncio.Queue(maxsize=self.config.stt_queue_maxsize)
        self._stop_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the orchestrator: consume STT and process turns."""
        self._worker_task = asyncio.create_task(self._run())
        # start prompt manager watcher (hot-reload) in background
        try:
            self._prompt_watch_task = asyncio.create_task(
                self.prompt_manager.watch_async(interval=1.0, on_change=lambda: print("prompts reloaded"))
            )
        except Exception:
            self._prompt_watch_task = None

    async def stop(self) -> None:
        """Signal stop and wait for worker termination."""
        self._stop_event.set()
        if self._worker_task:
            await self._worker_task
        # cancel prompt watcher if running
        if getattr(self, "_prompt_watch_task", None):
            self._prompt_watch_task.cancel()
            try:
                await self._prompt_watch_task
            except Exception:
                pass

    async def _run(self) -> None:
        """Main run loop: spawn STT consumer and turn processor."""
        stt_consumer = asyncio.create_task(self._consume_stt())
        turn_processor = asyncio.create_task(self._process_turns())

        done, pending = await asyncio.wait(
            [stt_consumer, turn_processor, self._stop_event.wait()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for p in pending:
            p.cancel()

    async def _consume_stt(self) -> None:
        """Consumes the STT async iterator and pushes transcripts into the internal queue."""
        async for item in self.stt_stream:
            # Expect item to be mapping-like; normalize
            try:
                text = item.get("text") if hasattr(item, "get") else getattr(item, "text", None)
                is_final = item.get("is_final") if hasattr(item, "get") else getattr(item, "is_final", False)
            except Exception:
                # fallback: try to stringify
                text = str(item)
                is_final = True

            if not text:
                continue

            # Put transcript into queue; drop if full to preserve latency
            try:
                self._stt_queue.put_nowait({"text": text, "is_final": bool(is_final)})
            except asyncio.QueueFull:
                # drop oldest to make room
                try:
                    _ = self._stt_queue.get_nowait()
                except Exception:
                    pass
                try:
                    self._stt_queue.put_nowait({"text": text, "is_final": bool(is_final)})
                except Exception:
                    # if still can't enqueue, skip
                    continue

            if self._stop_event.is_set():
                break

    async def _process_turns(self) -> None:
        """Simple turn processor: groups incoming transcripts and triggers LLM generation on final turns."""
        buffer = ""
        while not self._stop_event.is_set():
            try:
                item = await asyncio.wait_for(self._stt_queue.get(), timeout=self.config.turn_final_timeout_s)
            except asyncio.TimeoutError:
                # no new STT data; if buffer non-empty, consider it final by timeout
                if buffer.strip():
                    await self._handle_final_turn(buffer)
                    buffer = ""
                continue

            text = item.get("text", "")
            is_final = item.get("is_final", False)

            buffer = (buffer + " " + text).strip()

            if is_final:
                await self._handle_final_turn(buffer)
                buffer = ""

    async def _handle_final_turn(self, full_text: str) -> None:
        """Called when a user turn is considered final. Assembles prompt, calls LLM, handles token stream and TTS triggers."""
        if not full_text.strip():
            return

        # assemble prompt (this could include system instructions, mode, context)
        prompt = self._assemble_prompt(full_text)

        # collect generation options from PromptManager if available
        gen_opts = self.prompt_manager.get_generation_options(self.mode)
        # merge gen_opts into current config.llm_generation_options
        try:
            # convert dict into GenerationOptions; preserve existing defaults when keys missing
            merged = GenerationOptions(
                max_tokens=gen_opts.get("max_tokens", self.config.llm_generation_options.max_tokens),
                temperature=gen_opts.get("temperature", self.config.llm_generation_options.temperature),
                top_p=gen_opts.get("top_p", self.config.llm_generation_options.top_p),
                stop=getattr(self.config.llm_generation_options, "stop", None),
            )
        except Exception:
            merged = self.config.llm_generation_options

        # call LLM streaming API
        async for token_event in self.llm.generate_stream(prompt, merged):
            # simple policy: send tokens to TTS after first sentence or when generation finishes
            # TokenEvent.token may be partial; accumulate or buffer as needed
            if self.config.tts_start_on_first_sentence:
                # crude heuristic: if token contains sentence terminator, send the chunk
                if any(p in token_event.token for p in (".", "?", "!")):
                    await self.tts_callback(token_event.token)

        # optionally, at generation end, ensure any remaining buffered text is spoken
        # this sample implementation doesn't accumulate full text; orchestration layer can be extended

    def _assemble_prompt(self, user_turn: str) -> str:
        # Render system prompt from template files in prompts/ via PromptManager
        user_name = self.user_profile.get("name", "User")
        system = self.prompt_manager.render(self.mode, mode=self.mode, user_name=user_name)
        return f"{system}\nUser: {user_turn}\nAssistant:"
