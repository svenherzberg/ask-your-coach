from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Any, Optional, Protocol, List


@dataclass
class GenerationOptions:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.95
    stop: Optional[List[str]] = None


@dataclass
class TokenEvent:
    token: str
    is_first: bool = False
    is_last: bool = False
    latency_ms: Optional[float] = None
    meta: Dict[str, Any] | None = None


class LLMRunner(Protocol):
    """
    Abstrakte Schnittstelle für LLMs. Streaming API ist primär für niedrige Latenz.
    """

    async def generate_stream(self, prompt: str, options: GenerationOptions) -> AsyncIterator[TokenEvent]:
        """
        Streamt TokenEvents asynchron zurück.
        - prompt: kompletter Prompt (system + user assembled by orchestrator)
        - options: Generation options
        Yields TokenEvent(s) in token order (first token flagged).
        """
        ...

    async def generate(self, prompt: str, options: GenerationOptions) -> str:
        """
        Synchronous convenience wrapper returning the full generated text.
        """
        ...
