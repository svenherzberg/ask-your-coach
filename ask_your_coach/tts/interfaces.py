from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Any, Optional, Protocol


@dataclass
class SynthesisOptions:
    sample_rate: int = 22050
    voice: Optional[str] = None
    # optional quality/latency knobs
    speed: float = 1.0


@dataclass
class PCMChunk:
    pcm_bytes: bytes
    sample_rate: int
    seq: int = 0
    meta: Dict[str, Any] | None = None


class TTSVoice(Protocol):
    """Abstrakte Schnittstelle für TTS Voice implementations."""

    async def synthesize_stream(self, text: str, options: SynthesisOptions) -> AsyncIterator[PCMChunk]:
        """Erzeugt PCMChunks asynchron (Streaming vocoder path)."""
        ...

    async def synthesize_to_file(self, text: str, out_path: str, options: SynthesisOptions) -> None:
        """Synthetisiert synchron und speichert in `out_path`."""
        ...
