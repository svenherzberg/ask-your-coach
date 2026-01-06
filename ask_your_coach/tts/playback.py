from __future__ import annotations
import asyncio
import tempfile
import wave
import os
import shutil
import subprocess
from typing import Optional

from .interfaces import TTSVoice, SynthesisOptions, PCMChunk


def _find_system_player() -> Optional[list[str]]:
    # macOS
    if shutil.which("afplay"):
        return ["afplay"]
    # Linux
    if shutil.which("aplay"):
        return ["aplay"]
    # fallback: no system player found
    return None


async def _play_wav(path: str) -> None:
    cmd_base = _find_system_player()
    if cmd_base is None:
        # No system player available; try to use python sounddevice if installed
        try:
            import sounddevice as sd
            import soundfile as sf

            data, samplerate = sf.read(path, dtype="int16")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: sd.play(data, samplerate))
            # wait until playback finishes
            await loop.run_in_executor(None, sd.wait)
            return
        except Exception:
            # Give up gracefully
            return

    cmd = cmd_base + [path]
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: subprocess.run(cmd, check=False))


async def tts_playback_from_text(text: str, tts_voice: TTSVoice, options: SynthesisOptions) -> None:
    """
    Synthesizes `text` using `tts_voice` and plays the resulting audio.

    Implementation notes:
    - Collects PCMChunks from `synthesize_stream` and writes a temporary WAV file.
    - Uses `afplay` on macOS or `aplay` on Linux when available; otherwise tries `sounddevice`.
    - This is a best-effort playback helper for local testing and demo purposes.
    """
    chunks: list[bytes] = []
    sample_rate = options.sample_rate

    async for c in tts_voice.synthesize_stream(text, options):
        # collect raw PCM bytes
        chunks.append(c.pcm_bytes)
        # prefer sample rate from chunk if present
        if getattr(c, "sample_rate", None):
            sample_rate = c.sample_rate

    if not chunks:
        return

    # Write temporary WAV file (assumes 16-bit mono PCM in this skeleton)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name

    try:
        with wave.open(wav_path, "wb") as wf:
            nchannels = 1
            sampwidth = 2  # bytes per sample (16-bit)
            wf.setnchannels(nchannels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(chunks))

        await _play_wav(wav_path)
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


def tts_callback_wrapper(tts_voice: TTSVoice, options: SynthesisOptions):
    """Returns an async callable suitable as Orchestrator `tts_callback`.

    Usage: `orchestrator = Orchestrator(..., tts_callback=tts_callback_wrapper(tts, opts))`
    """

    async def _cb(text: str) -> None:
        await tts_playback_from_text(text, tts_voice, options)

    return _cb
