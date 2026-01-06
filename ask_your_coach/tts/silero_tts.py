from __future__ import annotations
import asyncio
import io
import math
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

from .interfaces import TTSVoice, SynthesisOptions, PCMChunk


class SileroTTS(TTSVoice):
    """Silero TTS adapter with lazy model loading and PCM conversion.

    This implementation attempts to load Silero via `torch.hub`. If `torch` or the
    hub model isn't available, it will raise a clear error. The adaptor converts
    model waveforms (torch tensors or numpy arrays) into 16-bit PCM bytes and
    yields `PCMChunk` objects in streaming fashion.
    """

    def __init__(self, language: str = "de", speaker: str | None = None, sample_rate: int = 22050):
        self.language = language
        self.speaker = speaker
        self.sample_rate = sample_rate
        self._model = None
        self._utils = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _init_model(self):
        if self._model is not None:
            return
        try:
            import torch
            import numpy as _np

            # Try loading silero tts from torch hub. API shape may vary across versions,
            # so we attempt a few common signatures. If these fail, surface an instructive error.
            try:
                res = torch.hub.load("snakers4/silero-models", "silero_tts", language=self.language)
                # often returns (model, utils) or model alone
                if isinstance(res, tuple) and len(res) >= 1:
                    self._model = res[0]
                    self._utils = res[1] if len(res) > 1 else None
                else:
                    self._model = res
                    self._utils = None
            except Exception:
                # some hub variants accept speaker param
                try:
                    res = torch.hub.load(
                        "snakers4/silero-models", "silero_tts", language=self.language, speaker=self.speaker
                    )
                    if isinstance(res, tuple) and len(res) >= 1:
                        self._model = res[0]
                        self._utils = res[1] if len(res) > 1 else None
                    else:
                        self._model = res
                        self._utils = None
                except Exception as e:
                    raise RuntimeError(
                        "Failed to load Silero TTS via torch.hub; ensure torch and silero hub models are available"
                    ) from e

            # determine sample rate if available in utils
            if self._utils and isinstance(self._utils, dict):
                self.sample_rate = int(self._utils.get("sample_rate", self.sample_rate))

        except Exception as e:
            raise RuntimeError("Silero TTS initialization failed; install torch and silero model assets") from e

    def _waveform_to_pcm_bytes(self, waveform) -> bytes:
        """Convert a waveform (torch.Tensor or numpy array) in float32 [-1,1] to int16 PCM bytes."""
        try:
            import numpy as _np
            import torch as _torch

            if hasattr(waveform, "numpy") and not isinstance(waveform, _np.ndarray):
                arr = waveform.numpy()
            else:
                arr = _np.array(waveform)
        except Exception:
            # fallback: if it's already bytes, return
            if isinstance(waveform, (bytes, bytearray)):
                return bytes(waveform)
            raise

        # Ensure mono
        if arr.ndim > 1:
            arr = arr.mean(axis=0)

        # normalize float -> int16
        if arr.dtype.kind == "f":
            arr = (arr * 32767.0).astype("int16")
        elif arr.dtype.kind in ("i", "u"):
            arr = arr.astype("int16")

        return arr.tobytes()

    async def synthesize_stream(self, text: str, options: SynthesisOptions) -> AsyncIterator[PCMChunk]:
        """Synthesize `text` and yield PCMChunk objects.

        The heavy model work runs in a thread. The produced waveform is split into
        small buffers and streamed as `PCMChunk` objects.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        self._init_model()

        def _blocking_synthesize():
            try:
                # Attempt several common call patterns against the loaded model:
                # 1) model.apply_tts(texts=[...], speakers=[...], sample_rate=...)
                # 2) model.tts(text)
                # 3) model(text)
                waveform = None
                try:
                    # preferred API
                    if hasattr(self._model, "apply_tts"):
                        out = self._model.apply_tts(texts=[text], speakers=[self.speaker] if self.speaker else None, sample_rate=self.sample_rate)
                        # apply_tts often returns list of numpy arrays
                        if isinstance(out, (list, tuple)) and len(out) > 0:
                            waveform = out[0]
                        else:
                            waveform = out
                    elif hasattr(self._model, "tts"):
                        waveform = self._model.tts(text)
                    elif callable(self._model):
                        waveform = self._model(text)
                except Exception:
                    waveform = None

                if waveform is None:
                    # fallback: try utils-based wrapper if provided
                    if self._utils and hasattr(self._utils, "save_wav"):
                        buf = io.BytesIO()
                        try:
                            # some utils expose helper to produce wav
                            self._utils.save_wav(text, buf)
                            pcm = buf.getvalue()
                            loop.call_soon_threadsafe(queue.put_nowait, (0, pcm))
                            return
                        except Exception:
                            pass

                    # as last resort, produce short silence
                    silence = b"\x00" * 4096
                    loop.call_soon_threadsafe(queue.put_nowait, (0, silence))
                    return

                pcm_bytes = self._waveform_to_pcm_bytes(waveform)

                # chunk the PCM bytes into 32KB frames
                chunk_size = 32 * 1024
                total = len(pcm_bytes)
                seq = 0
                for i in range(0, total, chunk_size):
                    chunk = pcm_bytes[i : i + chunk_size]
                    loop.call_soon_threadsafe(queue.put_nowait, (seq, chunk))
                    seq += 1
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, (None, None))

        loop.run_in_executor(self._executor, _blocking_synthesize)

        while True:
            seq, pcm = await queue.get()
            if seq is None:
                break
            yield PCMChunk(pcm_bytes=pcm, sample_rate=options.sample_rate or self.sample_rate, seq=seq)

    async def synthesize_to_file(self, text: str, out_path: str, options: SynthesisOptions) -> None:
        # collect full PCM and write a WAV file with correct headers
        chunks = []
        async for c in self.synthesize_stream(text, options):
            chunks.append(c.pcm_bytes)

        pcm = b"".join(chunks)

        # write WAV with 16-bit mono
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(options.sample_rate or self.sample_rate)
            wf.writeframes(pcm)

