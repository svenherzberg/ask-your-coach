import asyncio
import tempfile

import numpy as np
import pytest

from ask_your_coach.stt.adapter import normalize_async_iterator
from ask_your_coach.tts.silero_tts import SileroTTS
from ask_your_coach.tts.interfaces import SynthesisOptions


@pytest.mark.asyncio
async def test_normalize_async_iterator():
    class Obj:
        def __init__(self, text, is_final):
            self.text = text
            self.is_final = is_final

    async def gen():
        yield {"text": "one", "is_final": False}
        yield Obj("two", True)
        yield "three"

    out = []
    async for it in normalize_async_iterator(gen()):
        out.append(it)

    assert out[0]["text"] == "one" and out[0]["is_final"] is False
    assert out[1]["text"] == "two" and out[1]["is_final"] is True
    assert out[2]["text"] == "three" and out[2]["is_final"] is True


@pytest.mark.asyncio
async def test_silero_tts_stream_and_file(tmp_path, monkeypatch):
    # small synthetic waveform (10ms of a 440Hz sine)
    sr = 22050
    t = np.linspace(0, 0.01, int(sr * 0.01), endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    class DummyModel:
        def apply_tts(self, texts, speakers=None, sample_rate=None):
            return [waveform]

    tts = SileroTTS(sample_rate=sr)
    # avoid importing torch in the test by stubbing init
    monkeypatch.setattr(tts, "_init_model", lambda: None)
    tts._model = DummyModel()

    opts = SynthesisOptions(sample_rate=sr)
    chunks = []
    async for c in tts.synthesize_stream("hello", opts):
        chunks.append(c)

    assert len(chunks) > 0
    assert all(isinstance(c.pcm_bytes, (bytes, bytearray)) for c in chunks)

    out = tmp_path / "out.wav"
    await tts.synthesize_to_file("hello", str(out), opts)
    assert out.exists() and out.stat().st_size > 0
