import asyncio
import pytest

from ask_your_coach.orchestrator.orchestrator import Orchestrator, OrchestratorConfig
from ask_your_coach.llm.interfaces import TokenEvent, GenerationOptions


async def _stt_stream_partial_then_final():
    # simulate streaming partial followed by final
    yield {"text": "Hello", "is_final": False}
    await asyncio.sleep(0.01)
    yield {"text": " world", "is_final": True}


async def _stt_stream_partial_only():
    # simulate a partial that never signals final (to trigger timeout)
    yield {"text": "Partially said", "is_final": False}
    # stream ends


class MockLLMRunner:
    def __init__(self, tokens):
        self.tokens = tokens

    async def generate_stream(self, prompt: str, options: GenerationOptions):
        for t in self.tokens:
            await asyncio.sleep(0.01)
            yield TokenEvent(token=t)

    async def generate(self, prompt: str, options: GenerationOptions) -> str:
        return "".join(self.tokens)


@pytest.mark.asyncio
async def test_orchestrator_triggers_tts_on_sentence_end():
    stt = _stt_stream_partial_then_final()
    llm = MockLLMRunner(["This is an answer."])
    collected = []

    async def tts_cb(text: str):
        collected.append(text)

    cfg = OrchestratorConfig(turn_final_timeout_s=0.1)
    orch = Orchestrator(stt_stream=stt, llm_runner=llm, tts_callback=tts_cb, config=cfg)
    await orch.start()

    # give orchestrator time to process
    await asyncio.sleep(0.2)
    await orch.stop()

    assert any("answer" in t for t in collected), f"Expected tts to be called with answer, got {collected}"


@pytest.mark.asyncio
async def test_orchestrator_triggers_on_timeout_finalization():
    stt = _stt_stream_partial_only()
    llm = MockLLMRunner(["Timed out response."])
    collected = []

    async def tts_cb(text: str):
        collected.append(text)

    cfg = OrchestratorConfig(turn_final_timeout_s=0.05)
    orch = Orchestrator(stt_stream=stt, llm_runner=llm, tts_callback=tts_cb, config=cfg)
    await orch.start()

    # allow time for timeout to trigger
    await asyncio.sleep(0.2)
    await orch.stop()

    assert any("Timed out" in t for t in collected), f"Expected tts to be called with timeout response, got {collected}"
