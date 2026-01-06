"""Ein kleines Demo-Skript, das STT -> Orchestrator -> MockLLM -> TTS-Callback zeigt.

Dieses Skript benutzt Mock-Implementierungen, damit es lokal ohne Modelle läuft.
"""
import asyncio

from ask_your_coach.orchestrator.orchestrator import Orchestrator, OrchestratorConfig
from ask_your_coach.llm.interfaces import TokenEvent, GenerationOptions


class MockLLMRunner:
    def __init__(self, tokens, delay: float = 0.02):
        self.tokens = tokens
        self.delay = delay

    async def generate_stream(self, prompt: str, options: GenerationOptions):
        for i, t in enumerate(self.tokens):
            await asyncio.sleep(self.delay)
            yield TokenEvent(token=t, is_first=(i == 0))

    async def generate(self, prompt: str, options: GenerationOptions) -> str:
        return "".join(self.tokens)


async def mock_stt_stream():
    # Simulate a short user utterance coming as partial then final
    yield {"text": "Hello coach", "is_final": False}
    await asyncio.sleep(0.05)
    yield {"text": " please give me a short tip.", "is_final": True}


async def tts_cb(text: str) -> None:
    # In a real setup this would call the TTS playback wrapper.
    print(f"[TTS CALLBACK] would speak: {text}")


async def main() -> None:
    stt = mock_stt_stream()

    # Prefer LMStudio endpoint if configured
    # Load central configuration
    from ask_your_coach.config import get_config
    cfg = get_config()
    use_real_llm = False
    llm = None
    lmstudio_url = cfg.lmstudio_url
    lmstudio_model = cfg.lmstudio_model
    if lmstudio_url and lmstudio_model:
        try:
            from ask_your_coach.llm.lmstudio_runner import LMStudioRunner

            print(f"Using LMStudio endpoint {lmstudio_url} model {lmstudio_model}")
            llm = LMStudioRunner(base_url=lmstudio_url, model=lmstudio_model)
            use_real_llm = True
        except Exception:
            print("LMStudio runner not available; falling back")

    if llm is None:
        # Try to use real LlamaCPPRunner if available and model path exists
        try:
            from ask_your_coach.llm.llama_runner import LlamaCPPRunner

            model_path = cfg.ayc_ll_model or os.environ.get("AYC_LL_MODEL", "/path/to/7B-q4_0.gguf")
            if os.path.exists(model_path):
                print(f"Using real LlamaCPPRunner with model at {model_path}")
                llm = LlamaCPPRunner(model_path=model_path, n_threads=cfg.n_threads)
                use_real_llm = True
            else:
                print(f"Real model not found at {model_path}; using MockLLMRunner")
        except Exception:
            print("llama_cpp runner not available; using MockLLMRunner")

    if llm is None:
        llm = MockLLMRunner(["Here is a tip."])

    # Try to use real SileroTTS and playback if available
    try:
        from ask_your_coach.tts.silero_tts import SileroTTS
        from ask_your_coach.tts.playback import tts_callback_wrapper
        from ask_your_coach.tts.interfaces import SynthesisOptions

        tts = SileroTTS()
        tts_opts = SynthesisOptions(sample_rate=22050)
        tts_cb_real = tts_callback_wrapper(tts, tts_opts)
        print("Using SileroTTS for playback")
        tts_cb_used = tts_cb_real
    except Exception:
        print("SileroTTS/playback not available; using mock TTS callback")
        tts_cb_used = tts_cb

    cfg = OrchestratorConfig(turn_final_timeout_s=0.3)
    orch = Orchestrator(stt_stream=stt, llm_runner=llm, tts_callback=tts_cb_used, config=cfg)

    print("Starting orchestrator demo...")
    await orch.start()

    # Let the orchestrator run briefly to process the mocked streams
    await asyncio.sleep(1.0)

    await orch.stop()
    print("Demo finished.")


if __name__ == "__main__":
    asyncio.run(main())
