from __future__ import annotations
import asyncio
from .llama_runner import LlamaCPPRunner
from .interfaces import GenerationOptions


async def main():
    # Passe den Pfad zum quantisierten gguf Modell an
    runner = LlamaCPPRunner(model_path="/path/to/7B-q4_0.gguf", n_threads=8, n_ctx=1024)
    opts = GenerationOptions(max_tokens=64, temperature=0.2)
    prompt = "Write a concise coaching tip for improving daily focus."

    print("--- streaming generation ---")
    try:
        async for ev in runner.generate_stream(prompt, opts):
            print(ev.token, end="", flush=True)
    except Exception as e:
        print("Error during generation:", e)


if __name__ == "__main__":
    asyncio.run(main())
