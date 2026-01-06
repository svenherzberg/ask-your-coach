import asyncio
from .silero_tts import SileroTTS
from .interfaces import SynthesisOptions


async def main():
    tts = SileroTTS()
    opts = SynthesisOptions(sample_rate=22050, voice=None)
    async for chunk in tts.synthesize_stream("Hello coach, give me a tip.", opts):
        print(f"Got chunk seq={chunk.seq} len={len(chunk.pcm_bytes)}")


if __name__ == "__main__":
    asyncio.run(main())
