Ask Your Coach is an interactive, voice-driven learning system that lets you talk to a personal AI coach about a preselected topic.
Your voice is transcribed in real time (STT), processed by an LLM enhanced with a topic-specific knowledge base (RAG), and answered back instantly using text-to-speech (TTS).
The system is designed for hands-free learning, natural conversation, and sub-2-second end-to-end latency, making it ideal for learning on the go — for example while walking, commuting, or driving.

## Konfiguration

- Beispielkonfig findest du in `config.example.yaml` im Projektstamm. Kopiere sie bei Bedarf nach `config.yaml` oder lege eine Benutzerdatei in `~/.config/ask-your-coach/config.yaml` an.
- Alternativ kannst du Umgebungsvariablen setzen; wichtige Optionen:
	- `LMSTUDIO_URL` — LMStudio Endpoint (z.B. `http://localhost:1234`)
	- `LMSTUDIO_MODEL` — Modell‑ID in LMStudio (z.B. `openai/gpt-oss-20b`)
	- `AYC_LL_MODEL` — Pfad zu lokalem `.gguf` Modell (llama.cpp)
	- `AYC_N_THREADS` — Threads für llama.cpp

Beispiel: kopieren und Demo starten

```bash
cp config.example.yaml config.yaml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export LMSTUDIO_URL=http://localhost:1234
export LMSTUDIO_MODEL=openai/gpt-oss-20b
python ask_your_coach/demo/demo_orchestrator.py
```

- Prompts befinden sich im Ordner `ask_your_coach/prompts/`. Neue Modus‑Templates (z. B. `general_coaching.txt`) werden automatisch geladen; der Orchestrator hot‑reloadet Änderungen beim Start.

Bei Fragen oder wenn du eine andere Default‑Konfiguration wünschst, sag Bescheid.
