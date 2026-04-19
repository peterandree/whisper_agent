# whisper_agent

Windows-only Python daemon that watches a folder for audio/video files, transcribes them with WhisperX (faster-whisper + pyannote diarization, CUDA), then summarizes the transcript via a local Ollama instance. Outputs `.transcript.txt` and `.summary.md` per file. Runs as a long-lived process; entry point is `whisper-agent` CLI.

## Tech Stack

- Python >= 3.11, < 3.13
- `faster-whisper` >= 1.2.1 — CTranslate2-based Whisper inference
- `whisperx` >= 3.8.5 — word-level alignment + speaker diarization (pyannote)
- `torch` >= 2.7.0 / `torchaudio` / `torchvision` — from PyTorch CUDA 12.8 index
- `watchdog` >= 6.0.0 — filesystem watcher
- `fastapi` >= 0.135.3 + `uvicorn` >= 0.44.0 — optional REST interface
- `requests` >= 2.33.1 — Ollama HTTP calls
- Package manager: `uv` (lockfile: `uv.lock`)
- Testing: `pytest` (in `tests/`)
- **Requires**: NVIDIA GPU with CUDA 12.8, `ffmpeg` on PATH, `HF_TOKEN` env var, local Ollama with `mistral:7b`

## Project Structure

```
whisper_agent/
  main.py                   — entry point: startup checks, Ollama lifecycle, watcher launch, process() per file
audio/
  transcription_runner.py   — top-level transcription call (runs transcription in subprocess to isolate CUDA context)
  transcription.py          — WhisperX transcription orchestration
  transcription_core.py     — raw faster-whisper model call
  transcription_worker.py   — subprocess worker target for transcription
  alignment.py              — WhisperX word alignment
  diarization.py            — pyannote speaker diarization
  diarization_worker.py     — subprocess worker target for diarization
  speaker_db.py             — SQLite speaker embedding database
  speaker_resolver.py       — matches diarized speaker IDs to known names
  register_speaker.py       — CLI tool: assign names to unresolved speakers from .pending_speakers.json
  model_selector.py         — selects Whisper model size based on available VRAM
  language_detection.py     — detects audio language before transcription
  preprocessing.py          — audio normalization / format conversion via ffmpeg
  gpu_utils.py              — CUDA context helpers, device checks
  vram.py                   — VRAM measurement utilities
  audio_utils.py            — misc audio helpers
  progress.py               — progress reporting
summarization/
  summary.py                — sends transcript to Ollama, returns markdown summary
watcher/
  watcher.py                — watchdog-based folder watcher, calls process() callback
config/
  settings.py               — all configuration constants, read from env vars with defaults
  language_models.py        — Whisper model size / language mappings
tests/                      — pytest test suite
diagnose_alignment.py       — standalone debug script for alignment issues (not production)
run_in_venv.ps1             — PowerShell helper to launch in correct venv on Windows
```

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|---|---|---|
| `HF_TOKEN` | — (required) | HuggingFace token for pyannote diarization model |
| `WHISPER_AGENT_WATCH_DIR` | `C:/data/meetings/raw` | Folder to watch for new audio files |
| `WHISPER_AGENT_OUTPUT_DIR` | `C:/data/meetings/summaries` | Output folder for transcripts and summaries |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `OLLAMA_MODEL` | `mistral:7b` | Ollama model to use for summarization |
| `OLLAMA_TIMEOUT` | `600` | Ollama request timeout in seconds |
| `SPEAKER_DB_PATH` | `C:/data/meetings/speakers.db` | SQLite speaker embedding DB path |

All configuration lives in `config/settings.py`. Never hardcode paths or env values elsewhere.

## Key Architectural Constraint: CUDA Context Isolation

Ollama and WhisperX cannot share the CUDA context under Windows WDDM. The processing pipeline in `main.py` is therefore:

1. `_stop_ollama()` — kill Ollama process to release GPU
2. `transcribe()` — runs in a **subprocess** (via `transcription_runner.py`) to get a clean CUDA context
3. `_start_ollama()` — restart Ollama, poll until ready
4. `summarize()` — send transcript to Ollama

Do not collapse these steps or run transcription and Ollama in the same process.

## Commands

```bash
# Install (Windows, CUDA 12.8 required)
uv sync

# Run the agent
uv run whisper-agent

# Assign speaker names to a pending file
uv run python -m audio.register_speaker assign --pending "C:/data/meetings/summaries/file.pending_speakers.json"

# Run tests
uv run pytest

# Diagnose alignment issues
uv run python diagnose_alignment.py
```

## Coding Conventions

- Type hints on all public functions
- All configuration via `config/settings.py` constants — never `os.environ` inline elsewhere
- All ffmpeg/subprocess calls must handle Windows path quoting correctly
- Transcription must always run in a subprocess — never in the main process (CUDA isolation)
- Log with the module-level `logger = logging.getLogger(__name__)` pattern — never `print()`
- Audio extensions filter is defined in `config/settings.py` (`AUDIO_EXTENSIONS`) — do not duplicate it

## Agent Boundaries

- ✅ Always: read `config/settings.py` before adding any config value; run `uv run pytest` before marking a task done
- ✅ Always: preserve the stop-Ollama → transcribe-in-subprocess → start-Ollama sequence in `main.process()`
- ⚠️ Ask first: adding a new dependency, changing the subprocess isolation strategy, modifying speaker DB schema
- 🚫 Never: call Ollama and run WhisperX in the same process, hardcode paths outside `config/settings.py`, commit `HF_TOKEN` or any credentials, run ffmpeg directly without checking it is on PATH first
