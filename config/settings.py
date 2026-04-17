

import os
from pathlib import Path

# Set device and compute_type as constants here
DEVICE = "cuda"  # or "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "float32"

WATCH_DIR = Path(os.environ.get("WHISPER_AGENT_WATCH_DIR", "C:/data/meetings/raw"))
OUTPUT_DIR = Path(os.environ.get("WHISPER_AGENT_OUTPUT_DIR", "C:/data/meetings/summaries"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral:7b")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".mkv", ".m4a", ".flac"}
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))

SPEAKER_DB_PATH = Path(os.environ.get("SPEAKER_DB_PATH", "C:/data/meetings/speakers.db"))
