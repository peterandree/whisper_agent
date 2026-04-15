from pathlib import Path
import os

WATCH_DIR = Path(os.environ.get("WHISPER_AGENT_WATCH_DIR", "C:/data/meetings/raw"))
OUTPUT_DIR = Path(os.environ.get("WHISPER_AGENT_OUTPUT_DIR", "C:/data/meetings/summaries"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".mkv", ".m4a", ".flac"}

