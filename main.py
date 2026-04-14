import os
import sys
import gc
import torch
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from config.settings import WATCH_DIR, OUTPUT_DIR, OLLAMA_URL, OLLAMA_MODEL, AUDIO_EXTENSIONS
from audio.transcription import transcribe
from summarization.summary import summarize
from watcher.watcher import start_watcher

warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

_ffmpeg_exe = shutil.which("ffmpeg")
if _ffmpeg_exe:
    os.add_dll_directory(os.path.dirname(_ffmpeg_exe))

def process(path: Path):
    print(f"[+] Processing: {path.name}")
    try:
        device = "cuda"
        compute_type = "float16"
        hf_token = os.environ["HF_TOKEN"]
        transcript = transcribe(path, device, compute_type, hf_token)
        print(f"[+] Transcription done ({len(transcript)} chars)")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{path.stem}_{ts}"

        transcript_file = OUTPUT_DIR / (stem + ".transcript.txt")
        transcript_file.write_text(transcript, encoding="utf-8")
        print(f"[+] Transcript written to: {transcript_file}")

        summary = summarize(transcript, OLLAMA_URL, OLLAMA_MODEL)
        out_file = OUTPUT_DIR / (stem + ".md")
        out_file.write_text(summary, encoding="utf-8")
        print(f"[+] Summary written to: {out_file}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[!] Failed to process {path.name}: {e}", file=sys.stderr)


def main():
    start_watcher()

if __name__ == "__main__":
    main()
