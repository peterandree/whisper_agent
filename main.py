import os
import sys
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from config.settings import OUTPUT_DIR, OLLAMA_URL, OLLAMA_MODEL
from audio.transcription import transcribe
from summarization.summary import summarize
from watcher.watcher import start_watcher

warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

_ffmpeg_exe = shutil.which("ffmpeg")
if _ffmpeg_exe:
    os.add_dll_directory(os.path.dirname(_ffmpeg_exe))

def process(path: Path):
    print(f"[LOG] [START] Processing file: {path.name}")
    try:
        print(f"[LOG] [CHECKPOINT] About to set device and compute_type")
        device = "cuda"  # "cpu" or "cuda" 
        compute_type = "float32" if device == "cpu" else "float16"
        print(f"[LOG] [CHECKPOINT] About to read HF_TOKEN from environment")
        hf_token = os.environ["HF_TOKEN"]

        print(f"[LOG] [CHECKPOINT] About to start transcription and diarization for: {path}")
        transcript = transcribe(path, device, compute_type, hf_token)
        print(f"[LOG] [CHECKPOINT] Finished transcription/diarization. Transcript length: {len(transcript)} chars")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{path.stem}_{ts}"

        transcript_file = OUTPUT_DIR / (stem + ".transcript.txt")
        print(f"[LOG] [CHECKPOINT] About to write transcript to: {transcript_file}")
        transcript_file.write_text(transcript, encoding="utf-8")
        print(f"[LOG] [CHECKPOINT] Finished writing transcript to: {transcript_file}")

        print(f"[LOG] [CHECKPOINT] About to start summary generation...")
        summary = summarize(transcript, OLLAMA_URL, OLLAMA_MODEL)
        print(f"[LOG] [CHECKPOINT] Finished summary generation. Length: {len(summary)} chars")

        out_file = OUTPUT_DIR / (stem + ".md")
        print(f"[LOG] [CHECKPOINT] About to write summary to: {out_file}")
        out_file.write_text(summary, encoding="utf-8")
        print(f"[LOG] [CHECKPOINT] Finished writing summary to: {out_file}")
        print(f"[LOG] [END] Finished processing file: {path.name}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[!] Failed to process {path.name}: {e}", file=sys.stderr)



def main():
    start_watcher(process)

if __name__ == "__main__":
    main()
