
import os
import sys
import shutil
import warnings
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


def process(path: Path) -> None:
    """
    Process an audio file: transcribe, summarize, and write outputs.

    Args:
        path (Path): Path to the audio file to process.
    """
    logging.info(f"Processing: {path.name}")
    try:
        device = "cuda"  # "cpu" or "cuda" 
        compute_type = "float32" if device == "cpu" else "float16"
        hf_token = os.environ["HF_TOKEN"]

        transcript = transcribe(path, device, compute_type, hf_token)
        logging.info(f"Transcript generated. Length: {len(transcript)} chars")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{path.stem}_{ts}"

        transcript_file = OUTPUT_DIR / (stem + ".transcript.txt")
        transcript_file.write_text(transcript, encoding="utf-8")
        logging.info(f"Transcript written to: {transcript_file}")

        summary = summarize(transcript, OLLAMA_URL, OLLAMA_MODEL)
        logging.info(f"Summary generated. Length: {len(summary)} chars")

        out_file = OUTPUT_DIR / (stem + ".md")
        out_file.write_text(summary, encoding="utf-8")
        logging.info(f"Summary written to: {out_file}")
    except Exception as e:
        import traceback
        logging.error(f"Failed to process {path.name}: {e}")
        traceback.print_exc()




def main() -> None:
    """
    Main entry point. Starts the file watcher for processing audio files.
    """
    start_watcher(process)

if __name__ == "__main__":
    main()
