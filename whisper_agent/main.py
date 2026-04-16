import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import os
import sys
import shutil
import logging
from datetime import datetime
from pathlib import Path
from config.settings import OUTPUT_DIR, OLLAMA_URL, OLLAMA_MODEL, DEVICE, COMPUTE_TYPE
from audio.transcription import transcribe
from summarization.summary import summarize
from watcher.watcher import start_watcher

_ffmpeg_exe = shutil.which("ffmpeg")
if _ffmpeg_exe:
    os.add_dll_directory(os.path.dirname(_ffmpeg_exe))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def process(path: Path, hf_token: str) -> None:
    """
    Process an audio file: transcribe, summarize, and write outputs.

    Args:
        path (Path): Path to the audio file to process.
    """
    logging.info(f"Processing: {path.name}")
    try:
        logging.info("Starting transcription...")
        transcript = transcribe(path, DEVICE, COMPUTE_TYPE, hf_token)

        logging.info(f"Transcription complete. Length: {len(transcript)} chars. Starting summarization...")
        logging.info(f"Transcript type: {type(transcript)}, length: {len(transcript)}")
        logging.info(f"Transcript sample: {transcript[:200]!r}")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{path.stem}_{ts}"

        transcript_file = OUTPUT_DIR / (stem + ".transcript.txt")
        logging.info(f"About to write transcript to: {transcript_file}")
        try:
            transcript_file.write_text(transcript, encoding="utf-8")
            logging.info(f"Transcript written to: {transcript_file}")
        except Exception as e:
            logging.error(f"Failed to write transcript file: {e}")
            raise

        logging.info("Sending transcript to Ollama for summarization. If this is the first request, model loading may take several minutes...")
        summary = summarize(transcript, OLLAMA_URL, OLLAMA_MODEL)
        logging.info(f"Summary generated. Length: {len(summary)} chars. Writing summary to disk...")

        out_file = OUTPUT_DIR / (stem + ".md")
        logging.info(f"About to write summary to: {out_file}")
        try:
            out_file.write_text(summary, encoding="utf-8")
            logging.info(f"Summary written to: {out_file}")
        except Exception as e:
            logging.error(f"Failed to write summary file: {e}")
            raise
    except Exception as e:
        import traceback
        logging.error(f"Failed to process {path.name}: {e}")
        traceback.print_exc()

def _verify_output_dir(path: Path) -> None:
    test_file = path / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as e:
        raise RuntimeError(f"OUTPUT_DIR is not writable: {path} — {e}")

def main() -> None:
    """
    Main entry point. Starts the file watcher for processing audio files.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _verify_output_dir(OUTPUT_DIR)
    
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with: winget install Gyan.FFmpeg"
        )
    
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN environment variable is not set. Get a token at https://huggingface.co/settings/tokens")

    # Wrap process to provide hf_token
    def process_with_token(path: Path):
        return process(path, hf_token)

    start_watcher(process_with_token)

if __name__ == "__main__":
    main()
