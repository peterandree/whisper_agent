import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")


import os
import sys
import time
import shutil
import logging
import subprocess
import traceback
import torch
from datetime import datetime
from pathlib import Path
from config.settings import OUTPUT_DIR, OLLAMA_URL, OLLAMA_MODEL, DEVICE, COMPUTE_TYPE
from audio.transcription_runner import transcribe_in_subprocess as transcribe
from summarization.summary import summarize
from watcher.watcher import start_watcher


# Allow ffmpeg DLLs to be found on Windows before any media library loads
_ffmpeg_exe = shutil.which("ffmpeg")
if _ffmpeg_exe:
    os.add_dll_directory(os.path.dirname(_ffmpeg_exe))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ollama lifecycle
# ---------------------------------------------------------------------------


def _stop_ollama() -> None:
    """
    Kill the Ollama process to release its CUDA context before transcription.
    Under Windows WDDM, two processes cannot share the GPU context freely —
    Ollama holding it causes whisperx to crash when it attempts CUDA init.
    """
    if not shutil.which("ollama"):
        return
    result = subprocess.run(
        ["taskkill", "/F", "/IM", "ollama.exe"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode == 0:
        logger.info("Ollama stopped. Waiting for CUDA context to be released...")
        time.sleep(2.0)
    else:
        logger.debug("taskkill returned non-zero — Ollama may not have been running.")


def _wait_for_ollama(
    url: str = "http://localhost:11434/api/tags",
    timeout: float = 60.0,
    interval: float = 1.0,
) -> None:
    """Poll Ollama's tags endpoint until it responds or timeout is reached."""
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                elapsed = timeout - (deadline - time.time())
                logger.info(f"Ollama ready after {elapsed:.1f}s.")
                return
        except Exception:
            pass
        time.sleep(interval)
    logger.warning(
        f"Ollama did not respond within {timeout:.0f}s. "
        "Proceeding anyway — first request may be slow."
    )


def _start_ollama() -> None:
    """
    Restart Ollama after transcription completes so summarization can proceed.
    Polls /api/tags until the server is ready — no fixed sleep.
    """
    if not shutil.which("ollama"):
        logger.warning("ollama not found on PATH — skipping restart.")
        return
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("Ollama restarted. Waiting for server to become ready...")
    _wait_for_ollama()


# ---------------------------------------------------------------------------
# CUDA startup check
# ---------------------------------------------------------------------------


def _initialize_cuda() -> None:
    if not torch.cuda.is_available():
        logger.warning("CUDA not available — workers will run on CPU.")
        return
    props = torch.cuda.get_device_properties(0)
    logger.info(
        f"GPU available: {props.name}, "
        f"VRAM: {props.total_memory / 1024 ** 3:.1f} GB"
    )


# ---------------------------------------------------------------------------
# Output directory guard
# ---------------------------------------------------------------------------


def _verify_output_dir(path: Path) -> None:
    test_file = path / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
    except OSError as e:
        raise RuntimeError(f"OUTPUT_DIR is not writable: {path} — {e}") from e


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------


def process(path: Path, hf_token: str) -> None:
    """
    Process one audio file: stop Ollama, transcribe, restart Ollama,
    summarize, write outputs.

    Args:
        path:     Path to the audio file to process.
        hf_token: HuggingFace token for the diarization model.
    """
    logger.info(f"Processing: {path.name}")

    # Stop Ollama so its CUDA context is freed before any model loads
    _stop_ollama()

    transcript: str | None = None
    try:
        logger.info("Starting transcription...")
        transcript = transcribe(path, hf_token, DEVICE, COMPUTE_TYPE)
        logger.info(
            f"Transcription complete. "
            f"Length: {len(transcript)} chars."
        )
        logger.info(f"Transcript sample: {transcript[:200]!r}")

        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{path.stem}_{ts}"

        transcript_file = OUTPUT_DIR / (stem + ".transcript.txt")
        logger.info(f"Writing transcript to: {transcript_file}")
        try:
            transcript_file.write_text(transcript, encoding="utf-8")
            logger.info("Transcript written.")
        except Exception as e:
            logger.error(f"Failed to write transcript file: {e}")
            raise

    except Exception as e:
        logger.error(f"Transcription failed for {path.name}: {e}")
        traceback.print_exc()
        # Restart Ollama even on failure so the service is not left stopped
        _start_ollama()
        return

    # Restart Ollama before summarization
    _start_ollama()

    try:
        logger.info("Starting summarization...")
        summary = summarize(transcript, OLLAMA_URL, OLLAMA_MODEL)
        logger.info("Summarization complete.")

        summary_file = OUTPUT_DIR / (stem + ".summary.md")
        logger.info(f"Writing summary to: {summary_file}")
        try:
            summary_file.write_text(summary, encoding="utf-8")
            logger.info("Summary written.")
        except Exception as e:
            logger.error(f"Failed to write summary file: {e}")
            raise

    except Exception as e:
        logger.error(f"Summarization failed for {path.name}: {e}")
        traceback.print_exc()

    # Hint for unresolved speakers
    pending_file = OUTPUT_DIR / (stem + ".pending_speakers.json")
    if pending_file.exists():
        logger.info(
            f"─────────────────────────────────────────────────────────────\n"
            f"  {len(__import__('json').loads(pending_file.read_text()).get('speaker_turns', {}))} unresolved speaker(s) identified in this recording.\n"
            f"  To assign names and improve future recognition, run:\n"
            f"\n"
            f"    python -m audio.register_speaker assign --pending \"{pending_file}\"\n"
            f"─────────────────────────────────────────────────────────────"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Startup checks, then launch the file watcher.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _verify_output_dir(OUTPUT_DIR)

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it with: winget install Gyan.FFmpeg"
        )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Get a token at https://huggingface.co/settings/tokens"
        )

    # Stop any running Ollama instance before the CUDA check so the context
    # is free when PyTorch initializes
    _stop_ollama()
    _initialize_cuda()

    logger.info(f"Watching for audio files. Output directory: {OUTPUT_DIR}")

    def process_with_token(path: Path) -> None:
        process(path, hf_token)

    start_watcher(process_with_token)


if __name__ == "__main__":
    main()