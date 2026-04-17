"""
Orchestrates two worker subprocesses:
  1. transcription_worker.py  — preprocessing, transcription, alignment
  2. diarization_worker.py    — diarization, formatting

Each worker exits via os._exit(0) so CTranslate2 and pyannote destructors
never run inside a live Python process, avoiding 0xC0000409 on Windows WDDM.
Between the two workers the CUDA context is fully released by the OS.
"""
import sys
import json
import tempfile
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_worker(cmd: list[str], label: str) -> None:
    """Run a worker subprocess, streaming its output and raising on failure."""
    logger.info(f"Launching {label}...")
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} exited with code {result.returncode}."
        )
    logger.info(f"{label} completed.")


def transcribe_in_subprocess(
    path: Path,
    hf_token: str,
    device: str,
    compute_type: str,
) -> str:
    """
    Run the full transcription pipeline across two isolated worker processes
    and return the final transcript string.
    """
    worker_dir = Path(__file__).parent

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as tmp_json:
        segments_file = Path(tmp_json.name)

    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as tmp_txt:
        output_file = Path(tmp_txt.name)

    try:
        # Worker 1: transcription + alignment → segments JSON
        _run_worker(
            [
                sys.executable,
                str(worker_dir / "transcription_worker.py"),
                str(path), hf_token, device, compute_type,
                str(segments_file),
            ],
            label="transcription worker",
        )

        # Worker 2: diarization → final transcript
        _run_worker(
            [
                sys.executable,
                str(worker_dir / "diarization_worker.py"),
                str(path), hf_token, device,
                str(segments_file),
                str(output_file),
            ],
            label="diarization worker",
        )

        transcript = output_file.read_text(encoding="utf-8")
        logger.info(f"Transcript length: {len(transcript)} chars.")
        return transcript

    finally:
        segments_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)