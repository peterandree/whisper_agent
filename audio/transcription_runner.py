"""
Launches transcription_worker.py as a child process.
The child owns all CUDA/CTranslate2 resources. When it exits normally,
the OS reclaims everything without a destructor crash.
"""
import sys
import tempfile
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def transcribe_in_subprocess(
    path: Path,
    hf_token: str,
    device: str,
    compute_type: str,
) -> str:
    """
    Run the full transcription pipeline in a child process and return
    the transcript string. The child process exits cleanly, releasing
    all CUDA/CTranslate2 resources via OS teardown instead of manual
    Python destructors — avoiding the WDDM TDR crash on Windows.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        output_file = Path(tmp.name)

    worker = Path(__file__).parent / "transcription_worker.py"

    cmd = [
        sys.executable, str(worker),
        str(path),
        hf_token,
        device,
        compute_type,
        str(output_file),
    ]

    logger.info(f"Launching transcription worker: {worker.name}")
    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        output_file.unlink(missing_ok=True)
        raise RuntimeError(
            f"Transcription worker exited with code {result.returncode}."
        )

    transcript = output_file.read_text(encoding="utf-8")
    output_file.unlink(missing_ok=True)
    logger.info(f"Transcription worker completed. Transcript length: {len(transcript)} chars.")
    return transcript