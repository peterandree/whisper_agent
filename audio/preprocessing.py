import subprocess
import tempfile
import logging
from pathlib import Path

from audio.audio_utils import get_audio_duration, format_seconds

logger = logging.getLogger(__name__)


def preprocess_audio(path: Path) -> tuple[Path, float]:
    """
    Denoise (anlmdn) and trim silences from an audio file using ffmpeg.
    Returns the processed file path and total duration in seconds.
    If ffmpeg fails, returns the original path as fallback.

    The caller is responsible for deleting the returned temp file when
    it differs from the input path.
    """
    trimmed_path: Path | None = None
    audio_path = path

    try:
        with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as tmp:
            trimmed_path = Path(tmp.name)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-af", "anlmdn,silenceremove=1:0:-50dB",
            str(trimmed_path),
        ]
        logger.info(f"Preprocessing audio: {' '.join(ffmpeg_cmd)}")
        subprocess.run(
            ffmpeg_cmd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        logger.info(f"Preprocessing complete. Temp file: {trimmed_path}")
        audio_path = trimmed_path
    except Exception as e:
        logger.error(
            f"Audio preprocessing failed: {e}. Using original file."
        )
        if trimmed_path and trimmed_path.exists():
            trimmed_path.unlink(missing_ok=True)
        audio_path = path

    total_duration = get_audio_duration(audio_path)
    if total_duration > 0:
        logger.info(
            f"Audio duration: {format_seconds(total_duration)} "
            f"({total_duration:.1f}s)"
        )
    else:
        logger.warning("Could not determine audio duration.")

    return audio_path, total_duration


def cleanup_temp_audio(original: Path, processed: Path) -> None:
    """Delete the preprocessed temp file if it differs from the original."""
    if processed != original:
        try:
            processed.unlink()
            logger.info(f"Deleted temp file: {processed}")
        except Exception as e:
            logger.warning(f"Could not delete temp file {processed}: {e}")