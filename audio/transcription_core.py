import logging
from pathlib import Path
from typing import Any

import whisperx
from faster_whisper import WhisperModel

from audio.audio_utils import format_seconds
from audio.gpu_utils import log_device

logger = logging.getLogger(__name__)


def load_audio(audio_path: Path) -> Any:
    """Load audio file into a numpy array at 16 kHz."""
    try:
        audio = whisperx.load_audio(str(audio_path))
        logger.info(
            f"Audio loaded. Samples: {audio.shape[0]}, "
            f"Duration: {audio.shape[0] / 16000:.1f}s"
        )
        return audio
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        raise


def run_transcription(
    audio: Any,
    language: str,
    device: str,
    compute_type: str,
    total_duration: float,
) -> list[dict]:
    """
    Load WhisperModel and transcribe audio.
    Returns raw segment list. Does NOT delete the model — the worker
    process exits via os._exit(0) which skips all destructors, avoiding
    the 0xC0000409 stack corruption crash on Windows WDDM.
    """
    logger.info("Loading WhisperModel (large-v3-turbo)...")
    fw_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    log_device("WhisperModel (large-v3-turbo)", device)
    logger.info("WhisperModel loaded. Starting transcription...")

    segments_generator, _ = fw_model.transcribe(
        audio, beam_size=5, language=language
    )

    raw_segments: list[dict] = []
    for seg in segments_generator:
        if total_duration > 0:
            percent  = min(100.0, 100.0 * seg.end / total_duration)
            progress = f"[{percent:5.1f}%] "
        else:
            progress = ""
        t_start = format_seconds(seg.start)
        t_end   = format_seconds(seg.end)
        logger.info(f"{progress}[{t_start} -> {t_end}] {seg.text.strip()}")
        raw_segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

    logger.info(f"Transcription complete. Segments: {len(raw_segments)}")
    return raw_segments