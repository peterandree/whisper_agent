import logging
import traceback
from typing import Any

import whisperx
from whisperx.diarize import DiarizationPipeline

from audio.gpu_utils import log_device, cleanup_gpu

logger = logging.getLogger(__name__)


def run_diarization(
    audio: Any,
    result: dict,
    hf_token: str,
    device: str,
) -> dict:
    """
    Run speaker diarization and assign speaker labels to segments.
    Raises on failure — diarization is not optional.
    """
    logger.info("Loading diarization pipeline...")
    try:
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        log_device("DiarizationPipeline", device)
        logger.info("Diarization pipeline loaded. Running diarization...")
        diarize_segments = diarize_model(audio)
        logger.info("Diarization complete. Assigning speakers...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker assignment complete.")
        del diarize_model
        cleanup_gpu("diarization model deletion")
        return result
    except Exception as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during diarization.")
        logger.error(f"Diarization failed: {e}")
        traceback.print_exc()
        raise