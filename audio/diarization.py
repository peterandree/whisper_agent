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
    source_file: str = "",
) -> tuple[dict, Any]:
    """
    Run speaker diarization, resolve speaker labels to known names,
    and return (result_with_speakers, diarize_segments_dataframe).
    diarize_segments is returned so the caller can build a pending
    registration file without re-running diarization.
    """
    logger.info("Loading diarization pipeline...")
    try:
        diarize_model    = DiarizationPipeline(token=hf_token, device=device)
        log_device("DiarizationPipeline", device)
        logger.info("Diarization pipeline loaded. Running diarization...")
        diarize_segments = diarize_model(audio)
        logger.info("Diarization complete. Assigning speakers...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker assignment complete.")
        del diarize_model
        cleanup_gpu("diarization model deletion")

        from audio.speaker_resolver import resolve_speakers
        result = resolve_speakers(result, diarize_segments, audio, device, source_file)

        return result, diarize_segments

    except Exception as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during diarization.")
        logger.error(f"Diarization failed: {e}")
        traceback.print_exc()
        raise