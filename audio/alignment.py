import logging
import traceback
from typing import Any

import whisperx

from audio.gpu_utils import log_device, cleanup_gpu

logger = logging.getLogger(__name__)


def run_alignment(
    raw_segments: list[dict],
    language: str,
    align_model_name: str | None,
    audio: Any,
    device: str,
) -> dict:
    """
    Align word-level timestamps using wav2vec2.
    Falls back to raw segment timestamps on any failure.
    """
    logger.info(
        f"Loading alignment model for '{language}'"
        + (f": {align_model_name}" if align_model_name else " (whisperx built-in)")
        + f" for {len(raw_segments)} segments..."
    )
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
            model_name=align_model_name,
        )
        log_device(f"Alignment model ({align_model_name or 'whisperx built-in'})", device)
        logger.info("Alignment model loaded. Aligning...")
        result = whisperx.align(
            raw_segments, model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        logger.info("Alignment complete.")
        del model_a
        cleanup_gpu("alignment model deletion")
        return result
    except Exception as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during alignment.")
        logger.error(f"Alignment failed: {e}")
        traceback.print_exc()
        logger.warning("Falling back to raw segment timestamps.")
        return {"segments": raw_segments}