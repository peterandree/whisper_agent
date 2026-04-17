import os
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

import datetime
import logging
from pathlib import Path

from audio.preprocessing import preprocess_audio, cleanup_temp_audio
from audio.language_detection import detect_language
from audio.model_selector import select_align_model
from audio.transcription_core import load_audio, run_transcription
from audio.alignment import run_alignment
from audio.diarization import run_diarization
from audio.progress import format_transcript

logger = logging.getLogger(__name__)


def transcribe(path: Path, device: str, compute_type: str, hf_token: str) -> str:
    """
    Full transcription pipeline:
      Preprocessing → Language detection → Transcription →
      Alignment → Diarization → Formatting

    Intended to run inside a worker process that exits via os._exit(0)
    to avoid CTranslate2/WDDM destructor crashes on Windows.
    """
    start_time = datetime.datetime.now()
    logger.info(f"Transcription pipeline started at {start_time:%Y-%m-%d %H:%M:%S}")

    # Preprocessing
    audio_path, total_duration = preprocess_audio(path)

    # Phase 0: Language detection
    t0 = datetime.datetime.now()
    language = detect_language(audio_path, device)
    align_model_name = select_align_model(language)
    skip_alignment = align_model_name == "SKIP"
    logger.info(f"Phase 0 (language detection) took {(datetime.datetime.now() - t0).total_seconds():.2f}s")

    # Load audio into memory, then release temp file
    audio = load_audio(audio_path)
    cleanup_temp_audio(path, audio_path)

    # Phase 1: Transcription
    t1 = datetime.datetime.now()
    raw_segments = run_transcription(audio, language, device, compute_type, total_duration)
    logger.info(f"Phase 1 (transcription) took {(datetime.datetime.now() - t1).total_seconds():.2f}s")

    # Phase 2: Alignment
    t2 = datetime.datetime.now()
    if skip_alignment:
        logger.warning(
            f"Skipping alignment — no model for language '{language}'. "
            "Using raw segment timestamps."
        )
        result = {"segments": raw_segments}
    else:
        result = run_alignment(raw_segments, language, align_model_name, audio, device)
    logger.info(f"Phase 2 (alignment) took {(datetime.datetime.now() - t2).total_seconds():.2f}s")

    # Phase 3: Diarization
    t3 = datetime.datetime.now()
    result = run_diarization(audio, result, hf_token, device)
    logger.info(f"Phase 3 (diarization) took {(datetime.datetime.now() - t3).total_seconds():.2f}s")

    total_time = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"Transcription pipeline finished. Total time: {total_time:.2f}s")

    return format_transcript(result["segments"], total_duration)