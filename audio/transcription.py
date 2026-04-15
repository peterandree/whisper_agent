
import gc
import torch
import whisperx
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from faster_whisper import WhisperModel
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)




    path: Path, device: str, compute_type: str, hf_token: str
) -> str:
    """
    Transcribe an audio file, align word-level timestamps, and perform speaker diarization.

    Args:
        path (Path): Path to the audio file.
        device (str): Device to use ('cpu' or 'cuda').
        compute_type (str): Compute type for model ('float32' or 'float16').
        hf_token (str): HuggingFace token for diarization model.

    Returns:
        str: Formatted transcript with speaker changes.
    """
    logger.debug(f"Loading audio from: {path}")
    audio = whisperx.load_audio(str(path))
    logger.debug(f"Audio loaded. Type: {type(audio)}, Length: {getattr(audio, 'shape', 'unknown')}")

    # Phase 1: Streaming transcription via faster-whisper generator
    logger.info("Loading model and transcribing...")
    fw_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    logger.debug("WhisperModel loaded.")
    segments_generator, info = fw_model.transcribe(audio, beam_size=5)
    logger.info(f"Detected language: {info.language}")

    raw_segments: List[Dict[str, Any]] = []
    for seg in segments_generator:
        logger.info(f"[{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text.strip()}")
        raw_segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

    logger.debug(f"Transcription complete. Segments: {len(raw_segments)}")
    del fw_model; gc.collect(); torch.cuda.empty_cache()
    logger.debug("WhisperModel deleted and GPU cache cleared.")

    # Phase 2: Align word-level timestamps
    logger.info(f"Aligning timestamps... (segments: {len(raw_segments)}, audio type: {type(audio)}, device: {device})")
    try:
        logger.debug("Loading align model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        logger.debug("Align model loaded.")
        result = whisperx.align(
            raw_segments, model_a, metadata, audio, device,
            return_char_alignments=False
        )
        logger.debug("Alignment complete.")
        del model_a; gc.collect(); torch.cuda.empty_cache()
        logger.debug("Align model deleted and GPU cache cleared.")
    except Exception as e:
        logger.error(f"Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Phase 3: Diarize
    logger.info("Running speaker diarization...")
    try:
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        logger.debug("DiarizationPipeline loaded.")
        diarize_segments = diarize_model(audio)
        logger.debug("Diarization complete.")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.debug("Speaker assignment complete.")
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Format with speaker-change paragraphs
    lines: List[str] = []
    current_speaker: Optional[str] = None
    current_text: List[str] = []

    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        logger.info(f"[{start:6.1f}s -> {end:6.1f}s] {speaker}: {text}")

        if speaker != current_speaker:
            if current_text:
                lines.append(f"{current_speaker}: " + " ".join(current_text))
                lines.append("")
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)

    if current_text:
        lines.append(f"{current_speaker}: " + " ".join(current_text))

    return "\n".join(lines)
