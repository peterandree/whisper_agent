import subprocess
import tempfile
import gc
import time
import torch
import whisperx
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional
from faster_whisper import WhisperModel
from whisperx.diarize import DiarizationPipeline
from audio.language_detection import detect_language
from audio.model_selector import select_align_model

logger = logging.getLogger(__name__)


def _cleanup_gpu(label: str) -> None:
    """Synchronize CUDA, release memory, wait for driver to propagate, log VRAM state."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1.0)
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    logger.info(
        f"VRAM after {label}: {allocated:.2f} GB allocated, "
        f"{reserved:.2f} GB reserved"
    )


def transcribe(path: Path, device: str, compute_type: str, hf_token: str) -> str:
    """
    Transcribe an audio file, align word-level timestamps, and perform
    speaker diarization.

    Preprocessing: Denoise (anlmdn) and trim silences via ffmpeg.
    Phase 0: Fast language detection via tiny model (first 30s only).
    Phase 1: Full transcription via large-v3-turbo.
    Phase 2: Word-level timestamp alignment (skipped if no model for language).
    Phase 3: Speaker diarization.

    Args:
        path:         Path to the audio file.
        device:       'cuda' or 'cpu'.
        compute_type: 'float16' (cuda) or 'float32' (cpu).
        hf_token:     HuggingFace token for the diarization model.

    Returns:
        Formatted transcript string with speaker labels and timestamps.
    """

    # Preprocessing: Denoise (anlmdn) and trim silences using ffmpeg
    with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as tmp:
        trimmed_path = Path(tmp.name)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-af", "anlmdn,silenceremove=1:0:-50dB",
        str(trimmed_path)
    ]
    logger.info(f"Denoising (anlmdn) and trimming silences with ffmpeg: {' '.join(ffmpeg_cmd)}")
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Denoising and silence trimming complete. Using processed file: {trimmed_path}")
        audio_path = trimmed_path
    except Exception as e:
        logger.error(f"Denoising/silence trimming failed, using original file: {e}")
        audio_path = path

    # Phase 0: Language detection
    language = detect_language(audio_path, device)
    align_model_name = select_align_model(language)
    skip_alignment = align_model_name == "SKIP"

    # Load full audio for transcription and alignment
    logger.info(f"Loading audio: {audio_path.name}")
    audio = whisperx.load_audio(str(audio_path))
    logger.info(
        f"Audio loaded. Samples: {audio.shape[0]}, "
        f"Duration: {audio.shape[0] / 16000:.1f}s"
    )

    # Clean up temporary trimmed file
    if audio_path != path:
        try:
            trimmed_path.unlink()
            logger.info(f"Deleted temporary trimmed file: {trimmed_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary trimmed file: {e}")

    # Phase 1: Transcription
    logger.info("Loading WhisperModel (large-v3-turbo)...")
    fw_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    logger.info("WhisperModel loaded. Starting transcription...")

    segments_generator, info = fw_model.transcribe(
        audio, beam_size=5, language=language
    )

    raw_segments: List[Dict[str, Any]] = []
    for seg in segments_generator:
        logger.info(f"[{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text.strip()}")
        raw_segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

    logger.info(f"Transcription complete. Segments: {len(raw_segments)}")
    del fw_model
    _cleanup_gpu("WhisperModel deletion")

    # Phase 2: Alignment
    if skip_alignment:
        logger.warning(
            f"Skipping alignment — no model available for language '{language}'. "
            "Using raw segment timestamps."
        )
        result: Dict[str, Any] = {"segments": raw_segments}
    else:
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
            logger.info("Alignment model loaded. Aligning...")
            result = whisperx.align(
                raw_segments, model_a, metadata, audio, device,
                return_char_alignments=False,
            )
            logger.info("Alignment complete.")
            del model_a
            _cleanup_gpu("alignment model deletion")
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            traceback.print_exc()
            logger.warning(
                "Continuing without alignment — transcript will have "
                "segment-level timestamps only."
            )
            result = {"segments": raw_segments}

    # Phase 3: Diarization
    logger.info("Loading diarization pipeline...")
    try:
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        logger.info("Diarization pipeline loaded. Running diarization...")
        diarize_segments = diarize_model(audio)
        logger.info("Diarization complete. Assigning speakers...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker assignment complete.")
        del diarize_model
        _cleanup_gpu("diarization model deletion")
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        traceback.print_exc()
        raise

    # Format output with speaker-change paragraphs
    lines: List[str] = []
    current_speaker: Optional[str] = None
    current_text: List[str] = []

    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        text    = seg["text"].strip()
        start   = seg.get("start", 0)
        end     = seg.get("end", 0)
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