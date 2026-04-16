from audio.audio_utils import get_audio_duration, format_seconds
from audio.gpu_check import check_gpu_ready
import shutil
import datetime
import os
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

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


def _log_device(label: str, device: str) -> None:
    """Log whether a model is actually using the GPU or silently fell back to CPU."""
    if device != "cuda":
        logger.info(f"{label} running on CPU.")
        return
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    if allocated < 0.1:
        logger.warning(
            f"{label} — VRAM shows {allocated:.2f} GB allocated. "
            "Model may have silently fallen back to CPU. "
            "Check that no other process holds the CUDA context."
        )
    else:
        logger.info(
            f"{label} running on GPU. "
            f"VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved."
        )

    # Log total/free VRAM using torch (if available)
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"Total VRAM: {total:.2f} GB")
        try:
            free = torch.cuda.memory_stats(0)["reserved_bytes.all.current"] / 1024 ** 3
            logger.info(f"Free VRAM (reserved): {total - free:.2f} GB")
        except Exception:
            pass
    # Log nvidia-smi output if available
    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.free,memory.used", "--format=csv,nounits,noheader"], encoding="utf-8")
            logger.info(f"nvidia-smi: {smi.strip()}")
        except Exception as e:
            logger.warning(f"Could not query nvidia-smi: {e}")

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
        # Get total duration for progress tracking
        total_duration = get_audio_duration(audio_path)
        if total_duration > 0:
            logger.info(f"Total audio duration: {format_seconds(total_duration)} ({total_duration:.1f} seconds)")
        else:
            logger.warning("Could not determine total audio duration. Progress percentage will not be shown.")
    start_time = datetime.datetime.now()
    logger.info(f"Transcription pipeline started at {start_time:%Y-%m-%d %H:%M:%S}")
    _log_device("Startup", device)
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
    try:
        with tempfile.NamedTemporaryFile(suffix=path.suffix, delete=False) as tmp:
            trimmed_path = Path(tmp.name)
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(path),
            "-af", "anlmdn,silenceremove=1:0:-50dB",
            str(trimmed_path)
        ]
        logger.info(f"Denoising (anlmdn) and trimming silences with ffmpeg: {' '.join(ffmpeg_cmd)}")
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Denoising and silence trimming complete. Using processed file: {trimmed_path}")
        audio_path = trimmed_path
    except Exception as e:
        logger.error(f"Audio preprocessing with ffmpeg failed: {e}\nCheck that ffmpeg is installed and the input file is a valid audio file. Using original file.")
        audio_path = path

    # Phase 0: Language detection
    phase0_start = datetime.datetime.now()
    try:
        language = detect_language(audio_path, device)
        align_model_name = select_align_model(language)
        skip_alignment = align_model_name == "SKIP"
    except RuntimeError as e:
        logger.error(f"Language detection failed: {e}")
        raise
    logger.info(f"Phase 0 (language detection) took {(datetime.datetime.now() - phase0_start).total_seconds():.2f}s")

    # Load full audio for transcription and alignment
    try:
        logger.info(f"Loading audio: {audio_path.name}")
        audio = whisperx.load_audio(str(audio_path))
        logger.info(
            f"Audio loaded. Samples: {audio.shape[0]}, "
            f"Duration: {audio.shape[0] / 16000:.1f}s"
        )
    except Exception as e:
        logger.error(f"Failed to load audio file: {e}\nCheck that the file exists and is a supported audio format.")
        raise

    # Clean up temporary trimmed file
    if audio_path != path:
        try:
            trimmed_path.unlink()
            logger.info(f"Deleted temporary trimmed file: {trimmed_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary trimmed file: {e}")

    # Phase 1: Transcription
    phase1_start = datetime.datetime.now()
    logger.info("Loading WhisperModel (large-v3-turbo)...")
    try:
        if device.startswith("cuda"):
            if not check_gpu_ready():
                logger.error("GPU is not ready for WhisperModel. Aborting transcription phase.")
                raise RuntimeError("GPU not ready for WhisperModel.")
        try:
            fw_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
        except Exception as e:
            logger.error(f"Failed to load WhisperModel: {e}\nCheck that the model files are present and compatible with your hardware.")
            raise
        _log_device("WhisperModel (large-v3-turbo)", device)
        logger.info("WhisperModel loaded. Starting transcription...")
        segments_generator, info = fw_model.transcribe(
            audio, beam_size=5, language=language
        )
        raw_segments: List[Dict[str, Any]] = []
        for seg in segments_generator:
            seg_start = seg.start
            seg_end = seg.end
            seg_text = seg.text.strip()
            # Progress percentage
            if total_duration > 0:
                percent = min(100, 100 * seg_end / total_duration)
                progress = f"[{percent:5.1f}%] "
            else:
                progress = ""
            # Human-friendly time
            t_start = format_seconds(seg_start)
            t_end = format_seconds(seg_end)
            logger.info(f"{progress}[{t_start} -> {t_end}] {seg_text}")
            raw_segments.append({"start": seg_start, "end": seg_end, "text": seg.text})
        logger.info(f"Transcription complete. Segments: {len(raw_segments)}")
        logger.info(f"Segment collection done. About to delete WhisperModel. VRAM allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        del fw_model
        _cleanup_gpu("WhisperModel deletion")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during transcription. Try reducing batch size, using a smaller model, or switching to CPU.")
        raise
    except Exception as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during transcription. Try reducing batch size, using a smaller model, or switching to CPU.")
        logger.error(f"Transcription failed: {e}")
        raise
    logger.info(f"Phase 1 (transcription) took {(datetime.datetime.now() - phase1_start).total_seconds():.2f}s")

    # Phase 2: Alignment
    phase2_start = datetime.datetime.now()
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
            if device.startswith("cuda"):
                if not check_gpu_ready():
                    logger.error("GPU is not ready for alignment model. Skipping alignment phase.")
                    raise RuntimeError("GPU not ready for alignment model.")
            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=language,
                    device=device,
                    model_name=align_model_name,
                )
            except Exception as e:
                logger.error(f"Failed to load alignment model: {e}\nCheck that the alignment model files are present and compatible with your hardware.")
                raise
            _log_device(f"Alignment model ({align_model_name or 'whisperx built-in'})", device)
            logger.info("Alignment model loaded. Aligning...")
            result = whisperx.align(
                raw_segments, model_a, metadata, audio, device,
                return_char_alignments=False,
            )
            logger.info("Alignment complete.")
            del model_a
            _cleanup_gpu("alignment model deletion")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA OOM during alignment. Try reducing batch size, using a smaller model, or switching to CPU.")
            logger.error(f"Alignment failed: {e}")
            traceback.print_exc()
            logger.warning(
                "Continuing without alignment — transcript will have "
                "segment-level timestamps only."
            )
            result = {"segments": raw_segments}
        except Exception as e:
            if "CUDA out of memory" in str(e):
                logger.error("CUDA OOM during alignment. Try reducing batch size, using a smaller model, or switching to CPU.")
            logger.error(f"Alignment failed: {e}")
            traceback.print_exc()
            logger.warning(
                "Continuing without alignment — transcript will have "
                "segment-level timestamps only."
            )
            result = {"segments": raw_segments}
    logger.info(f"Phase 2 (alignment) took {(datetime.datetime.now() - phase2_start).total_seconds():.2f}s")

    # Phase 3: Diarization
    phase3_start = datetime.datetime.now()
    logger.info("Loading diarization pipeline...")
    try:
        if device.startswith("cuda"):
            if not check_gpu_ready():
                logger.error("GPU is not ready for DiarizationPipeline. Aborting diarization phase.")
                raise RuntimeError("GPU not ready for DiarizationPipeline.")
        try:
            diarize_model = DiarizationPipeline(token=hf_token, device=device)
        except Exception as e:
            logger.error(f"Failed to load DiarizationPipeline: {e}\nCheck that the diarization model files are present and compatible with your hardware.")
            raise
        _log_device("DiarizationPipeline", device)
        logger.info("Diarization pipeline loaded. Running diarization...")
        diarize_segments = diarize_model(audio)
        logger.info("Diarization complete. Assigning speakers...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        logger.info("Speaker assignment complete.")
        del diarize_model
        _cleanup_gpu("diarization model deletion")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during diarization. Try reducing batch size, using a smaller model, or switching to CPU.")
        logger.error(f"Diarization failed: {e}")
        traceback.print_exc()
        raise
    except Exception as e:
        if "CUDA out of memory" in str(e):
            logger.error("CUDA OOM during diarization. Try reducing batch size, using a smaller model, or switching to CPU.")
        logger.error(f"Diarization failed: {e}")
        traceback.print_exc()
        raise
    logger.info(f"Phase 3 (diarization) took {(datetime.datetime.now() - phase3_start).total_seconds():.2f}s")
    total_time = (datetime.datetime.now() - start_time).total_seconds()
    logger.info(f"Transcription pipeline finished. Total time: {total_time:.2f}s")

    # Format output with speaker-change paragraphs
    lines: List[str] = []
    current_speaker: Optional[str] = None
    current_text: List[str] = []

    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        text    = seg["text"].strip()
        start   = seg.get("start", 0)
        end     = seg.get("end", 0)
        # Progress percentage for diarization output
        if total_duration > 0:
            percent = min(100, 100 * end / total_duration)
            progress = f"[{percent:5.1f}%] "
        else:
            progress = ""
        t_start = format_seconds(start)
        t_end = format_seconds(end)
        logger.info(f"{progress}[{t_start} -> {t_end}] {speaker}: {text}")

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