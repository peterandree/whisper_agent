import logging
import whisperx
from pathlib import Path
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

_detector: WhisperModel | None = None
_SAMPLE_RATE = 16000
_DETECTION_SECONDS = 30


def detect_language(audio_path: Path, device: str) -> str:
    """
    Detect the spoken language of an audio file using a fast tiny Whisper model.
    Only the first 30 seconds of audio are used to keep detection cheap.

    Args:
        audio_path: Path to the audio file.
        device: 'cuda' or 'cpu'.

    Returns:
        ISO 639-1 language code, e.g. 'de', 'en'.
    """
    global _detector
    if _detector is None:
        logger.info("Loading language detection model (tiny)...")
        _detector = WhisperModel("tiny", device=device, compute_type="int8")
        logger.info("Language detection model loaded.")

    audio = whisperx.load_audio(str(audio_path))
    audio_slice = audio[:_SAMPLE_RATE * _DETECTION_SECONDS]

    _, info = _detector.transcribe(audio_slice, beam_size=1)
    logger.info(
        f"Detected language: '{info.language}' "
        f"(confidence: {info.language_probability:.0%})"
    )
    return info.language