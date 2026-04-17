import logging
from typing import Optional

from audio.audio_utils import format_seconds

logger = logging.getLogger(__name__)


def format_transcript(segments: list[dict], total_duration: float) -> str:
    """
    Format diarized segments into a readable transcript with
    speaker-change paragraphs and progress/timestamp prefixes in the log.
    """
    lines: list[str] = []
    current_speaker: Optional[str] = None
    current_text: list[str] = []

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text    = seg["text"].strip()
        start   = seg.get("start", 0)
        end     = seg.get("end", 0)

        if total_duration > 0:
            percent  = min(100.0, 100.0 * end / total_duration)
            progress = f"[{percent:5.1f}%] "
        else:
            progress = ""
        t_start = format_seconds(start)
        t_end   = format_seconds(end)
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