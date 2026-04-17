"""
Worker process: diarization only.
Reads aligned segments JSON produced by transcription_worker.py,
runs pyannote diarization, writes final transcript to output file,
then exits via os._exit(0).
"""
import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import sys
import os
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    # Args: audio_path hf_token device segments_json_file output_txt_file
    audio_path         = Path(sys.argv[1])
    hf_token           = sys.argv[2]
    device             = sys.argv[3]
    segments_json_file = Path(sys.argv[4])
    output_file        = Path(sys.argv[5])

    from audio.transcription_core import load_audio
    from audio.diarization import run_diarization
    from audio.progress import format_transcript
    from audio.preprocessing import preprocess_audio, cleanup_temp_audio

    payload        = json.loads(segments_json_file.read_text(encoding="utf-8"))
    total_duration = payload["total_duration"]
    result         = {
        "segments":      payload["segments"],
        "word_segments": payload.get("word_segments", []),
    }

    # Reload audio for diarization (needs raw waveform)
    audio_path_processed, _ = preprocess_audio(audio_path)
    audio = load_audio(audio_path_processed)
    cleanup_temp_audio(audio_path, audio_path_processed)

    result, diarize_segments = run_diarization(audio, result, hf_token, device)

    transcript = format_transcript(result["segments"], total_duration)
    output_file.write_text(transcript, encoding="utf-8")

    # --- Pending registration ---
    # Collect any speaker labels that were not resolved to a known name.
    # Write a pending file so the user can assign names after the fact
    # without blocking summarization.
    unresolved: set[str] = set()
    for seg in result.get("segments", []):
        label = seg.get("speaker", "")
        if label.startswith("SPEAKER_"):
            unresolved.add(label)

    if unresolved and diarize_segments is not None:
        speaker_turns: dict[str, list[dict]] = {}
        for _, row in diarize_segments.iterrows():
            label = row["speaker"]
            if label in unresolved:
                speaker_turns.setdefault(label, []).append(
                    {"start": float(row["start"]), "end": float(row["end"])}
                )
        pending = {
            "audio_file":    str(audio_path),
            "speaker_turns": speaker_turns,
        }
        pending_file = output_file.with_suffix(".pending_speakers.json")
        pending_file.write_text(json.dumps(pending, indent=2), encoding="utf-8")
        logging.getLogger(__name__).info(
            f"{len(unresolved)} unresolved speaker(s). "
            f"Run: python -m audio.register_speaker assign --pending \"{pending_file}\""
        )

    os._exit(0)