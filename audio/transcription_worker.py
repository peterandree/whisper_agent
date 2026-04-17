"""
Worker process: preprocessing, language detection, transcription, alignment.
Writes aligned segments as JSON to a temp file, then exits via os._exit(0)
to avoid CTranslate2/WDDM destructor crash.
"""
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
    # Args: path hf_token device compute_type segments_json_file
    path              = Path(sys.argv[1])
    hf_token          = sys.argv[2]
    device            = sys.argv[3]
    compute_type      = sys.argv[4]
    segments_json_file = Path(sys.argv[5])

    from audio.preprocessing import preprocess_audio, cleanup_temp_audio
    from audio.language_detection import detect_language
    from audio.model_selector import select_align_model
    from audio.transcription_core import load_audio, run_transcription
    from audio.alignment import run_alignment

    audio_path, total_duration = preprocess_audio(path)
    language = detect_language(audio_path, device)
    align_model_name = select_align_model(language)
    skip_alignment = align_model_name == "SKIP"

    audio = load_audio(audio_path)
    cleanup_temp_audio(path, audio_path)

    raw_segments, _ = run_transcription(audio, language, device, compute_type, total_duration)

    if skip_alignment:
        import logging as _log
        _log.getLogger(__name__).warning(
            f"Skipping alignment — no model for language '{language}'."
        )
        result = {"segments": raw_segments}
    else:
        result = run_alignment(raw_segments, language, align_model_name, audio, device)

    # Write intermediate: segments + metadata needed by diarization worker
    payload = {
        "language":       language,
        "total_duration": total_duration,
        "segments":       result["segments"],
        "word_segments":  result.get("word_segments", []),
    }
    segments_json_file.write_text(json.dumps(payload), encoding="utf-8")

    os._exit(0)