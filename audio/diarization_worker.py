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

    result = run_diarization(audio, result, hf_token, device)

    transcript = format_transcript(result["segments"], total_duration)
    output_file.write_text(transcript, encoding="utf-8")

    os._exit(0)