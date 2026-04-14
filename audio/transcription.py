import os
import gc
import torch
import whisperx
from pathlib import Path
from faster_whisper import WhisperModel
from whisperx.diarize import DiarizationPipeline


def transcribe(path: Path, device: str, compute_type: str, hf_token: str) -> str:
    audio = whisperx.load_audio(str(path))

    # Phase 1: Streaming transcription via faster-whisper generator
    print("[+] Loading model and transcribing...")
    fw_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    segments_generator, info = fw_model.transcribe(audio, beam_size=5)
    print(f"[+] Detected language: {info.language}")

    raw_segments = []
    for seg in segments_generator:
        print(f"  [{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text.strip()}", flush=True)
        raw_segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

    del fw_model; gc.collect(); torch.cuda.empty_cache()

    # Phase 2: Align word-level timestamps
    print("[+] Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(
        language_code=info.language, device=device
    )
    result = whisperx.align(
        raw_segments, model_a, metadata, audio, device,
        return_char_alignments=False
    )
    del model_a; gc.collect(); torch.cuda.empty_cache()

    # Phase 3: Diarize
    print("[+] Running speaker diarization...")
    diarize_model = DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Format with speaker-change paragraphs
    lines = []
    current_speaker = None
    current_text = []

    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        print(f"  [{start:6.1f}s -> {end:6.1f}s] {speaker}: {text}")

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
