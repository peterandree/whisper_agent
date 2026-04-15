import gc
import torch
import whisperx
from pathlib import Path
from faster_whisper import WhisperModel
from whisperx.diarize import DiarizationPipeline


def transcribe(path: Path, device: str, compute_type: str, hf_token: str) -> str:

    print(f"[DEBUG] Loading audio from: {path}")
    audio = whisperx.load_audio(str(path))
    print(f"[DEBUG] Audio loaded. Type: {type(audio)}, Length: {getattr(audio, 'shape', 'unknown')}")

    # Phase 1: Streaming transcription via faster-whisper generator
    print("[+] Loading model and transcribing...")
    fw_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    print("[DEBUG] WhisperModel loaded.")
    segments_generator, info = fw_model.transcribe(audio, beam_size=5)
    print(f"[+] Detected language: {info.language}")

    raw_segments = []
    for seg in segments_generator:
        print(f"  [{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text.strip()}", flush=True)
        raw_segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

    print(f"[DEBUG] Transcription complete. Segments: {len(raw_segments)}")
    del fw_model; gc.collect(); torch.cuda.empty_cache()
    print("[DEBUG] WhisperModel deleted and GPU cache cleared.")

    # Phase 2: Align word-level timestamps
    print(f"[+] Aligning timestamps... (segments: {len(raw_segments)}, audio type: {type(audio)}, device: {device})")
    try:
        print("[DEBUG] Loading align model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=info.language, device=device
        )
        print("[DEBUG] Align model loaded.")
        result = whisperx.align(
            raw_segments, model_a, metadata, audio, device,
            return_char_alignments=False
        )
        print("[DEBUG] Alignment complete.")
        del model_a; gc.collect(); torch.cuda.empty_cache()
        print("[DEBUG] Align model deleted and GPU cache cleared.")
    except Exception as e:
        print(f"[!] Alignment failed: {e}")
        import traceback
        traceback.print_exc()
        raise


    # Phase 3: Diarize
    print("[+] Running speaker diarization...")
    try:
        diarize_model = DiarizationPipeline(token=hf_token, device=device)
        print("[DEBUG] DiarizationPipeline loaded.")
        diarize_segments = diarize_model(audio)
        print("[DEBUG] Diarization complete.")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print("[DEBUG] Speaker assignment complete.")
    except Exception as e:
        print(f"[!] Diarization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

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
