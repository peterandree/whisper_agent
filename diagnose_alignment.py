import whisperx
import torch
import gc
import os
import sys
from pathlib import Path
from faster_whisper import WhisperModel
from whisperx.diarize import DiarizationPipeline
import faulthandler

faulthandler.enable()

AUDIO_PATH = r"C:\Users\h3hqwt\Music\meetily-recordings\Meeting 2026-04-14_14-32-10_2026-04-14_12-32\audio.mp4"  # Set your large file path here
DEVICE = "cpu"  # or "cuda" if you want to try GPU
COMPUTE_TYPE = "float32" if DEVICE == "cpu" else "float16"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"[DIAG] Loading audio: {AUDIO_PATH}")
audio = whisperx.load_audio(AUDIO_PATH)
print(f"[DIAG] Audio loaded. Type: {type(audio)}, Shape: {getattr(audio, 'shape', 'unknown')}")

print("[DIAG] Loading WhisperModel...")
fw_model = WhisperModel("large-v3-turbo", device=DEVICE, compute_type=COMPUTE_TYPE)
segments_generator, info = fw_model.transcribe(audio, beam_size=5)
print(f"[DIAG] Detected language: {info.language}")

raw_segments = []
for seg in segments_generator:
    print(f"  [{seg.start:6.1f}s -> {seg.end:6.1f}s] {seg.text.strip()}", flush=True)
    raw_segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

print(f"[DIAG] Segments: {len(raw_segments)}")
del fw_model; gc.collect(); torch.cuda.empty_cache()
print("[DIAG] WhisperModel deleted and GPU cache cleared.")

print("[DIAG] Loading align model...")
model_a, metadata = whisperx.load_align_model(language_code=info.language, device=DEVICE)
print("[DIAG] Align model loaded.")

print("[DIAG] Running alignment...")
result = whisperx.align(raw_segments, model_a, metadata, audio, DEVICE, return_char_alignments=False)
print("[DIAG] Alignment complete.")
del model_a; gc.collect(); torch.cuda.empty_cache()
print("[DIAG] Align model deleted and GPU cache cleared.")

print("[DIAG] Alignment result keys:", result.keys())
print("[DIAG] Alignment result segments count:", len(result.get('segments', [])))

# Diarization step
print("[DIAG] Running diarization...")
try:
    diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)
    print("[DIAG] DiarizationPipeline loaded.")
    diarize_segments = diarize_model(audio)
    print(f"[DIAG] Diarization complete. Segments: {len(diarize_segments)}")
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print("[DIAG] Speaker assignment complete.")
    print(f"[DIAG] Result keys after speaker assignment: {result.keys()}")
    print(f"[DIAG] Segments after speaker assignment: {len(result.get('segments', []))}")
except Exception as e:
    print(f"[DIAG] Diarization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("[DIAG] Done.")
