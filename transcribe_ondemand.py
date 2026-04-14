import os
import shutil

_ffmpeg_exe = shutil.which("ffmpeg")
if _ffmpeg_exe:
    os.add_dll_directory(os.path.dirname(_ffmpeg_exe))

import gc
import sys
import time
import requests
import whisperx
import warnings
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import torch
from faster_whisper import WhisperModel
from datetime import datetime

warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")


WATCH_DIR = Path("C:/data/meetings/raw")
OUTPUT_DIR = Path("C:/data/meetings/summaries")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".mkv", ".m4a", ".flac"}


PROMPT_TEMPLATE = """You are an expert meeting analyst. Analyze the following meeting transcript and produce a structured summary.

Output the following sections in Markdown:

## Meeting Overview
One paragraph (3-5 sentences) describing the meeting's purpose, participants if mentioned, and overall outcome.

## Key Decisions
Bullet list of every concrete decision made. Each bullet must be a complete, standalone sentence. Omit discussion and deliberation — only final decisions.

## Action Items
Table with columns: | Action | Owner | Deadline |
Extract every committed task, assigned owner (write "Unassigned" if unclear), and deadline (write "No deadline" if not mentioned).

## Open Questions
Bullet list of unresolved questions or topics explicitly deferred to a future meeting.

## Technical Details
Bullet list of any specific technical information: version numbers, architecture decisions, system names, metrics, thresholds, configurations. Omit this section entirely if none are present.

---
Rules:
- Ignore greetings, small talk, and filler phrases entirely.
- Do not infer or invent information not explicitly stated in the transcript.
- If a section has no content, write "None" under its heading — do not omit the heading.
- Use professional, concise language. No padding.

Transcript:
{transcript}"""




import whisperx
import gc
import torch
import os

def transcribe(path: Path) -> str:
    device = "cuda"
    compute_type = "float16"
    hf_token = os.environ["HF_TOKEN"]

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
    diarize_model = whisperx.diarize.DiarizationPipeline(
        token=hf_token, device=device
    )
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
def summarize(transcript: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": PROMPT_TEMPLATE.format(transcript=transcript),
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()["response"]


def process(path: Path):
    print(f"[+] Processing: {path.name}")
    try:
        transcript = transcribe(path)
        print(f"[+] Transcription done ({len(transcript)} chars)")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{path.stem}_{ts}"

        transcript_file = OUTPUT_DIR / (stem + ".transcript.txt")
        transcript_file.write_text(transcript, encoding="utf-8")
        print(f"[+] Transcript written to: {transcript_file}")

        summary = summarize(transcript)
        out_file = OUTPUT_DIR / (stem + ".md")
        out_file.write_text(summary, encoding="utf-8")
        print(f"[+] Summary written to: {out_file}")
    except Exception as e:
        print(f"[!] Failed to process {path.name}: {e}", file=sys.stderr)


class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        # Wait for OBS to finish writing
        time.sleep(3)
        process(path)


if __name__ == "__main__":
    print(f"[*] Watching {WATCH_DIR} ...")
    observer = Observer()
    observer.schedule(AudioHandler(), str(WATCH_DIR), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()