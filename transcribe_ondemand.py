
import os
import shutil
import logging

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
from whisperx.diarize import DiarizationPipeline

warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")


WATCH_DIR = Path("C:/data/meetings/raw")
OUTPUT_DIR = Path("C:/data/meetings/summaries")
OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "gpt-oss:20b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".mp4", ".mkv", ".m4a", ".flac"}


PROMPT_TEMPLATE = """You are an expert meeting analyst. Your job is to produce a comprehensive, detailed summary that captures everything of substance from the transcript. Completeness is the priority — do not omit topics, arguments, context, or conclusions, even if they seem minor.

Output the following sections in Markdown:

## Meeting Overview
2-4 sentences: meeting purpose, attendees if mentioned, and overall outcome.

## Topics Discussed
For each distinct topic or agenda item covered in the meeting, write a subsection:
### [Topic Name]
- What was discussed, including the arguments, positions, data, and context raised by participants
- Any conclusions or interim decisions reached on this topic
- Any disagreements or differing viewpoints expressed
Be thorough. Each topic should have 3-8 bullet points capturing the substance of what was said.

## Key Decisions
Every concrete decision made. Each bullet must be a complete, standalone sentence including the rationale if one was stated.

## Action Items
| Action | Owner | Deadline | Context |
|--------|-------|----------|---------|
Extract every committed task. Add a brief Context column explaining why the task was assigned.

## Open Questions
Unresolved questions or topics explicitly deferred. Include who raised the question and why it was not resolved if stated.

## Key Numbers & Facts
Any specific figures mentioned: budgets, timelines, metrics, percentages, version numbers, dates, headcounts, thresholds. One fact per bullet with context.

## Technical Details
Architecture decisions, system names, stack choices, configurations. Omit section if none present.

---
Rules:
- Do NOT summarize away substance. If someone made an argument, capture the argument.
- Do NOT infer or invent anything not in the transcript.
- If a section has no content, write "None" under its heading.
- Cover the entire transcript from start to finish — do not tail off toward the end.

Transcript:
{transcript}"""




import whisperx
import gc
import torch
import os

def transcribe(path: Path) -> str:
    """
    Transcribe an audio file using WhisperModel and return the transcript.

    Args:
        path (Path): Path to the audio file.

    Returns:
        str: The transcript of the audio file.
    """
    logger = logging.getLogger(__name__)
    device = "cuda"
    compute_type = "float16"
    hf_token = os.environ["HF_TOKEN"]

    audio = whisperx.load_audio(str(path))

    # Phase 1: Streaming transcription via faster-whisper generator
    logger.info("Loading model and transcribing...")
    fw_model = WhisperModel("large-v3-turbo", device=device, compute_type=compute_type)
    segments_generator, info = fw_model.transcribe(audio, beam_size=5)
    logger.info(f"Detected language: {info.language}")

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

def summarize(transcript: str) -> str:
    # Chunk if transcript is very long (~12k chars ≈ ~60 min meeting)
    CHUNK_SIZE = 12000
    OVERLAP = 500

    if len(transcript) <= CHUNK_SIZE:
        chunks = [transcript]
    else:
        chunks = []
        start = 0
        while start < len(transcript):
            end = start + CHUNK_SIZE
            chunks.append(transcript[start:end])
            start = end - OVERLAP

    if len(chunks) == 1:
        partial_summaries = [_call_ollama(PROMPT_TEMPLATE.format(transcript=chunks[0]))]
    else:
        print(f"[+] Summarizing in {len(chunks)} chunks...")
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  [+] Chunk {i+1}/{len(chunks)}...")
            partial_summaries.append(_call_ollama(PROMPT_TEMPLATE.format(transcript=chunk)))

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    # Merge pass
    print("[+] Merging partial summaries...")
    merge_prompt = """You are merging multiple partial summaries of the same meeting into one complete summary.
Combine all sections, deduplicate, and preserve all unique content. Do not drop any action items, decisions, or discussed topics.
Produce the same section structure as the inputs.

Partial summaries:
{transcript}"""
    return _call_ollama(merge_prompt.format(transcript="\n\n---\n\n".join(partial_summaries)))

def _call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 16384  #  32768 = 32k covers ~2-hour meetings with headroom
        }
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=600)
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
        import traceback
        traceback.print_exc()
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