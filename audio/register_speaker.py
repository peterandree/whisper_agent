"""
CLI tool to register and manage speaker voice embeddings.

Subcommands:
  enroll       Register one speaker from a clean single-speaker audio file.
  assign       Interactively assign names to unresolved speakers from a
               .pending_speakers.json file produced after processing.
  list         List all registered speakers and sample counts.
"""
import warnings
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")

import argparse
import sys
import json
import numpy as np
import torch
from pathlib import Path


def _load_embedding_model(device: str):
    from pyannote.audio import Inference, Model
    print("Loading speaker embedding model...")
    pretrained = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    if pretrained is None:
        print("Failed to load pyannote/wespeaker-voxceleb-resnet34-LM.")
        print("Ensure you have accepted the model license at:")
        print("  https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
        sys.exit(1)
    return Inference(pretrained, window="whole", device=torch.device(device))


def _compute_embedding(
    embedding_model,
    audio: np.ndarray,
    turns: list[dict],
    sample_rate: int = 16000,
    min_duration: float = 0.5,
) -> np.ndarray:
    """Compute mean embedding across all turns for one speaker."""
    waveform = torch.tensor(audio).unsqueeze(0)
    vecs = []
    for turn in turns:
        start_s = int(turn["start"] * sample_rate)
        end_s   = int(turn["end"]   * sample_rate)
        if end_s - start_s < int(min_duration * sample_rate):
            continue
        seg = waveform[:, start_s:end_s]
        vec = embedding_model({"waveform": seg, "sample_rate": sample_rate})
        vecs.append(np.array(vec))
    if not vecs:
        raise ValueError("No usable turns found (all turns shorter than minimum duration).")
    return np.mean(vecs, axis=0)


def cmd_enroll(args) -> None:
    """Register one speaker from a clean audio file."""
    from audio.transcription_core import load_audio
    from audio.preprocessing import preprocess_audio, cleanup_temp_audio
    from audio.speaker_db import register_speaker

    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    print(f"Loading audio: {audio_path.name}")
    processed_path, _ = preprocess_audio(audio_path)
    audio = load_audio(processed_path)
    cleanup_temp_audio(audio_path, processed_path)

    embedding_model = _load_embedding_model(args.device)
    waveform  = torch.tensor(audio).unsqueeze(0)
    embedding = embedding_model({"waveform": waveform, "sample_rate": 16000})

    register_speaker(args.name, np.array(embedding), source_file=str(audio_path))
    print(f"Registered '{args.name}' from {audio_path.name}.")


def cmd_assign(args) -> None:
    """
    Interactively assign names to unresolved speakers from a
    .pending_speakers.json file written after diarization.
    """
    from audio.transcription_core import load_audio
    from audio.preprocessing import preprocess_audio, cleanup_temp_audio
    from audio.speaker_db import register_speaker, list_speakers

    pending_path = Path(args.pending)
    if not pending_path.exists():
        print(f"Pending file not found: {pending_path}")
        sys.exit(1)

    payload       = json.loads(pending_path.read_text(encoding="utf-8"))
    audio_path    = Path(payload["audio_file"])
    speaker_turns: dict[str, list[dict]] = payload["speaker_turns"]

    if not speaker_turns:
        print("No unresolved speakers in this file.")
        pending_path.unlink(missing_ok=True)
        return

    if not audio_path.exists():
        print(f"Original audio file not found: {audio_path}")
        print("Cannot compute embeddings without the source audio.")
        sys.exit(1)

    print(f"\nMeeting: {audio_path.name}")
    print(f"Unresolved speakers: {len(speaker_turns)}\n")

    registered = list_speakers()
    if registered:
        print("Already registered speakers:")
        for name, count in registered:
            print(f"  {name:<30} ({count} sample(s))")
        print()

    print("Loading audio...")
    processed_path, _ = preprocess_audio(audio_path)
    audio = load_audio(processed_path)
    cleanup_temp_audio(audio_path, processed_path)

    embedding_model = _load_embedding_model(args.device)

    for label in sorted(speaker_turns.keys()):
        turns         = speaker_turns[label]
        total_seconds = sum(t["end"] - t["start"] for t in turns)
        print(f"  {label}: {len(turns)} turn(s), {total_seconds:.0f}s total speech")
        name = input(
            f"  Enter name for {label} (or press Enter to skip): "
        ).strip()
        if not name:
            print(f"  Skipping {label}.\n")
            continue
        try:
            embedding = _compute_embedding(embedding_model, audio, turns)
            register_speaker(name, embedding, source_file=str(audio_path))
            print(f"  Registered '{name}' from {len(turns)} turn(s).\n")
        except ValueError as e:
            print(f"  Could not register {label}: {e}\n")

    pending_path.unlink(missing_ok=True)
    print("Assignment complete. Pending file removed.")
    print("These speakers will be recognized automatically in future recordings.")


def cmd_list(args) -> None:
    """List all registered speakers."""
    from audio.speaker_db import list_speakers
    speakers = list_speakers()
    if not speakers:
        print("No speakers registered yet.")
        return
    print(f"\n{'Name':<30} {'Samples':>7}")
    print("-" * 40)
    for name, count in speakers:
        print(f"{name:<30} {count:>7}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m audio.register_speaker",
        description="Manage speaker voice embeddings for automatic recognition.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for embedding model: cuda or cpu (default: cuda)"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # enroll
    p_enroll = subparsers.add_parser(
        "enroll",
        help="Register one speaker from a clean single-speaker audio file.",
    )
    p_enroll.add_argument("--name", required=True, help="Speaker full name")
    p_enroll.add_argument("--file", required=True, help="Path to audio file")
    p_enroll.set_defaults(func=cmd_enroll)

    # assign
    p_assign = subparsers.add_parser(
        "assign",
        help="Assign names to unresolved speakers from a processed meeting.",
    )
    p_assign.add_argument(
        "--pending", required=True,
        help="Path to the .pending_speakers.json file",
    )
    p_assign.set_defaults(func=cmd_assign)

    # list
    p_list = subparsers.add_parser(
        "list",
        help="List all registered speakers.",
    )
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()