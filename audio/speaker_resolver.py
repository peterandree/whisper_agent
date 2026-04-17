"""
Extracts per-speaker mean embeddings from diarization output and resolves
SPEAKER_XX labels to known names via speaker_db.
Uses pyannote/wespeaker-voxceleb-resnet34-LM for embedding extraction.
"""
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

RESOLUTION_THRESHOLD = 0.85


def resolve_speakers(
    result: dict,
    diarize_segments: Any,   # pandas DataFrame from DiarizationPipeline
    audio: np.ndarray,
    device: str,
    source_file: str = "",
) -> dict:
    """
    For each anonymous speaker label in result['segments']:
      1. Collect all audio turns for that speaker from diarize_segments.
      2. Compute mean embedding via wespeaker-resnet34.
      3. Query speaker_db — if match above threshold, replace label with name.
      4. If unmatched, keep SPEAKER_XX and log for manual registration.
    """
    from audio.speaker_db import query_speaker

    try:
        embeddings = _extract_embeddings(diarize_segments, audio, device)
    except Exception as e:
        logger.warning(f"Embedding extraction failed: {e}. Skipping speaker resolution.")
        return result

    label_map: dict[str, str] = {}

    for speaker_label, embedding in embeddings.items():
        name, score = query_speaker(embedding, threshold=RESOLUTION_THRESHOLD)
        if name:
            logger.info(
                f"Resolved {speaker_label} → '{name}' (cosine similarity: {score:.3f})"
            )
            label_map[speaker_label] = name
        else:
            logger.info(
                f"Unknown speaker {speaker_label} (best match score: {score:.3f}). "
                "Register via: python -m audio.register_speaker"
            )

    if not label_map:
        return result

    for seg in result.get("segments", []):
        original = seg.get("speaker", "")
        if original in label_map:
            seg["speaker"] = label_map[original]
        for word in seg.get("words", []):
            if word.get("speaker", "") in label_map:
                word["speaker"] = label_map[word["speaker"]]

    return result


def _extract_embeddings(
    diarize_segments: Any,   # pandas DataFrame with columns: start, end, speaker
    audio: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    """
    Load wespeaker-resnet34, compute one mean embedding per unique speaker
    using their turn windows from diarize_segments DataFrame.
    """
    import torch
    from pyannote.audio import Inference, Model

    SAMPLE_RATE = 16000
    MIN_TURN_SAMPLES = int(0.5 * SAMPLE_RATE)  # skip turns shorter than 0.5s

    logger.info("Loading speaker embedding model for identity resolution...")
    pretrained = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    if pretrained is None:
        raise RuntimeError("Failed to load pyannote/wespeaker-voxceleb-resnet34-LM.")

    embedding_model = Inference(
        pretrained,
        window="whole",
        device=torch.device(device),
    )
    logger.info("Embedding model loaded.")

    waveform = torch.tensor(audio).unsqueeze(0)  # (1, samples)
    speaker_vecs: dict[str, list[np.ndarray]] = {}

    for _, row in diarize_segments.iterrows():
        speaker    = row["speaker"]
        start_s    = int(row["start"] * SAMPLE_RATE)
        end_s      = int(row["end"]   * SAMPLE_RATE)
        if end_s - start_s < MIN_TURN_SAMPLES:
            continue
        segment_audio = waveform[:, start_s:end_s]
        vec = embedding_model({"waveform": segment_audio, "sample_rate": SAMPLE_RATE})
        speaker_vecs.setdefault(speaker, []).append(np.array(vec))

    del embedding_model
    del pretrained
    logger.info("Embedding model unloaded.")

    return {
        speaker: np.mean(vecs, axis=0)
        for speaker, vecs in speaker_vecs.items()
        if vecs
    }