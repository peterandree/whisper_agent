"""
Persistent speaker identity store.
Embeddings are 512-d float32 numpy arrays (pyannote wespeaker-resnet34).
"""
import sqlite3
import numpy as np
import logging
from pathlib import Path
from config.settings import SPEAKER_DB_PATH

logger = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    SPEAKER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(SPEAKER_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS speakers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            embedding   BLOB NOT NULL,
            source_file TEXT,
            added_at    TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    return conn


def _to_blob(embedding: np.ndarray) -> bytes:
    return embedding.astype(np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def register_speaker(name: str, embedding: np.ndarray, source_file: str = "") -> None:
    """Store a new speaker embedding. Multiple entries per name are allowed
    and averaged during lookup for robustness."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO speakers (name, embedding, source_file) VALUES (?, ?, ?)",
            (name, _to_blob(embedding), source_file),
        )
    logger.info(f"Registered speaker '{name}'.")


def query_speaker(
    embedding: np.ndarray,
    threshold: float = 0.85,
) -> tuple[str | None, float]:
    """
    Find the closest known speaker by cosine similarity.
    Returns (name, score) or (None, best_score) if below threshold.
    """
    with _connect() as conn:
        rows = conn.execute("SELECT name, embedding FROM speakers").fetchall()

    if not rows:
        return None, 0.0

    query = embedding / (np.linalg.norm(embedding) + 1e-9)

    # Group by name, average embeddings per name
    grouped: dict[str, list[np.ndarray]] = {}
    for name, blob in rows:
        grouped.setdefault(name, []).append(_from_blob(blob))

    best_name, best_score = None, -1.0
    for name, vecs in grouped.items():
        mean_vec = np.mean(vecs, axis=0)
        mean_vec /= (np.linalg.norm(mean_vec) + 1e-9)
        score = float(np.dot(query, mean_vec))
        if score > best_score:
            best_score, best_name = score, name

    if best_score >= threshold:
        return best_name, best_score
    return None, best_score


def list_speakers() -> list[tuple[str, int]]:
    """Return list of (name, sample_count) for all registered speakers."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, COUNT(*) FROM speakers GROUP BY name ORDER BY name"
        ).fetchall()
    return rows