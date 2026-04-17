import json
import requests
import logging
import time
from .prompts import PROMPT_TEMPLATE
from config.settings import OLLAMA_TIMEOUT

logger = logging.getLogger(__name__)

MAX_RETRIES   = 3
RETRY_BACKOFF = [2, 5, 15]

_ollama_first_request  = True
_selected_model: str | None = None

SHORT_TRANSCRIPT_CHARS = 800

SHORT_PROMPT = (
    "Summarize this meeting transcript concisely in Markdown. "
    "Include: what was discussed, any decisions made, and any action items.\n\n"
    "Transcript:\n{transcript}"
)


def _resolve_model(ollama_url: str, configured_model: str) -> str:
    return configured_model


def _call_ollama(
    prompt: str,
    ollama_url: str,
    ollama_model: str,
) -> str:
    global _ollama_first_request

    estimated_tokens = max(2048, min(16384, int(len(prompt) / 3.5)))

    payload = {
        "model":  ollama_model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_ctx": estimated_tokens,
        },
    }

    for attempt, wait in enumerate(RETRY_BACKOFF, start=1):
        try:
            timeout = OLLAMA_TIMEOUT
            if _ollama_first_request:
                timeout = max(OLLAMA_TIMEOUT, 1200)
                logger.info(
                    f"First Ollama request: using extended timeout of {timeout}s "
                    f"(model: {ollama_model}, ctx: {estimated_tokens} tokens)."
                )

            response = requests.post(
                ollama_url, json=payload, timeout=timeout, stream=True
            )
            response.raise_for_status()
            _ollama_first_request = False

            parts: list[str] = []
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    parts.append(token)
                    print(token, end="", flush=True)
                if chunk.get("done"):
                    break
            print()  # newline after stream ends
            return "".join(parts)

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {ollama_url}. "
                "Ensure Ollama is running: Start-Process ollama serve"
            )
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                logger.error(f"Ollama request failed after {MAX_RETRIES} attempts: {e}")
                raise RuntimeError(
                    f"Ollama request failed after {MAX_RETRIES} attempts: {e}"
                )
            logger.warning(
                f"Ollama request failed (attempt {attempt}/{MAX_RETRIES}): {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError("Ollama request failed: no response returned.")


def _split_transcript(transcript: str, chunk_size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(transcript):
        end = start + chunk_size
        if end < len(transcript):
            boundary = transcript.rfind("\n", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        chunks.append(transcript[start:end])
        start = end - overlap
    return chunks


def summarize(transcript: str, ollama_url: str, configured_model: str) -> str:
    """
    Summarize a transcript using the best locally available Ollama model.

    Model selection:
    - Queries Ollama for installed models at first call.
    - Ranks by parameter count vs available VRAM.
    - Prefers the configured model if it fits; otherwise picks the best fit.

    Prompt selection:
    - Short transcripts (< 800 chars) use a lightweight prompt.
    - Long transcripts use the full 7-section meeting template.
    - Very long transcripts are chunked and merged.

    Args:
        transcript:       Full transcript text.
        ollama_url:       Ollama API URL.
        configured_model: Model name from settings / env var.
    """
    CHUNK_SIZE = 12000
    OVERLAP    = 500

    model    = _resolve_model(ollama_url, configured_model)
    template = SHORT_PROMPT if len(transcript) < SHORT_TRANSCRIPT_CHARS else PROMPT_TEMPLATE

    logger.info(
        f"Summarizing with model '{model}'. "
        f"Transcript length: {len(transcript)} chars. "
        f"Prompt: {'short' if template is SHORT_PROMPT else 'full'}."
    )

    if len(transcript) <= CHUNK_SIZE:
        chunks = [transcript]
    else:
        chunks = _split_transcript(transcript, CHUNK_SIZE, OVERLAP)

    if len(chunks) == 1:
        return _call_ollama(
            template.format(transcript=chunks[0]),
            ollama_url,
            model,
        )

    logger.info(f"Summarizing in {len(chunks)} chunks...")
    partial_summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Chunk {i + 1}/{len(chunks)}...")
        partial_summaries.append(
            _call_ollama(
                template.format(transcript=chunk),
                ollama_url,
                model,
            )
        )

    logger.info("Merging partial summaries...")
    merge_prompt = (
        "You are merging multiple partial summaries of the same meeting into one "
        "complete summary.\n"
        "Combine all sections, deduplicate, and preserve all unique content. "
        "Do not drop any action items, decisions, or discussed topics.\n"
        "Produce the same section structure as the inputs.\n\n"
        "Partial summaries:\n{transcript}"
    )
    return _call_ollama(
        merge_prompt.format(transcript="\n\n---\n\n".join(partial_summaries)),
        ollama_url,
        model,
    )