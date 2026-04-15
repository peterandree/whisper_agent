
import requests
import logging
import time
from .prompts import PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 15]  # seconds

def _call_ollama(prompt: str, ollama_url: str, ollama_model: str) -> str:
    """
    Call the Ollama API to generate a summary or merge summaries.

    Args:
        prompt (str): The prompt to send to Ollama.
        ollama_url (str): The URL of the Ollama API.
        ollama_model (str): The model to use.

    Returns:
        str: The response from the Ollama API.
    """
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 16384
        }
    }
    for attempt, wait in enumerate(RETRY_BACKOFF, start=1):
        try:
            response = requests.post(ollama_url, json=payload, timeout=600)
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(f"Ollama request failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {wait}s...")
            time.sleep(wait)


def summarize(transcript: str, ollama_url: str, ollama_model: str) -> str:
    """
    Summarize a transcript using the Ollama API, chunking if necessary.

    Args:
        transcript (str): The transcript to summarize.
        ollama_url (str): The URL of the Ollama API.
        ollama_model (str): The model to use.

    Returns:
        str: The summary of the transcript.
    """
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
        partial_summaries = [_call_ollama(PROMPT_TEMPLATE.format(transcript=chunks[0]), ollama_url, ollama_model)]
    else:
        logger.info(f"Summarizing in {len(chunks)} chunks...")
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1}/{len(chunks)}...")
            partial_summaries.append(_call_ollama(PROMPT_TEMPLATE.format(transcript=chunk), ollama_url, ollama_model))

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    logger.info("Merging partial summaries...")
    merge_prompt = (
        "You are merging multiple partial summaries of the same meeting into one complete summary.\n"
        "Combine all sections, deduplicate, and preserve all unique content. Do not drop any action items, decisions, or discussed topics.\n"
        "Produce the same section structure as the inputs.\n\nPartial summaries:\n{transcript}"
    )
    return _call_ollama(merge_prompt.format(transcript="\n\n---\n\n".join(partial_summaries)), ollama_url, ollama_model)
