import requests
from .prompts import PROMPT_TEMPLATE

def _call_ollama(prompt: str, ollama_url: str, ollama_model: str) -> str:
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 16384
        }
    }
    response = requests.post(ollama_url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["response"]

def summarize(transcript: str, ollama_url: str, ollama_model: str) -> str:
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
        print(f"[+] Summarizing in {len(chunks)} chunks...")
        partial_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  [+] Chunk {i+1}/{len(chunks)}...")
            partial_summaries.append(_call_ollama(PROMPT_TEMPLATE.format(transcript=chunk), ollama_url, ollama_model))

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    print("[+] Merging partial summaries...")
    merge_prompt = """You are merging multiple partial summaries of the same meeting into one complete summary.\nCombine all sections, deduplicate, and preserve all unique content. Do not drop any action items, decisions, or discussed topics.\nProduce the same section structure as the inputs.\n\nPartial summaries:\n{transcript}"""
    return _call_ollama(merge_prompt.format(transcript="\n\n---\n\n".join(partial_summaries)), ollama_url, ollama_model)
