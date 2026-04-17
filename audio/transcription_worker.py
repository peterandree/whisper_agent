"""
Runs as a child process. Imports and executes the transcription pipeline,
then exits — letting the OS release all CUDA/CTranslate2 resources cleanly
instead of manually tearing down models inside a long-lived parent process.
"""
import sys
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    # Args: path hf_token device compute_type output_file
    path         = Path(sys.argv[1])
    hf_token     = sys.argv[2]
    device       = sys.argv[3]
    compute_type = sys.argv[4]
    output_file  = Path(sys.argv[5])

    from audio.transcription import transcribe
    transcript = transcribe(path, device, compute_type, hf_token)
    output_file.write_text(transcript, encoding="utf-8")
    # os._exit() bypasses all Python atexit handlers, __del__ methods,
    # and C++ destructors (CTranslate2, CUDA driver teardown).
    # This prevents the 0xC0000409 stack corruption crash that occurs
    # when CTranslate2 tears down its CUDA context on Windows WDDM.
    os._exit(0)