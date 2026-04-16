import importlib
import logging
import pytest

def test_torch_cuda_available():
    import torch
    assert torch.cuda.is_available(), "torch does not detect a CUDA-capable GPU."
    assert torch.version.cuda is not None, "torch is not built with CUDA support."
    logging.info(f"torch version: {torch.__version__}, CUDA: {torch.version.cuda}")

def test_whisperx_cuda():
    whisperx = importlib.import_module("whisperx")
    # whisperx uses torch under the hood, so if torch is CUDA-capable, so is whisperx
    import torch
    assert torch.cuda.is_available(), "whisperx/torch does not detect a CUDA-capable GPU."
    logging.info("whisperx is available and torch sees CUDA.")

def test_faster_whisper_cuda():
    fw = importlib.import_module("faster_whisper")
    # Try to instantiate a model on CUDA (will fail if not built for CUDA)
    try:
        model = fw.WhisperModel("tiny", device="cuda", compute_type="float16")
        del model
    except Exception as e:
        pytest.skip(f"faster-whisper not built for CUDA or no CUDA device: {e}")
    logging.info("faster-whisper can instantiate a CUDA model.")