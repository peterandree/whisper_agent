import gc
import time
import shutil
import subprocess
import logging
import torch

logger = logging.getLogger(__name__)


def log_device(label: str, device: str) -> None:
    """
    Log whether a model is actually running on GPU or CPU.
    Uses nvidia-smi for ground truth — CTranslate2 (faster-whisper) and
    pyannote bypass PyTorch's memory allocator so torch always reports
    0.00 GB even when the GPU is genuinely in use.
    """
    if device != "cuda":
        logger.info(f"{label} running on CPU.")
        return
    if not shutil.which("nvidia-smi"):
        logger.warning(f"{label} — nvidia-smi not found on PATH. Cannot verify GPU usage.")
        return
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            used, free, total = [int(x.strip()) for x in result.stdout.strip().split(",")]
            logger.info(
                f"{label} running on GPU. "
                f"VRAM: {used} MiB used / {total} MiB total ({free} MiB free)."
            )
        else:
            logger.warning(f"{label} — nvidia-smi error: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        logger.warning(f"{label} — nvidia-smi timed out.")
    except Exception as e:
        logger.warning(f"{label} — Could not query nvidia-smi: {e}")


def cleanup_gpu(label: str) -> None:
    """
    For PyTorch-native models (wav2vec2, pyannote) where PyTorch owns the
    CUDA stream and synchronize() is safe.
    """
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1.0)
    allocated = torch.cuda.memory_allocated() / 1024 ** 3
    reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
    logger.info(
        f"VRAM after {label}: {allocated:.2f} GB allocated, "
        f"{reserved:.2f} GB reserved."
    )
