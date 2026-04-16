import torch
import logging
import subprocess
import shutil

logger = logging.getLogger(__name__)

def check_gpu_ready(min_free_gb: float = 2.0) -> bool:
    """
    Check if a CUDA GPU is available, has sufficient free VRAM, and is operational.
    Logs warnings/errors for any issues. Returns True if ready, False otherwise.
    """
    if not torch.cuda.is_available():
        logger.error("No CUDA-capable GPU detected. Check your drivers and hardware.")
        return False
    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    total_gb = props.total_memory / 1024 ** 3
    logger.info(f"GPU detected: {props.name} (device {device_idx}), total VRAM: {total_gb:.2f} GB")
    # Check free VRAM using torch
    try:
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(device_idx) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(device_idx) / 1024 ** 3
        logger.info(f"VRAM: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    except Exception as e:
        logger.warning(f"Could not query VRAM via torch: {e}")
    # Check free VRAM using nvidia-smi if available
    if shutil.which("nvidia-smi"):
        try:
            smi = subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"
            ], encoding="utf-8")
            free_str, total_str = smi.strip().split(',')
            free_gb = float(free_str) / 1024
            logger.info(f"nvidia-smi: {free_gb:.2f} GB free VRAM")
            if free_gb < min_free_gb:
                logger.warning(f"Low free VRAM: {free_gb:.2f} GB. May not be enough for large models.")
        except Exception as e:
            logger.warning(f"Could not query nvidia-smi: {e}")
    # Check CUDA version
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)  # type: ignore[attr-defined]
    try:
        nvcc = subprocess.check_output(["nvcc", "--version"], encoding="utf-8") if shutil.which("nvcc") else None
    except Exception:
        nvcc = None
    logger.info(f"PyTorch CUDA version: {torch_cuda_version}")
    if nvcc:
        logger.info(f"nvcc version: {nvcc.strip().splitlines()[-1]}")
    # Try allocating a small tensor
    try:
        x = torch.empty((1, 1), device=f"cuda:{device_idx}")
        del x
        logger.info("Successfully allocated a test tensor on the GPU.")
    except Exception as e:
        logger.error(f"Failed to allocate a tensor on the GPU: {e}")
        return False
    return True