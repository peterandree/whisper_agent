import time
import logging
import torch

logger = logging.getLogger(__name__)

_SETTLE_INTERVAL = 0.5   # seconds between polls
_SETTLE_COUNT    = 3     # consecutive identical readings required
_SETTLE_TIMEOUT  = 30.0  # give up after this many seconds and use last reading
_TOLERANCE_MB    = 50    # readings within this many MB are considered identical


def get_free_vram_gb() -> float:
    """
    Returns free VRAM in GB on the primary CUDA device, waiting for the
    value to stabilize before returning. This handles the case where a
    previous process (e.g. Ollama, whisper model) just finished and VRAM
    is still being released by the OS.

    Returns a large number on CPU so all models are always eligible.
    """
    if not torch.cuda.is_available():
        return 999.0

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    last_free_mb  = -1
    stable        = 0
    elapsed       = 0.0

    while elapsed < _SETTLE_TIMEOUT:
        free_bytes, _ = torch.cuda.mem_get_info(device=0)
        free_mb = free_bytes / (1024 ** 2)

        if abs(free_mb - last_free_mb) <= _TOLERANCE_MB:
            stable += 1
            if stable >= _SETTLE_COUNT:
                free_gb = free_mb / 1024
                logger.info(f"VRAM stable at {free_gb:.2f} GB free (settled after {elapsed:.1f}s)")
                return free_gb
        else:
            if last_free_mb >= 0:
                logger.debug(
                    f"VRAM still settling: {free_mb:.0f} MB "
                    f"(was {last_free_mb:.0f} MB, delta {abs(free_mb - last_free_mb):.0f} MB)"
                )
            stable = 0

        last_free_mb = free_mb
        time.sleep(_SETTLE_INTERVAL)
        elapsed += _SETTLE_INTERVAL

    # Timeout — use whatever the last reading was
    free_gb = last_free_mb / 1024
    logger.warning(
        f"VRAM did not stabilize within {_SETTLE_TIMEOUT:.0f}s. "
        f"Using last reading: {free_gb:.2f} GB free."
    )
    return free_gb