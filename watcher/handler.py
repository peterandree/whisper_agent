import time
import logging
from pathlib import Path
from typing import Callable
from concurrent.futures import ThreadPoolExecutor
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from config.settings import AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)


def _normalize_src_path(src_path) -> str:
    if isinstance(src_path, str):
        return src_path
    if isinstance(src_path, (bytes, bytearray)):
        return src_path.decode()
    if isinstance(src_path, memoryview):
        return src_path.tobytes().decode()
    raise TypeError(f"Unsupported src_path type: {type(src_path)}")


def _wait_until_stable(
    path: Path,
    interval: float = 1.0,
    stable_count: int = 5,
    timeout: float = 300.0,
) -> bool:
    """
    Poll file size until it stops growing for stable_count consecutive checks.

    Returns True if the file stabilized, False if timeout was reached.
    """
    last_size, stable, elapsed = -1, 0, 0.0
    while stable < stable_count:
        if elapsed >= timeout:
            logger.warning(
                f"File did not stabilize within {timeout:.0f}s: {path.name}. "
                "Processing anyway."
            )
            return False
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            stable = 0
            last_size = -1
            time.sleep(interval)
            elapsed += interval
            continue
        if size == last_size and size > 0:
            stable += 1
        else:
            stable = 0
        last_size = size
        time.sleep(interval)
        elapsed += interval
    logger.info(f"File stable at {last_size} bytes: {path.name}")
    return True


def _wait_until_readable(
    path: Path,
    max_attempts: int = 10,
    interval: float = 1.0,
) -> bool:
    """
    Try to open the file in binary read mode to confirm the OS write lock
    has been released.

    Returns True if the file became readable, False if all attempts failed.
    """
    for attempt in range(max_attempts):
        try:
            with open(path, "rb"):
                return True
        except (PermissionError, OSError):
            logger.debug(
                f"File not yet readable "
                f"(attempt {attempt + 1}/{max_attempts}): {path.name}"
            )
            time.sleep(interval)
    logger.error(f"File never became readable: {path.name}")
    return False


class AudioHandler(FileSystemEventHandler):
    """
    Handles file creation events for audio files and triggers processing.

    Processing is dispatched to a single-worker ThreadPoolExecutor so the
    watchdog observer thread remains responsive to new file events while a
    long transcription job is running. max_workers=1 ensures sequential
    processing — the GPU cannot handle two jobs concurrently.
    """

    def __init__(self, process_func: Callable[[Path], None]) -> None:
        super().__init__()
        self.process = process_func
        self._executor = ThreadPoolExecutor(max_workers=1)

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        try:
            src = _normalize_src_path(event.src_path)
        except TypeError as e:
            logger.error(f"Could not normalize event path: {e}")
            return
        path = Path(src)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        logger.info(f"Detected new audio file: {path.name} — queuing for processing.")
        self._executor.submit(self._handle, path)

    def _handle(self, path: Path) -> None:
        """
        Worker method: wait for file stability and readability, then process.
        Runs on the executor thread, not the watchdog observer thread.
        """
        try:
            _wait_until_stable(path)
            if not _wait_until_readable(path):
                logger.error(
                    f"Skipping {path.name} — file could not be opened for reading."
                )
                return
            self.process(path)
        except Exception as e:
            logger.error(f"Unhandled error processing {path.name}: {e}")
            import traceback
            traceback.print_exc()

    def shutdown(self, wait: bool = True) -> None:
        """
        Gracefully shut down the executor. Called by the watcher on
        KeyboardInterrupt to avoid cutting off an in-progress job.
        """
        logger.info("AudioHandler shutting down executor...")
        self._executor.shutdown(wait=wait)
        logger.info("AudioHandler executor shut down.")