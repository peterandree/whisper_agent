def _normalize_src_path(src_path) -> str:
    if isinstance(src_path, str):
        return src_path
    if isinstance(src_path, (bytes, bytearray)):
        return src_path.decode()
    if isinstance(src_path, memoryview):
        return src_path.tobytes().decode()
    raise TypeError(f"Unsupported src_path type: {type(src_path)}")
import time
import logging
from pathlib import Path
from typing import Callable
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from config.settings import AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)

def _wait_until_stable(path: Path, interval: float = 1.0, stable_count: int = 5) -> None:
    """Poll file size until it stops growing for stable_count consecutive checks."""
    last_size = -1
    stable = 0
    while stable < stable_count:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            stable = 0
            last_size = -1
            time.sleep(interval)
            continue
        if size == last_size and size > 0:
            stable += 1
        else:
            stable = 0
        last_size = size
        time.sleep(interval)
    logger.info(f"File stable at {last_size} bytes: {path.name}")


def _wait_until_readable(path: Path, max_attempts: int = 10, interval: float = 1.0) -> bool:
    """Try to open the file exclusively to confirm the OS has released the write lock."""
    for attempt in range(max_attempts):
        try:
            with open(path, "rb"):
                return True
        except (PermissionError, OSError):
            logger.debug(f"File not yet readable (attempt {attempt + 1}/{max_attempts}): {path.name}")
            time.sleep(interval)
    logger.error(f"File never became readable: {path.name}")
    return False


class AudioHandler(FileSystemEventHandler):
    """
    Handles file creation events for audio files and triggers processing.
    """
    def __init__(self, process_func: Callable[[Path], None]) -> None:
        super().__init__()
        self.process = process_func

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = _normalize_src_path(event.src_path)
        path = Path(src)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        logger.info(f"Detected new audio file: {path.name}")
        _wait_until_stable(path)
        if not _wait_until_readable(path):
            logger.error(f"Skipping {path.name} — file could not be opened for reading.")
            return
        self.process(path)