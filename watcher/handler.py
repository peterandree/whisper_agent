

import time
import logging
from pathlib import Path
from typing import Callable
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from config.settings import AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)


def _wait_until_stable(path: Path, interval: float = 0.5, stable_count: int = 4) -> None:
    """
    Wait until the file size is stable for several consecutive checks.
    """
    last_size = -1
    stable = 0
    while stable < stable_count:
        size = path.stat().st_size
        if size == last_size:
            stable += 1
        else:
            stable = 0
        last_size = size
        time.sleep(interval)

class AudioHandler(FileSystemEventHandler):
    """
    Handles file creation events for audio files and triggers processing.
    """
    def __init__(self, process_func: Callable[[Path], None]) -> None:
        """
        Initialize the handler.

        Args:
            process_func (Callable[[Path], None]): Function to process audio files.
        """
        super().__init__()
        self.process = process_func

    def on_created(self, event: FileSystemEvent) -> None:
        """
        Handle the creation of a new file. If it's an audio file, process it.

        Args:
            event (FileSystemEvent): The file system event.
        """
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        logger.info(f"Detected new audio file: {path}")
        # Wait for OBS to finish writing (wait for file size to stabilize)
        _wait_until_stable(path)
        self.process(path)
