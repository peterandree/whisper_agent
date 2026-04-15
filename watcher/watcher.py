
from watchdog.observers import Observer
from .handler import AudioHandler
from config.settings import WATCH_DIR
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def start_watcher(process_func: Callable) -> None:
    """
    Start the file system watcher for the watch directory.

    Args:
        process_func (Callable): Function to process audio files.
    """
    logger.info(f"Watching {WATCH_DIR} ...")
    observer = Observer()
    observer.schedule(AudioHandler(process_func), str(WATCH_DIR), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
