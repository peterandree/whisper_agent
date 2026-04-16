from watchdog.observers import Observer
from .handler import AudioHandler
from config.settings import WATCH_DIR
import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def start_watcher(process_func: Callable) -> None:
    logger.info(f"Watching {WATCH_DIR} ...")
    handler = AudioHandler(process_func)
    observer = Observer()
    observer.schedule(handler, str(WATCH_DIR), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested. Waiting for current job to finish...")
        observer.stop()
        handler.shutdown(wait=True)
    observer.join()
    logger.info("Watcher stopped.")