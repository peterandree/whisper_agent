from pathlib import Path
from watchdog.observers import Observer
from .handler import AudioHandler
from config.settings import WATCH_DIR
import time

def start_watcher():
    print(f"[*] Watching {WATCH_DIR} ...")
    observer = Observer()
    observer.schedule(AudioHandler(), str(WATCH_DIR), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
