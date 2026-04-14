
import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from config.settings import AUDIO_EXTENSIONS

class AudioHandler(FileSystemEventHandler):
    def __init__(self, process_func):
        super().__init__()
        self.process = process_func

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        # Wait for OBS to finish writing
        time.sleep(3)
        self.process(path)
