import time
from pathlib import Path
from watchdog.events import FileSystemEventHandler
from . import AUDIO_EXTENSIONS, process

class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            return
        # Wait for OBS to finish writing
        time.sleep(3)
        process(path)
