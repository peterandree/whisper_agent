import pytest
from unittest.mock import MagicMock
from pathlib import Path
from watcher.handler import AudioHandler
from unittest.mock import patch

def test_audiohandler_on_created_valid_audio():
    called = {}
    def fake_process(path):
        called['path'] = path
    handler = AudioHandler(fake_process)
    event = MagicMock()
    event.is_directory = False
    event.src_path = 'file.wav'
    # Patch Path.stat to simulate file size stability
    sizes = [100, 100, 100, 100]  # stable after first call
    class FakeStat:
        def __init__(self, size):
            self.st_size = size
    def fake_stat(self):
        return FakeStat(sizes.pop(0) if sizes else 100)
    with patch.object(Path, 'stat', fake_stat):
        handler.on_created(event)
    assert called['path'] == Path('file.wav')

def test_audiohandler_on_created_non_audio():
    called = {}
    def fake_process(path):
        called['path'] = path
    handler = AudioHandler(fake_process)
    event = MagicMock()
    event.is_directory = False
    event.src_path = 'file.txt'
    handler.on_created(event)
    assert 'path' not in called
