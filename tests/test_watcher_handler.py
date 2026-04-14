import pytest
from unittest.mock import MagicMock
from pathlib import Path
from watcher.handler import AudioHandler

def test_audiohandler_on_created_valid_audio():
    called = {}
    def fake_process(path):
        called['path'] = path
    handler = AudioHandler(fake_process)
    event = MagicMock()
    event.is_directory = False
    event.src_path = 'file.wav'
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
