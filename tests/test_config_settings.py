import os
from config import settings

def test_env_var_override(monkeypatch):
    monkeypatch.setenv('WHISPER_AGENT_WATCH_DIR', '/tmp/raw')
    monkeypatch.setenv('WHISPER_AGENT_OUTPUT_DIR', '/tmp/summaries')
    monkeypatch.setenv('OLLAMA_URL', 'http://fake')
    monkeypatch.setenv('OLLAMA_MODEL', 'fake-model')
    import importlib
    importlib.reload(settings)
    assert settings.WATCH_DIR.as_posix() == '/tmp/raw'
    assert settings.OUTPUT_DIR.as_posix() == '/tmp/summaries'
    assert settings.OLLAMA_URL == 'http://fake'
    assert settings.OLLAMA_MODEL == 'fake-model'
