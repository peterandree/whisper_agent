import shutil
import tempfile
from config.settings import OUTPUT_DIR

def test_no_side_effect_on_import():
    # OUTPUT_DIR should not exist after import
    temp_dir = tempfile.mkdtemp()
    try:
        # Patch OUTPUT_DIR to a temp path
        import importlib
        import os
        os.environ['WHISPER_AGENT_OUTPUT_DIR'] = temp_dir + '/should_not_exist'
        import config.settings as settings_reload
        importlib.reload(settings_reload)
        assert not settings_reload.OUTPUT_DIR.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
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
