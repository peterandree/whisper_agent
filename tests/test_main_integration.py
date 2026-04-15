import tempfile
import shutil
from config.settings import OUTPUT_DIR

def test_output_dir_created_on_main(monkeypatch):
    temp_dir = tempfile.mkdtemp()
    try:
        monkeypatch.setenv('WHISPER_AGENT_OUTPUT_DIR', temp_dir + '/created')
        import importlib
        import whisper_agent.main as main_mod
        importlib.reload(main_mod)
        main_mod.main()
        assert main_mod.OUTPUT_DIR.exists()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
import pytest
from unittest.mock import patch, MagicMock
from whisper_agent.main import process
from pathlib import Path


@patch('main.transcribe')
@patch('main.summarize')
def test_process_success(mock_summarize, mock_transcribe, tmp_path):
    mock_transcribe.return_value = 'transcript'
    mock_summarize.return_value = 'summary'
    test_file = tmp_path / 'audio.wav'
    test_file.write_text('dummy')
    with patch('main.OUTPUT_DIR', tmp_path):
        process(test_file)
    # Check that a transcript and summary file were created with the correct stem
    files = [f.name for f in tmp_path.iterdir()]
    assert any(f.startswith(f'{test_file.stem}_') and f.endswith('.transcript.txt') for f in files)
    assert any(f.startswith(f'{test_file.stem}_') and f.endswith('.md') for f in files)

@patch('main.transcribe', side_effect=Exception('fail'))
def test_process_failure(mock_transcribe, tmp_path):
    test_file = tmp_path / 'audio.wav'
    test_file.write_text('dummy')
    with patch('main.OUTPUT_DIR', tmp_path):
        try:
            process(test_file)
        except Exception:
            pytest.fail('process() should not raise')
