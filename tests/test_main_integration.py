import pytest
from unittest.mock import patch, MagicMock
from main import process
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
