import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from audio.transcription import transcribe

@patch('audio.transcription.whisperx')
@patch('audio.transcription.WhisperModel')
@patch('audio.transcription.DiarizationPipeline')
@patch('audio.transcription.torch')
@patch('audio.transcription.gc')
def test_transcribe_basic(mock_gc, mock_torch, mock_diar, mock_whispermodel, mock_whisperx):
    # Setup mocks
    mock_audio = MagicMock()
    mock_whisperx.load_audio.return_value = mock_audio
    mock_fw_model = MagicMock()
    mock_whispermodel.return_value = mock_fw_model
    mock_fw_model.transcribe.return_value = ([MagicMock(start=0, end=1, text='Hello')], MagicMock(language='en'))
    mock_whisperx.load_align_model.return_value = (MagicMock(), MagicMock())
    mock_whisperx.align.return_value = {'segments': [{'speaker': 'A', 'text': 'Hello', 'start': 0, 'end': 1}]}
    mock_diar_model = MagicMock()
    mock_diar.return_value = mock_diar_model
    mock_diar_model.return_value = [{'speaker': 'A', 'text': 'Hello', 'start': 0, 'end': 1}]
    mock_whisperx.assign_word_speakers.return_value = {'segments': [{'speaker': 'A', 'text': 'Hello', 'start': 0, 'end': 1}]}

    # Call function
    result = transcribe(Path('dummy.wav'), 'cpu', 'float32', 'token')
    assert 'A:' in result
    assert 'Hello' in result
