import pytest
from unittest.mock import patch, MagicMock
from summarization.summary import summarize

@patch('summarization.summary._call_ollama')
def test_summarize_single_chunk(mock_call_ollama):
    mock_call_ollama.return_value = 'summary'
    transcript = 'short transcript'
    result = summarize(transcript, 'http://fake', 'model')
    assert result == 'summary'
    mock_call_ollama.assert_called_once()

@patch('summarization.summary._call_ollama')
def test_summarize_multiple_chunks(mock_call_ollama):
    mock_call_ollama.side_effect = lambda prompt, url, model: 'partial' if 'Transcript:' in prompt else 'merged'
    transcript = 'a' * 13000  # Will be split into 2 chunks
    result = summarize(transcript, 'http://fake', 'model')
    assert 'merged' in result or 'partial' in result
