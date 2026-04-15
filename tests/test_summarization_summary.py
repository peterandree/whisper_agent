import requests
import time

from unittest.mock import patch, MagicMock

@patch('summarization.summary.requests.post')
def test_call_ollama_retries_on_failure(mock_post):
    # Simulate two failures, then a success
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = [requests.RequestException("fail1"), requests.RequestException("fail2"), None]
    mock_response.json.return_value = {"response": "ok"}
    mock_post.return_value = mock_response

    from summarization import summary
    # Patch time.sleep to avoid actual waiting
    with patch.object(time, "sleep", return_value=None) as mock_sleep:
        result = summary._call_ollama("prompt", "url", "model")
        assert result == "ok"
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2
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
