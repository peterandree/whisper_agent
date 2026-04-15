## Running Tests

To run all tests using uv and pytest:

```
uv pip install pytest
uv run -m pytest
```

Or, with pip:

```
pip install pytest
pytest
```
# Whisper Agent


## Installation (with uv)

Install dependencies and the package using uv (recommended):

```
uv pip install -e .
```

Or, for standard pip:

```
pip install -e .
```


## Usage



After installation, run the app using uv (recommended):

```
uv run whisper-agent
```

Or, if you prefer pip or Python directly:

```
python main.py
```

This will start the directory watcher and process new audio files as described above.

## Requirements
- Python 3.12
- CUDA-enabled GPU (for best performance)
- Set the `HF_TOKEN` environment variable for diarization

## Configuration
- Edit paths and model names in the code or (after refactor) in the config module.
