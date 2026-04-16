import subprocess
from pathlib import Path

def get_audio_duration(path: Path) -> float:
    """
    Get the duration of an audio file in seconds using ffprobe.
    Returns 0.0 if duration cannot be determined.
    """
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0

def format_seconds(seconds: float) -> str:
    """
    Format seconds as hh:mm:ss if >= 1 hour, else mm:ss.
    """
    seconds = int(seconds)
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    if h > 0:
        return f"{h:02}:{m:02}:{s:02}"
    else:
        return f"{m:02}:{s:02}"