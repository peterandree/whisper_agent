from dataclasses import dataclass

@dataclass
class AlignModelCandidate:
    model_id: str
    vram_gb: float
    notes: str = ""

# Ordered best-first within each language.
# Selector picks the highest-ranked model that fits available VRAM.
ALIGN_MODEL_REGISTRY: dict[str, list[AlignModelCandidate]] = {
    "en": [
        AlignModelCandidate("facebook/wav2vec2-large-960h-lv60-self", vram_gb=1.2, notes="large, best accuracy"),
        AlignModelCandidate("facebook/wav2vec2-base-960h",            vram_gb=0.4, notes="base fallback"),
    ],
    "de": [
        AlignModelCandidate("jonatasgrosman/wav2vec2-xls-r-1b-german",      vram_gb=4.0, notes="1B, broadest training data"),
        AlignModelCandidate("jonatasgrosman/wav2vec2-large-xlsr-53-german", vram_gb=1.2, notes="300M, good general German"),
        AlignModelCandidate("WHISPERX_BUILTIN",                             vram_gb=0.3, notes="VoxPopuli base, last resort"),
    ],
    "zh": [
        AlignModelCandidate("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn", vram_gb=1.2, notes="multi-dataset XLSR"),
        AlignModelCandidate("WHISPERX_BUILTIN",                                    vram_gb=0.3, notes="whisperx default"),
    ],
}

WHISPERX_BUILTIN_LANGUAGES: set[str] = {
    "en", "fr", "de", "es", "it", "ja", "zh", "nl", "uk", "pt",
    "tl", "sv", "cs", "pl", "hu", "fi", "fa", "el", "tr", "ru",
    "da", "he", "ro", "sk", "sl", "hr", "bg", "lt", "lv", "et",
}

# Safety margin kept free to avoid OOM during model load
VRAM_SAFETY_MARGIN_GB: float = 1.5