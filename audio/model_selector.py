import logging
from config.language_models import (
    ALIGN_MODEL_REGISTRY,
    WHISPERX_BUILTIN_LANGUAGES,
    VRAM_SAFETY_MARGIN_GB,
    AlignModelCandidate,
)
from audio.vram import get_free_vram_gb

logger = logging.getLogger(__name__)


def select_align_model(language: str) -> str | None:
    """
    Select the best available alignment model for the given language
    based on current free VRAM.

    Returns:
        HuggingFace model ID string,
        None if whisperx should use its built-in default,
        or 'SKIP' if no model fits or language is unsupported.
    """
    free_vram = get_free_vram_gb()
    budget = free_vram - VRAM_SAFETY_MARGIN_GB

    candidates: list[AlignModelCandidate] = ALIGN_MODEL_REGISTRY.get(language, [])

    if candidates:
        for candidate in candidates:
            if candidate.vram_gb <= budget:
                if candidate.model_id == "WHISPERX_BUILTIN":
                    logger.info(
                        f"Selected alignment model for '{language}': "
                        f"whisperx built-in ({candidate.notes}) "
                        f"[requires {candidate.vram_gb:.1f} GB, available {budget:.1f} GB]"
                    )
                    return None
                logger.info(
                    f"Selected alignment model for '{language}': "
                    f"{candidate.model_id} ({candidate.notes}) "
                    f"[requires {candidate.vram_gb:.1f} GB, available {budget:.1f} GB]"
                )
                return candidate.model_id
        logger.warning(
            f"No alignment model for '{language}' fits in available VRAM "
            f"({budget:.1f} GB). Skipping alignment."
        )
        return "SKIP"

    # Language not in registry — fall back to whisperx built-in if known
    if language in WHISPERX_BUILTIN_LANGUAGES:
        logger.info(
            f"Language '{language}' not in registry, "
            f"using whisperx built-in alignment model."
        )
        return None

    logger.warning(
        f"No alignment model available for language '{language}'. "
        f"Alignment will be skipped. To add support, add an entry to "
        f"ALIGN_MODEL_REGISTRY in config/language_models.py."
    )
    return "SKIP"