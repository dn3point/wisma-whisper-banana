import os
from functools import lru_cache

WHISPER_MODELS = {"tiny", "base", "small", "medium"}


@lru_cache(maxsize=1)
def get_whisper_model() -> str:
    model = os.getenv("WHISPER_MODEL", "small")
    model = "small" if model not in WHISPER_MODELS else model
    return model
