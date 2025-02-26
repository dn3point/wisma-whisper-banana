# This file runs during container build time to get model weights built into the container
# In this example: A Whisper model

from functools import lru_cache
import os
from transformers import pipeline
import torch


WHISPER_MODELS = {"tiny", "base", "small", "medium"}


@lru_cache(maxsize=1)
def get_whisper_model() -> str:
    model = os.getenv("WHISPER_MODEL", "small")
    model = "small" if model not in WHISPER_MODELS else model
    return model


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # typically, you want this single pipeline() call to match what is in your app.py
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipeline(
        model=f"openai/whisper-{get_whisper_model()}.en",
        chunk_length_s=30,
        device=device,
    )


if __name__ == "__main__":
    download_model()
