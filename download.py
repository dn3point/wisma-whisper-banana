# This file runs during container build time to get model weights built into the container
# In this example: A Whisper model

from transformers import pipeline
from .utils.constants import get_whisper_model
import torch


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
