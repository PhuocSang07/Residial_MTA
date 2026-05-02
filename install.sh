#!/bin/bash
set -e

# Core deep learning
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HuggingFace ecosystem
pip install \
    transformers \
    datasets \
    accelerate \
    peft \
    safetensors \
    sentencepiece \
    tokenizers \
    huggingface-hub

# Training utilities
pip install \
    deepspeed \
    wandb \
    tqdm \
    numpy \
    scipy \
    einops \
    ninja

# Evaluation metrics
pip install \
    rouge-score \
    bert-score \
    nltk \
    evaluate

# Misc
pip install \
    fire \
    PyYAML \
    openai \
    packaging \
    psutil

echo ""