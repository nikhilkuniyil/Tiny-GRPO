# tiny-grpo

Minimal scaffold for loading a Hugging Face causal language model and using it
from SFT, GRPO, or generation entry points.

## Model location in the codebase

- `config.py`: canonical model id and loader defaults
- `model.py`: shared `load_model_and_tokenizer()` helper
- `generate.py`: quick smoke test for text generation
- `train_sft.py`: place to start a supervised fine-tuning loop
- `grpo.py` and `train_grpo.py`: place to start GRPO-specific logic

## Default model

This repo is hard-coded to load:

`HuggingFaceTB/SmolLM2-135M`

If you want a different checkpoint later, update `MODEL_NAME` in `config.py`.

## Install

```bash
pip install -r requirements.txt
```

## Quick test

```bash
python generate.py --prompt "The capital of France is"
```

The model weights are downloaded by Hugging Face into the local cache, usually:

`~/.cache/huggingface/hub`

If you want a custom cache location, set `CACHE_DIR` in `config.py`.
