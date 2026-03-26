from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CACHE_DIR, DEVICE_MAP, MODEL_NAME, TORCH_DTYPE, USE_FAST_TOKENIZER


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"

    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=USE_FAST_TOKENIZER,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    *,
    device_map: Optional[str] = DEVICE_MAP,
    torch_dtype: str = TORCH_DTYPE,
):
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=device_map,
        torch_dtype=resolve_torch_dtype(torch_dtype),
        cache_dir=CACHE_DIR,
    )


def load_model_and_tokenizer(
    *,
    device_map: Optional[str] = DEVICE_MAP,
    torch_dtype: str = TORCH_DTYPE,
):
    tokenizer = load_tokenizer()
    model = load_model(
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer
