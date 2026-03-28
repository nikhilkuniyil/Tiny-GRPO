from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CACHE_DIR, DEVICE, DEVICE_MAP, MODEL_NAME, TORCH_DTYPE, USE_FAST_TOKENIZER


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


def load_tokenizer(model_source: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=USE_FAST_TOKENIZER,
        cache_dir=CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_source: str = MODEL_NAME,
    *,
    device: str = DEVICE,
    device_map: Optional[str] = DEVICE_MAP,
    torch_dtype: str = TORCH_DTYPE,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map=device_map,
        dtype=resolve_torch_dtype(torch_dtype),
        cache_dir=CACHE_DIR,
    )
    if device_map is None:
        model = model.to(device)
    model.eval()
    return model
 

def load_model_and_tokenizer(
    model_source: str = MODEL_NAME,
    *,
    device: str = DEVICE,
    device_map: Optional[str] = DEVICE_MAP,
    torch_dtype: str = TORCH_DTYPE,
):
    tokenizer = load_tokenizer(model_source=model_source)
    model = load_model(
        model_source = model_source,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer
