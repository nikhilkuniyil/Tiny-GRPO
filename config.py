from dataclasses import dataclass
from typing import Optional

import torch

@dataclass(frozen=True)
class CheckpointConfig:
    sft_checkpoint_dir: str = "artifacts/sft_model"
    grpo_checkpoint_dir: str = "artifacts/grpo_model"


@dataclass(frozen=True)
class ModelConfig:
    name: str = "HuggingFaceTB/SmolLM2-135M"
    torch_dtype: str = "auto"
    device_map: Optional[str] = None
    use_fast_tokenizer: bool = True
    cache_dir: Optional[str] = None


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    # Default to CPU for this teaching setup 
    device: str = "cpu"
    log_every: int = 10
    save_every: int = 100


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    num_epochs: int = 3
    warmup_steps: int = 10
    max_prompt_length: int = 128
    max_completion_length: int = 128
    max_sequence_length: int = 256


@dataclass(frozen=True)
class GRPOConfig:
    group_size: int = 4
    num_generations_per_prompt: int = 4
    learning_rate: float = 1e-5
    kl_beta: float = 0.04
    clip_range: float = 0.2
    max_prompt_length: int = 128
    max_completion_length: int = 128
    num_update_epochs: int = 2
    minibatch_size: int = 8


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


CHECKPOINTS = CheckpointConfig()
MODEL = ModelConfig()
RUNTIME = RuntimeConfig()
TRAINING = TrainingConfig()
GRPO = GRPOConfig()
GENERATION = GenerationConfig()

# Backward-compatible aliases for the current loader code.
MODEL_NAME = MODEL.name
TORCH_DTYPE = MODEL.torch_dtype
DEVICE_MAP = MODEL.device_map
USE_FAST_TOKENIZER = MODEL.use_fast_tokenizer
CACHE_DIR = MODEL.cache_dir
DEVICE = RUNTIME.device
