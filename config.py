from dataclasses import dataclass
from typing import Optional


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
TORCH_DTYPE = "auto"
DEVICE_MAP = "auto"
USE_FAST_TOKENIZER = True
CACHE_DIR: Optional[str] = None
DEFAULT_MAX_NEW_TOKENS = 128


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = 0.8
    top_p: float = 0.95
    do_sample: bool = True
