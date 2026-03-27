# tiny-grpo

Tiny educational GRPO repo built around one deliberately narrow task:
solving single-variable integer linear equations.

The point of the project is to make the reward loop obvious before adding
the full GRPO update. If the task, parser, and reward are easy to inspect,
the optimization code becomes much easier to understand.

## Current scope

- Hard-coded base model: `HuggingFaceTB/SmolLM2-135M`
- Synthetic math task in `data.py`
- Deterministic integer parser
- Binary exact-match reward
- Minimal model-loading and generation scaffold

## Project structure

- `config.py`: simple config objects for model, runtime, training, GRPO, and generation settings
- `model.py`: shared Hugging Face model/tokenizer loader
- `data.py`: synthetic equation generation, ground-truth solving, response parsing, and reward
- `generate.py`: quick smoke test for loading the model and generating text
- `eval.py`: fixed-set evaluation script that saves metrics and raw outputs for later comparison
- `train_sft.py`: placeholder entry point for supervised fine-tuning
- `grpo.py` and `train_grpo.py`: placeholder entry points for GRPO-specific logic

## Why start with synthetic equations

For a toy GRPO example, we want a task where:

- prompts are easy to generate
- correct answers are exact and cheap to compute
- rewards are deterministic
- failures are easy to inspect by hand

This repo currently generates equations like:

```text
Solve for x: -3x + 7 = -11
```

The gold answer is computed directly from the equation parameters, and the
reward is:

- `1.0` if the first valid integer in the model response matches the true answer
- `0.0` otherwise

That keeps the training signal simple and makes the GRPO loop easy to explain.

## Config

The main settings live in `config.py`:

- `MODEL`: model-loading settings
- `RUNTIME`: seed, device, logging, checkpoint cadence
- `TRAINING`: batch size, learning rate, sequence lengths, epochs
- `GRPO`: group size, KL coefficient, GRPO learning rate
- `GENERATION`: decoding defaults

The model is intentionally hard-coded through:

```python
MODEL.name = "HuggingFaceTB/SmolLM2-135M"
```

If you want to swap checkpoints later, change that value in `config.py`.

## Install

```bash
pip install -r requirements.txt
```

## Quick model test

```bash
python generate.py --prompt "Solve for x: 2x + 3 = 11"
```

Hugging Face will usually cache the model under:

```text
~/.cache/huggingface/hub
```

If needed, set a custom cache path through `MODEL.cache_dir` in `config.py`.

## Synthetic task API

`data.py` currently provides:

- `generate_equation_example()`
- `generate_dataset()`
- `solve_linear_equation()`
- `extract_first_integer()`
- `exact_match_reward()`

Example:

```python
from data import generate_equation_example, exact_match_reward

example = generate_equation_example()
print(example.prompt)

model_response = "x = 5"
reward = exact_match_reward(model_response, example.solution)
print(reward)
```

## Evaluation baseline

Before SFT or GRPO, run the raw base model on a small fixed held-out set:

```bash
python eval.py --stage base --num-examples 32
```

This script:

- generates and saves a fixed prompt set in `artifacts/eval_prompts.json`
- runs the current model on those prompts
- parses the first integer from each response
- computes exact-match reward and summary metrics
- saves the full result set in `artifacts/base_eval.json`

The key saved metrics are:

- `accuracy`
- `average_reward`
- `parse_rate`
- `invalid_output_rate`

Later you can run the same script for other stages:

```bash
python eval.py --stage sft --num-examples 32
python eval.py --stage grpo --num-examples 32
```

As long as you reuse the same `artifacts/eval_prompts.json`, those files are
directly comparable.

Use `--overwrite-eval-set` only when you intentionally want a new held-out set.

## Intended next step

The next implementation step is to connect the current task and evaluation loop into
`train_grpo.py`:

1. Sample a batch of equations from `data.py`.
2. Generate multiple responses per prompt.
3. Parse each response into an integer.
4. Compute exact-match rewards.
5. Use those rewards inside the GRPO loss.
