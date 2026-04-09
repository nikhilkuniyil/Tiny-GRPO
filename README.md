# tiny-grpo

Tiny educational GRPO repo built around one deliberately narrow task:
solving single-variable integer linear equations.

The point of the project is to make the reward loop obvious before adding
the full GRPO update. If the task, parser, and reward are easy to inspect,
the optimization code becomes much easier to understand.

The current goal of the repo is not just to implement GRPO, but to show
measurable evidence that GRPO improves performance over a base model and an
SFT-only baseline on the same held-out task.

## Current scope

- Hard-coded base model: `HuggingFaceTB/SmolLM2-135M`
- Synthetic math task in `data.py`
- Deterministic integer parser
- Binary exact-match reward
- SFT training loop
- GRPO rollout, loss, and outer training loop
- Evaluation and metric logging for base vs SFT vs GRPO comparison

## Current results

Using a fixed 32-example held-out set, the current repo produces:

| Stage | Model source | Accuracy | Parse rate |
|---|---|---:|---:|
| Base | `HuggingFaceTB/SmolLM2-135M` | `0.031` | `1.000` |
| SFT | `artifacts/sft_model` | `0.031` | `1.000` |
| GRPO | `artifacts/grpo_model` | `0.156` | `1.000` |

At the moment, most of the gain comes from GRPO improving answer selection on
top of a weak-but-format-aligned SFT checkpoint.

## Project structure

- `config.py`: simple config objects for model, runtime, training, GRPO, and generation settings
- `model.py`: shared Hugging Face model/tokenizer loader
- `data.py`: synthetic equation generation, ground-truth solving, response parsing, and reward
- `generate.py`: quick smoke test for loading the model and generating text
- `eval.py`: fixed-set evaluation script that saves metrics and raw outputs for later comparison
- `train_sft.py`: supervised fine-tuning entry point for teaching the answer format
- `grpo.py`: grouped rollout collection, reward normalization, and GRPO loss
- `train_grpo.py`: GRPO outer loop, minibatch updates, checkpoint saving, and training logs

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

## Evidence-first workflow

This repo is designed around direct comparisons on the same task:

1. Evaluate the base model.
2. Train and evaluate the SFT checkpoint.
3. Train and evaluate the GRPO checkpoint.

The main outputs to compare are:

- `artifacts/base_eval.json`
- `artifacts/sft_eval.json`
- `artifacts/grpo_eval.json`
- `artifacts/grpo_training_log.json`

That means the repo is not only demonstrating how GRPO is coded, but also
showing whether it actually improves exact-match accuracy on this toy task.

## Tracking GRPO experiments

`train_grpo.py` now runs an outer RL loop:

1. Generate fresh prompts.
2. Sample grouped completions from the current policy.
3. Compute rewards and within-group advantages.
4. Reuse that rollout batch for multiple update epochs and minibatches.
5. Save the latest GRPO checkpoint.
6. Save rollout/update metrics to `artifacts/grpo_training_log.json`.

This makes it easy to vary `GRPO.num_outer_steps` in `config.py` and compare:

- reward hit rate by outer step
- loss/KL trends by minibatch
- held-out accuracy after different numbers of GRPO outer steps

Suggested workflow:

```bash
python3 train_grpo.py
python3 eval.py --stage grpo --num-examples 32 --overwrite-eval-set
```

If you want to study scaling with outer steps, change `GRPO.num_outer_steps`,
rerun training, and compare the resulting `artifacts/grpo_training_log.json`
and `artifacts/grpo_eval.json`.
