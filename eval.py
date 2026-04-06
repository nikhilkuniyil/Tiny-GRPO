from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from config import CHECKPOINTS, GENERATION, MODEL_NAME, RUNTIME
from data import EquationExample, exact_match_reward, extract_first_integer, format_generation_prompt, generate_dataset
from model import load_model_and_tokenizer


ARTIFACTS_DIR = Path("artifacts")
EVAL_SET_PATH = ARTIFACTS_DIR / "eval_prompts.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="base", help="Label to use for saved results, e.g. base, sft, grpo.")
    parser.add_argument("--num-examples", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=GENERATION.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=GENERATION.temperature)
    parser.add_argument("--top-p", type=float, default=GENERATION.top_p)
    parser.add_argument("--overwrite-eval-set", action="store_true")
    return parser.parse_args()


def save_eval_set(examples: list[EquationExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(example) for example in examples], handle, indent=2)


def load_eval_set(path: Path) -> list[EquationExample]:
    with path.open("r", encoding="utf-8") as handle:
        raw_examples = json.load(handle)
    return [EquationExample(**example) for example in raw_examples]


def get_eval_examples(num_examples: int, overwrite: bool) -> list[EquationExample]:
    # Reuse the same held-out set across base, SFT, and GRPO runs so the
    # comparison is apples-to-apples.
    if EVAL_SET_PATH.exists() and not overwrite:
        examples = load_eval_set(EVAL_SET_PATH)
        return examples[:num_examples]

    examples = generate_dataset(num_examples, seed=RUNTIME.seed)
    save_eval_set(examples, EVAL_SET_PATH)
    return examples


def resolve_model_source(stage: str) -> str:
    if stage == "base":
        return MODEL_NAME
    if stage == "sft":
        return CHECKPOINTS.sft_checkpoint_dir
    if stage == "grpo":
        return CHECKPOINTS.grpo_checkpoint_dir
    return MODEL_NAME


def generate_response(model, tokenizer, prompt: str, *, max_new_tokens: int, temperature: float, top_p: float) -> str:
    generation_prompt = format_generation_prompt(prompt)
    inputs = tokenizer(generation_prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        # Keep generation settings explicit here because these choices affect
        # the baseline you compare against later.
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=GENERATION.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Only score the generated completion, not the original prompt tokens.
    completion_ids = output_ids[0][prompt_length:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def evaluate_examples(
    examples: list[EquationExample],
    *,
    model_source: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    model, tokenizer = load_model_and_tokenizer(model_source=model_source)
    results = []

    for example in examples:
        raw_output = generate_response(
            model,
            tokenizer,
            example.prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        parsed_output = extract_first_integer(raw_output)
        reward = exact_match_reward(raw_output, example.solution)

        # Save both the raw text and the parsed integer so you can inspect
        # formatting failures separately from reasoning failures.
        results.append(
            {
                "prompt": example.prompt,
                "equation": example.equation,
                "target": example.solution,
                "raw_output": raw_output,
                "parsed_output": parsed_output,
                "reward": reward,
            }
        )

    return results


def summarize_results(results: list[dict], stage: str, model_source: str) -> dict:
    num_examples = len(results)
    num_parsed = sum(result["parsed_output"] is not None for result in results)
    num_correct = sum(result["reward"] == 1.0 for result in results)

    # For the current binary reward, average reward and exact-match accuracy
    # are the same quantity, but both names are useful for clarity.
    return {
        "stage": stage,
        "model_name": model_source,
        "num_examples": num_examples,
        "accuracy": num_correct / num_examples if num_examples else 0.0,
        "average_reward": sum(result["reward"] for result in results) / num_examples if num_examples else 0.0,
        "parse_rate": num_parsed / num_examples if num_examples else 0.0,
        "invalid_output_rate": (num_examples - num_parsed) / num_examples if num_examples else 0.0,
    }


def save_results(stage: str, summary: dict, results: list[dict]) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACTS_DIR / f"{stage}_eval.json"
    payload = {
        "summary": summary,
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def print_summary(summary: dict) -> None:
    print(f"Stage: {summary['stage']}")
    print(f"Model: {summary['model_name']}")
    print(f"Examples: {summary['num_examples']}")
    print(f"Accuracy: {summary['accuracy']:.3f}")
    print(f"Average reward: {summary['average_reward']:.3f}")
    print(f"Parse rate: {summary['parse_rate']:.3f}")
    print(f"Invalid output rate: {summary['invalid_output_rate']:.3f}")


def print_example_block(result: dict) -> None:
    print(f"Prompt: {result['prompt']}")
    print(f"Target: {result['target']}")
    print(f"Parsed output: {result['parsed_output']}")
    print(f"Reward: {result['reward']}")
    print(f"Raw output: {result['raw_output']}")
    print()


def print_example_analysis(results: list[dict], *, max_incorrect: int = 3, max_correct: int = 2) -> None:
    incorrect_results = [result for result in results if result["reward"] == 0.0]
    correct_results = [result for result in results if result["reward"] == 1.0]

    if incorrect_results:
        print("Sample incorrect examples:")
        print()
        for result in incorrect_results[:max_incorrect]:
            print_example_block(result)

    if correct_results:
        print("Sample correct examples:")
        print()
        for result in correct_results[:max_correct]:
            print_example_block(result)


def main():
    args = parse_args()
    model_source = resolve_model_source(args.stage)
    examples = get_eval_examples(args.num_examples, args.overwrite_eval_set)
    results = evaluate_examples(
        examples,
        model_source=model_source,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    summary = summarize_results(results, stage=args.stage, model_source=model_source)
    output_path = save_results(args.stage, summary, results)

    print_summary(summary)
    print()
    print_example_analysis(results)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
