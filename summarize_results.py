from __future__ import annotations

import json
from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_eval_table() -> None:
    eval_paths = {
        "base": ARTIFACTS_DIR / "base_eval.json",
        "sft": ARTIFACTS_DIR / "sft_eval.json",
        "grpo": ARTIFACTS_DIR / "grpo_eval.json",
    }

    print("Evaluation summary")
    print()
    print("| Stage | Model source | Accuracy | Parse rate |")
    print("|---|---|---:|---:|")

    for stage, path in eval_paths.items():
        payload = load_json(path)
        if payload is None:
            print(f"| {stage} | missing | - | - |")
            continue

        summary = payload["summary"]
        print(
            f"| {stage} | {summary['model_name']} | "
            f"{summary['accuracy']:.3f} | {summary['parse_rate']:.3f} |"
        )


def print_grpo_log_summary() -> None:
    payload = load_json(ARTIFACTS_DIR / "grpo_training_log.json")
    if payload is None:
        print("\nNo GRPO training log found.")
        return

    print("\nGRPO outer-step summary")
    print()
    print("| Outer step | Rollout mean reward | Reward hit rate | Eval accuracy |")
    print("|---:|---:|---:|---:|")

    for entry in payload:
        eval_accuracy = entry["eval_accuracy"]
        eval_text = "-" if eval_accuracy is None else f"{eval_accuracy:.3f}"
        print(
            f"| {entry['outer_step']} | "
            f"{entry['rollout_mean_reward']:.3f} | "
            f"{entry['rollout_reward_hit_rate']:.3f} | "
            f"{eval_text} |"
        )


def main():
    print_eval_table()
    print_grpo_log_summary()


if __name__ == "__main__":
    main()
