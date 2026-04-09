import torch
import json
from pathlib import Path
from torch.optim import AdamW

from config import CHECKPOINTS, GRPO, RUNTIME
from data import generate_dataset
from eval import run_evaluation
from grpo import (
    compute_group_advantages,
    compute_grpo_loss_from_scored_samples,
    generate_grouped_rollouts,
    iter_minibatches,
    load_grpo_components,
)


def save_grpo_checkpoint(model, tokenizer, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def save_training_log(training_log: list[dict], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(training_log, indent=2), encoding="utf-8")


def summarize_rollout_batch(scored_samples) -> dict[str, float]:
    rewards = [sample.reward for sample in scored_samples]
    if not rewards:
        return {
            "mean_reward": 0.0,
            "reward_hit_rate": 0.0,
        }

    return {
        "mean_reward": sum(rewards) / len(rewards),
        # Binary reward makes this an easy "how often did we get it right?"
        # signal for each rollout batch.
        "reward_hit_rate": sum(reward == 1.0 for reward in rewards) / len(rewards),
    }


def main():
    torch.manual_seed(RUNTIME.seed)
    components = load_grpo_components()
    training_log = []

    policy_model = components["policy_model"]
    reference_model = components["reference_model"]
    tokenizer = components["tokenizer"]

    optimizer = AdamW(policy_model.parameters(), lr=GRPO.learning_rate)

    print(f"Running {GRPO.num_outer_steps} GRPO outer steps")
    print(f"Rollout prompts per step: {GRPO.num_rollout_prompts}")
    print(f"Group size: {GRPO.group_size}")
    print(f"Update epochs per step: {GRPO.num_update_epochs}")
    print(f"Minibatch size: {GRPO.minibatch_size}")

    for outer_step in range(GRPO.num_outer_steps):
        print(f"\n=== GRPO outer step {outer_step + 1}/{GRPO.num_outer_steps} ===")

        # Each outer step should use fresh prompts and fresh samples from the
        # current policy so RL stays online rather than reusing stale rollouts.
        examples = generate_dataset(
            GRPO.num_rollout_prompts,
            seed=RUNTIME.seed + outer_step,
        )

        # Phase 1: collect a fresh rollout batch with the current policy.
        rollouts = generate_grouped_rollouts(
            policy_model,
            tokenizer,
            examples,
        )

        # Turn raw rewards into within-group relative advantages once for this
        # rollout batch, then reuse them across later update epochs.
        scored_samples = compute_group_advantages(rollouts)
        rollout_summary = summarize_rollout_batch(scored_samples)
        epoch_metrics = []

        print(f"Collected {len(scored_samples)} scored samples")
        print(
            f"Rollout mean_reward={rollout_summary['mean_reward']:.4f} "
            f"reward_hit_rate={rollout_summary['reward_hit_rate']:.4f}"
        )

        print("\nSampled completions:")
        for rollout in rollouts:
            print(f"\nPROMPT: {rollout.prompt}")
            for sample in rollout.samples:
                print(f"  TEXT: {sample.text!r}")
                print(f"  REWARD: {sample.reward}")

        print("\nScored samples:")
        for sample in scored_samples[:8]:
            print(
                f"prompt={sample.prompt!r} "
                f"reward={sample.reward} "
                f"advantage={sample.advantage:.4f} "
                f"text={sample.completion_text!r}"
            )

        # Phase 2: reuse this rollout batch for several optimizer passes.
        for epoch in range(GRPO.num_update_epochs):
            print(f"\nGRPO update epoch {epoch + 1}/{GRPO.num_update_epochs}")

            minibatch_seed = RUNTIME.seed + outer_step * 1000 + epoch

            for step, minibatch in enumerate(
                iter_minibatches(scored_samples, GRPO.minibatch_size, minibatch_seed),
                start=1,
            ):
                # Minibatches recompute fresh policy/reference logprobs under the
                # latest policy parameters, even though sampled completions stay fixed.
                loss, metrics = compute_grpo_loss_from_scored_samples(
                    policy_model,
                    reference_model,
                    tokenizer,
                    minibatch,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"step={step} "
                    f"total_loss={metrics['total_loss']:.4f} "
                    f"policy_loss={metrics['policy_loss']:.4f} "
                    f"mean_reward={metrics['mean_reward']:.4f} "
                    f"mean_advantage={metrics['mean_advantage']:.4f} "
                    f"mean_kl={metrics.get('mean_kl', 0.0):.4f}"
                )
                epoch_metrics.append(
                    {
                        "epoch": epoch + 1,
                        "minibatch_step": step,
                        "total_loss": metrics["total_loss"],
                        "policy_loss": metrics["policy_loss"],
                        "mean_reward": metrics["mean_reward"],
                        "mean_advantage": metrics["mean_advantage"],
                        "mean_kl": metrics.get("mean_kl", 0.0),
                    }
                )

        if (outer_step + 1) % GRPO.save_every_steps == 0:
            # Save the current policy so we can later evaluate GRPO directly.
            save_grpo_checkpoint(policy_model, tokenizer, CHECKPOINTS.grpo_checkpoint_dir)
            print(f"Saved GRPO checkpoint to {CHECKPOINTS.grpo_checkpoint_dir}")

            # Evaluate the freshly saved GRPO checkpoint on the fixed held-out set
            # so training produces evidence, not just logs.
            eval_summary, _, _ = run_evaluation(stage="grpo")
            print(
                "GRPO eval "
                f"accuracy={eval_summary['accuracy']:.4f} "
                f"parse_rate={eval_summary['parse_rate']:.4f}"
            )
        else:
            eval_summary = None

        training_log.append(
            {
                "outer_step": outer_step + 1,
                "num_scored_samples": len(scored_samples),
                "rollout_mean_reward": rollout_summary["mean_reward"],
                "rollout_reward_hit_rate": rollout_summary["reward_hit_rate"],
                "eval_accuracy": None if eval_summary is None else eval_summary["accuracy"],
                "eval_parse_rate": None if eval_summary is None else eval_summary["parse_rate"],
                "epoch_metrics": epoch_metrics,
            }
        )
        save_training_log(training_log, "artifacts/grpo_training_log.json")
        print("Saved GRPO training log to artifacts/grpo_training_log.json")


if __name__ == "__main__":
    main()
