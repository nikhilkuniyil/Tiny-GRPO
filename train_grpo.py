import torch
from torch.optim import AdamW

from config import GRPO, RUNTIME
from data import generate_dataset
from grpo import (
    compute_group_advantages,
    compute_grpo_loss_from_scored_samples,
    generate_grouped_rollouts,
    iter_minibatches,
    load_grpo_components,
)

def main():
    torch.manual_seed(RUNTIME.seed)

    examples = generate_dataset(4, seed=RUNTIME.seed)
    components = load_grpo_components()

    policy_model = components["policy_model"]
    reference_model = components["reference_model"]
    tokenizer = components["tokenizer"]

    optimizer = AdamW(policy_model.parameters(), lr=GRPO.learning_rate)

    # Phase 1: collect grouped rollouts from the current policy
    rollouts = generate_grouped_rollouts(
        policy_model,
        tokenizer,
        examples,
    )

    # Turn raw rewards into relative within-group advantages
    scored_samples = compute_group_advantages(rollouts)

    print(f"Collected {len(scored_samples)} scored samples")
    print(f"Running {GRPO.num_update_epochs} update epochs")
    print(f"Minibatch size: {GRPO.minibatch_size}")

    # Phase 2: reuse this rollout batch for several optimization passes
    for epoch in range(GRPO.num_update_epochs):
        print(f"\nGRPO update epoch {epoch + 1}/{GRPO.num_update_epochs}")

        minibatch_seed = RUNTIME.seed + epoch

        for step, minibatch in enumerate(
            iter_minibatches(scored_samples, GRPO.minibatch_size, minibatch_seed),
            start=1,
        ):
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


if __name__ == "__main__":
    main()
