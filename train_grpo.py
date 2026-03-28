from config import MODEL_NAME
from data import generate_dataset
from grpo import generate_grouped_rollouts, load_grpo_components

def main():
    examples = generate_dataset(4, seed=42)
    components = load_grpo_components()

    rollouts = generate_grouped_rollouts(
        components["policy_model"],
        components["tokenizer"],
        examples,
    )

    for rollout in rollouts:
        print("PROMPT:", rollout.prompt)
        for sample in rollout.samples:
            print("  TEXT:", sample.text)
            print("  REWARD:", sample.reward)


if __name__ == "__main__":
    main()
