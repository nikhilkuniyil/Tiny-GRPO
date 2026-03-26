from config import MODEL_NAME
from grpo import load_grpo_components


def main():
    components = load_grpo_components()

    print(f"Loaded policy/reference models for GRPO: {MODEL_NAME}")
    print(f"Tokenizer vocab size: {components['tokenizer'].vocab_size}")
    print("Next step: add rollout generation, reward computation, and GRPO loss.")


if __name__ == "__main__":
    main()
