from config import MODEL_NAME
from model import load_model_and_tokenizer


def main():
    model, tokenizer = load_model_and_tokenizer()

    print(f"Loaded model for SFT: {MODEL_NAME}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
    print("Next step: attach your dataset and optimizer loop here.")


if __name__ == "__main__":
    main()
