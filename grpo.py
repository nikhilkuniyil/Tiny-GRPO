from model import load_model_and_tokenizer


def load_grpo_components():
    policy_model, tokenizer = load_model_and_tokenizer()

    # GRPO commonly keeps a separate frozen reference model for KL control.
    reference_model, _ = load_model_and_tokenizer()
    reference_model.eval()

    return {
        "policy_model": policy_model,
        "reference_model": reference_model,
        "tokenizer": tokenizer,
    }
