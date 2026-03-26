import argparse

import torch

from config import GENERATION
from model import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=GENERATION.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=GENERATION.temperature)
    parser.add_argument("--top-p", type=float, default=GENERATION.top_p)
    return parser.parse_args()


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=GENERATION.do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
