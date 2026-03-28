from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from config import MODEL_NAME, RUNTIME, TRAINING
from data import SFTExample, generate_sft_dataset
from model import load_model_and_tokenizer


class SFTDataset(Dataset):
    def __init__(self, examples: list[SFTExample], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTExample:
        return self.examples[index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--output-dir", default="artifacts/sft_model")
    return parser.parse_args()


def build_batch(examples: list[SFTExample], tokenizer):
    input_id_rows = []
    label_rows = []
    attention_mask_rows = []

    for example in examples:
        # Train on the answer tokens only. The prompt is context, not a target.
        prompt_text = f"{example.prompt}\nAnswer:"
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=TRAINING.max_prompt_length,
        )["input_ids"]
        target_ids = tokenizer(
            f" {example.target}",
            add_special_tokens=False,
            truncation=True,
            max_length=TRAINING.max_completion_length,
        )["input_ids"]

        # We concatenate prompt + target manually so the label mask can cleanly
        # ignore prompt tokens and score only the answer tokens.
        full_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        full_ids = full_ids[: TRAINING.max_sequence_length]

        labels = ([-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id])[: len(full_ids)]
        attention_mask = [1] * len(full_ids)

        input_id_rows.append(torch.tensor(full_ids, dtype=torch.long))
        label_rows.append(torch.tensor(labels, dtype=torch.long))
        attention_mask_rows.append(torch.tensor(attention_mask, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_rows,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        label_rows,
        batch_first=True,
        padding_value=-100,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        attention_mask_rows,
        batch_first=True,
        padding_value=0,
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def save_model(model, tokenizer, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def train_one_epoch(model, dataloader, optimizer, device: str, epoch_index: int) -> None:
    model.train()
    optimizer.zero_grad()

    for step, examples in enumerate(dataloader, start=1):
        # The dataset stores plain Python examples; batching/tokenization stays
        # here so it is easy to see exactly what supervision the model gets.
        batch = build_batch(examples, dataloader.dataset.tokenizer)
        batch = move_batch_to_device(batch, device)

        outputs = model(**batch)
        loss = outputs.loss / TRAINING.grad_accum_steps
        loss.backward()

        if step % TRAINING.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        if step % RUNTIME.log_every == 0 or step == 1:
            print(f"epoch={epoch_index + 1} step={step} loss={loss.item() * TRAINING.grad_accum_steps:.4f}")

    # Flush a final optimizer step when the number of batches is not divisible
    # by gradient accumulation.
    if len(dataloader) % TRAINING.grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()


def main():
    args = parse_args()
    torch.manual_seed(RUNTIME.seed)

    model, tokenizer = load_model_and_tokenizer(device=RUNTIME.device)
    examples = generate_sft_dataset(args.num_examples, seed=RUNTIME.seed)
    dataset = SFTDataset(examples, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=TRAINING.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=TRAINING.learning_rate,
        weight_decay=TRAINING.weight_decay,
    )

    print(f"Loaded model for SFT: {MODEL_NAME}")
    print(f"Training examples: {len(examples)}")
    print(f"Batch size: {TRAINING.batch_size}")
    print(f"Learning rate: {TRAINING.learning_rate}")
    print(f"Device: {RUNTIME.device}")

    for epoch_index in range(TRAINING.num_epochs):
        train_one_epoch(model, dataloader, optimizer, RUNTIME.device, epoch_index)

    save_model(model, tokenizer, args.output_dir)
    print(f"Saved SFT model to {args.output_dir}")


if __name__ == "__main__":
    main()
