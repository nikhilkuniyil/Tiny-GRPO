from dataclasses import dataclass

import torch

from config import CHECKPOINTS, GENERATION, GRPO, RUNTIME
from data import exact_match_reward
from model import load_model_and_tokenizer


@dataclass
class CompletionSample:
    text: str
    completion_ids: list[int]
    logprobs: list[float]
    reward: float
    stopped_by_eos: bool


@dataclass
class PromptRollout:
    prompt: str
    target: int
    samples: list[CompletionSample]


def load_grpo_components():
    model_source = CHECKPOINTS.sft_checkpoint_dir
    # The policy model is the model we will eventually optimize with GRPO.
    policy_model, tokenizer = load_model_and_tokenizer(model_source)

    # GRPO typically keeps a separate frozen reference model so later we can
    # compare the policy against a stable baseline with a KL-style penalty.
    reference_model, _ = load_model_and_tokenizer(model_source)
    reference_model.eval()

    return {
        "policy_model": policy_model,
        "reference_model": reference_model,
        "tokenizer": tokenizer,
    }


def sample_next_token_with_logprob(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")

    scaled_logits = logits / temperature
    full_logprobs = torch.log_softmax(scaled_logits, dim=-1)
    full_probs = torch.exp(full_logprobs)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(full_probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        keep_mask = cumulative_probs <= top_p
        keep_mask[..., 0] = True

        filtered_probs = sorted_probs * keep_mask
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        sampled_offsets = torch.multinomial(filtered_probs, num_samples=1)
        next_tokens = sorted_indices.gather(-1, sampled_offsets).squeeze(-1)
    else:
        next_tokens = torch.multinomial(full_probs, num_samples=1).squeeze(-1)

    chosen_logprobs = full_logprobs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
    return next_tokens, chosen_logprobs


def generate_grouped_rollouts(model, tokenizer, examples):
    prompts = [example.prompt for example in examples]
    targets = [example.solution for example in examples]

    expanded_prompts = []
    expanded_targets = []
    for prompt, target in zip(prompts, targets):
        for _ in range(GRPO.group_size):
            expanded_prompts.append(prompt)
            expanded_targets.append(target)

    encoded = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=GRPO.max_prompt_length,
    )

    input_ids = encoded["input_ids"].to(RUNTIME.device)
    attention_mask = encoded["attention_mask"].to(RUNTIME.device)

    batch_size = input_ids.size(0)
    generated_token_lists = [[] for _ in range(batch_size)]
    generated_logprob_lists = [[] for _ in range(batch_size)]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=RUNTIME.device)
    stopped_by_eos = [False] * batch_size

    model.eval()

    for _ in range(GENERATION.max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token_logits = outputs.logits[:, -1, :]

        next_tokens, next_logprobs = sample_next_token_with_logprob(
            next_token_logits,
            temperature=GENERATION.temperature,
            top_p=GENERATION.top_p,
        )

        next_tokens = torch.where(
            finished,
            torch.full_like(next_tokens, tokenizer.pad_token_id),
            next_tokens,
        )

        for i in range(batch_size):
            if finished[i]:
                continue

            token_id = next_tokens[i].item()
            generated_token_lists[i].append(token_id)
            generated_logprob_lists[i].append(next_logprobs[i].item())

            if token_id == tokenizer.eos_token_id:
                finished[i] = True
                stopped_by_eos[i] = True

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)

        new_attention_column = torch.ones(
            batch_size,
            1,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([attention_mask, new_attention_column], dim=1)

        if finished.all():
            break

    flat_samples = []
    for i in range(batch_size):
        completion_ids = generated_token_lists[i]
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        reward = exact_match_reward(completion_text, expanded_targets[i])

        flat_samples.append(
            CompletionSample(
                text=completion_text,
                completion_ids=completion_ids,
                logprobs=generated_logprob_lists[i],
                reward=reward,
                stopped_by_eos=stopped_by_eos[i],
            )
        )

    grouped_rollouts = []
    cursor = 0
    for prompt, target in zip(prompts, targets):
        samples = flat_samples[cursor : cursor + GRPO.group_size]
        grouped_rollouts.append(
            PromptRollout(
                prompt=prompt,
                target=target,
                samples=samples,
            )
        )
        cursor += GRPO.group_size

    return grouped_rollouts
