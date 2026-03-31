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


@dataclass
class ScoredSample:
    prompt: str
    target: int
    completion_ids: list[int]
    completion_text: str
    reward: float
    advantage: float
    stopped_by_eos: bool


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

    # Each rollout in the list contains the prompt as well as its G completions and target
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

# Group reward normalization
def compute_group_advantages(rollouts, eps):
    scored_samples = []

    for rollout in rollouts:
        # For each prompt group, collect its G rewards
        rewards = torch.tensor([sample.reward for sample in rollout.samples], dtype=torch.float32)
        # Compute mean, std inside this group
        reward_mean = rewards.mean()
        reward_std = rewards.std(unbiased=False)

        # convert each reward into a relative advantage
        normalized_advantages = (rewards - reward_mean) / (reward_std + eps)

        # flatten the results into a list of ScoredSample
        for sample, advantage in zip(rollout.samples, normalized_advantages):
            scored_samples.append(
                ScoredSample(
                    prompt=rollout.prompt,
                    target=rollout.target,
                    completion_ids=sample.completion_ids,
                    reward=sample.reward,
                    advantage=advantage.item(),
                    stopped_by_eos=sample.stopped_by_eos,
                )
            )
    return scored_samples

# Helper function to build prompt + completion sequences
def build_completion_training_batch(scored_samples, tokenizer):
    input_id_rows = []
    attention_mask_rows = []
    completion_mask_rows = []

    for sample in scored_samples:
        # Combine ids for both the prompt and completion tokens
        prompt_ids = tokenizer(
            sample.prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=GRPO.max_prompt_length,
        )['input_ids']

        completion_ids = sample.completion_ids[: GRPO.max_completion_length]
        full_ids = prompt_ids + completion_ids

        attention_mask = [1] * len(full_ids)
        # To ensure that we are only learning from completion tokens, we mask out prompt tokens
        completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

        input_id_rows.append(torch.tensor(full_ids, dtype=torch.long))
        attention_mask_rows.append(torch.tensor(attention_mask, dtype=torch.long))
        completion_mask_rows.append(torch.tensor(completion_mask, dtype=torch.long))

    input_ids = torch.nn.utils.rnn.pad_sequence(
            input_id_rows,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_rows,
            batch_first=True,
            padding_value=0,
    )
    completion_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_rows,
            batch_first=True,
            padding_value=0,
    )

    return {
        "input_ids": input_ids.to(RUNTIME.device),
        "attention_mask": attention_mask.to(RUNTIME.device),
        "completion_mask": completion_mask.to(RUNTIME.device)
    }

# Helper function to compute completion-only token logprobs
def compute_completion_logprobs(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"]
    )

    # Since language models predict token t from the tokens before it, we shift:
    logits = outputs.logits[:, :-1, :]
    target_ids = batch["input_ids"][:, 1:]
    target_completion_mask = batch["completion_mask"][:, 1:]

    # Gather the log prob of the actual next token
    logprobs = torch.log_softmax(logits, dim=-1)
    selected_logprobs = logprobs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

    # Return token-level logprobs for every predicted token, along with
    # a mask saying which of those belong to the completion
    return selected_logprobs, target_completion_mask

# Helper function to average only over completion tokens
def masked_mean(values, mask, dim=-1, eps: float = 1e-8):
    mask = mask.float()
    # Average only over positions where mask is 1 (completion-token positions)
    return (values * mask).sum(dim=dim) / (mask.sum(dim=dim) + eps)

# GRPO loss function
def compute_grpo_loss(policy_model, reference_model, tokenizer, rollouts):
    # Compute advantages
    scored_samples = compute_group_advantages(rollouts)
    batch = build_completion_training_batch(scored_samples, tokenizer)

    # Recompute policy logprobs, giving one scalar per sampled completion
    # Average logprob over just the generated tokens
    policy_token_logprobs, completion_mask = compute_completion_logprobs(policy_model, batch)
    policy_sequence_logprobs = masked_mean(policy_token_logprobs, completion_mask)

    advantages = torch.tensor(
        [sample.advantage for sample in scored_samples],
        dtype=policy_sequence_logprobs.dtype,
        device=policy_sequence_logprobs.device,
    )

    # Positive advantage should incraese the possibility of the sampled completion 
    # Negative advantage should decrease it
    policy_loss = -(advantages * policy_sequence_logprobs).mean()

    metrics = {
        "policy_loss": policy_loss.item(),
        "mean_advantage": advantages.mean().item(),
        "mean_reward": sum(sample.reward for sample in scored_samples) / len(scored_samples),
    }

    # If no reference model we do not enforce KL constraint
    if reference_model is None:
        return policy_loss, metrics
    
    # Compute logprobs with reference model
    with torch.no_grad():
        reference_token_logprobs, _ = compute_completion_logprobs(reference_model, batch)
    
    # KL stabilization preventing current policy from drifting too agressively from the frozen reference
    token_logprob_diff = policy_token_logprobs - reference_token_logprobs
    kl_per_sequence = masked_mean(token_logprob_diff, completion_mask)
    kl_loss = GRPO.kl_beta * kl_per_sequence.mean()

    total_loss = policy_loss + kl_loss

    metrics["kl_loss"] = kl_loss.item()
    metrics["mean_kl"] = kl_per_sequence.mean()
    metrics["total_loss"] = total_loss.item()

    return total_loss, metrics
