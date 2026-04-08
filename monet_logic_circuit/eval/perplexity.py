"""Perplexity evaluation on held-out data."""

import math

import torch
import torch.nn as nn
from tqdm import tqdm


def compute_perplexity(
    model: nn.Module,
    tokenized_data: torch.Tensor,
    stride: int = 512,
    max_length: int = 2048,
    device: torch.device | str = "cuda",
    batch_size: int = 1,
) -> dict:
    """Compute perplexity on a tokenized sequence using sliding window.

    Args:
        model: Language model with a forward that returns logits or a
            CausalLMOutput with .logits attribute.
        tokenized_data: 1-D tensor of token IDs.
        stride: Sliding window stride.
        max_length: Maximum context length per window.
        device: Device for computation.
        batch_size: Number of windows to process in parallel.

    Returns:
        Dict with 'perplexity', 'loss_nats', 'loss_bits', 'num_tokens'.
    """
    model.eval()
    device = torch.device(device)

    seq_len = len(tokenized_data)
    nlls = []
    num_tokens = 0

    # Build sliding windows
    windows = []
    for begin in range(0, seq_len - max_length + 1, stride):
        end = begin + max_length
        input_ids = tokenized_data[begin:end].unsqueeze(0)
        target_start = max(0, begin - (begin // stride) * stride)
        # Only count loss on the new tokens (after the overlap)
        target_start = max_length - stride if begin > 0 else 0
        windows.append((input_ids, target_start))

    with torch.no_grad():
        for i in tqdm(range(0, len(windows), batch_size), desc="Perplexity"):
            batch_windows = windows[i : i + batch_size]
            input_ids = torch.cat([w[0] for w in batch_windows], dim=0).to(device)

            outputs = model(input_ids)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            for j, (_, target_start) in enumerate(batch_windows):
                shift_logits = logits[j, target_start:-1].contiguous()
                shift_labels = input_ids[j, target_start + 1 :].contiguous()

                loss = torch.nn.functional.cross_entropy(
                    shift_logits, shift_labels, reduction="sum"
                )
                nlls.append(loss.item())
                num_tokens += shift_labels.numel()

    total_nll = sum(nlls)
    avg_nll = total_nll / num_tokens
    perplexity = math.exp(avg_nll)

    return {
        "perplexity": perplexity,
        "loss_nats": avg_nll,
        "loss_bits": avg_nll / math.log(2),
        "num_tokens": num_tokens,
    }
