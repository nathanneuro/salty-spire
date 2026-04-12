"""Symmetric int4 quantization with straight-through estimator (STE).

For weight tensors shaped (out_features, in_features), quantizes per output row:
each row gets its own scale, values go to int4 signed range [-7, 7] (15 levels,
skipping -8 for symmetry).
"""

import torch


def quantize_int4_ste(w: torch.Tensor) -> torch.Tensor:
    """Per-row symmetric int4 quantization with STE.

    Forward: returns the quantized weight.
    Backward: gradient passes through as identity.
    """
    w_detached = w.detach()
    abs_max = w_detached.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 7.0).clamp(min=1e-8)
    w_q = (w_detached / scale).round().clamp(-7, 7) * scale
    # STE trick: forward = w_q, backward = identity
    return w + (w_q - w).detach()


def quantize_int4_hard(w: torch.Tensor) -> torch.Tensor:
    """Same as above but without STE (for inference)."""
    abs_max = w.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / 7.0).clamp(min=1e-8)
    return (w / scale).round().clamp(-7, 7) * scale
