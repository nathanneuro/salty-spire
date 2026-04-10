"""
BF16 + Int8 Residual Accumulator

Stores weights as (bf16_value, int8_residual) pairs, where the true value is:

    w_true = w_bf16 + (r_int8 / 256) * ulp(w_bf16)

This gives ~11 bits of effective mantissa (vs bf16's 7) in 3 bytes (vs fp32's 4).

Key design choice: RTN (round-to-nearest) for the bf16 quantization, SR only for
the int8 residual quantization. This keeps residuals in [-0.5, +0.5] ULP (always
fits int8) and puts SR noise on a 1/256 ULP scale instead of a full ULP scale.
"""

import torch


def bf16_ulp(w_bf16: torch.Tensor) -> torch.Tensor:
    """Compute the unit-in-the-last-place for each bf16 value.

    ulp(x) = the gap between x and the next representable bf16 value.
    For bf16 with 7 mantissa bits: ulp(x) = 2^(exponent(x) - 7).

    We compute this in float32 to avoid precision issues.
    """
    w = w_bf16.float()
    # torch.frexp: x = mantissa * 2^exp, |mantissa| in [0.5, 1)
    # Actual exponent of x is (exp - 1), ulp = 2^(exp - 1 - 7) = 2^(exp - 8)
    _, exp = torch.frexp(w)
    ulp = torch.ldexp(torch.ones_like(w), exp - 8)
    # Handle zero: ulp(0) = smallest bf16 subnormal = 2^(-133)
    ulp = torch.where(w == 0, torch.tensor(2.0**-133, dtype=torch.float32), ulp)
    return ulp


def reconstruct_fp32(w_bf16: torch.Tensor, r_int8: torch.Tensor) -> torch.Tensor:
    """Reconstruct the approximate fp32 value from bf16 + int8 residual."""
    ulp = bf16_ulp(w_bf16)
    return w_bf16.float() + (r_int8.float() / 256.0) * ulp


def accumulate_with_residual(
    w_bf16: torch.Tensor,
    r_int8: torch.Tensor,
    grad: torch.Tensor,
) -> None:
    """Accumulate gradient into (bf16, int8_residual) pair in-place.

    Algorithm:
    1. Reconstruct current true value from bf16 + residual
    2. Add gradient in fp32
    3. Round to nearest bf16 (deterministic RTN)
    4. Compute exact sub-ULP residual in fp32
    5. Stochastically round residual to int8 (SR noise on 1/256 ULP scale)
    """
    # Step 1: reconstruct current value in fp32
    ulp_old = bf16_ulp(w_bf16)
    w_true = w_bf16.float() + (r_int8.float() / 256.0) * ulp_old

    # Step 2: add gradient
    w_new = w_true + grad.float()

    # Step 3: deterministic round-to-nearest bf16
    w_bf16_new = w_new.bfloat16()

    # Step 4: compute residual in units of (1/256 * ulp)
    ulp_new = bf16_ulp(w_bf16_new)
    residual_units = (w_new - w_bf16_new.float()) / ulp_new * 256.0
    # With RTN, residual_units is in [-128, +128] (half a ULP each direction)

    # Step 5: stochastic rounding of residual to int8
    # This keeps E[r_int8] = residual_units, maintaining unbiasedness
    # while adding noise only on the 1/256 ULP scale
    r_floor = residual_units.floor()
    frac = residual_units - r_floor
    rand = torch.rand_like(frac)
    r_sr = torch.where(rand < frac, r_floor + 1, r_floor)
    r_new = r_sr.clamp(-128, 127).to(torch.int8)

    # Write back in-place
    w_bf16.copy_(w_bf16_new)
    r_int8.copy_(r_new)


def accumulate_with_residual_sr_bf16(
    w_bf16: torch.Tensor,
    r_int8: torch.Tensor,
    grad: torch.Tensor,
) -> None:
    """Original scheme: SR on bf16, then compute residual.

    Kept for comparison. This has higher variance because SR noise is on the
    full ULP scale, and residuals can overflow int8 range → clamping → info loss.
    """
    ulp_old = bf16_ulp(w_bf16)
    w_true = w_bf16.float() + (r_int8.float() / 256.0) * ulp_old
    w_new = w_true + grad.float()

    # SR on bf16
    w_bf16_rtn = w_new.bfloat16()
    w_rtn_f32 = w_bf16_rtn.float()
    ulp_rtn = bf16_ulp(w_bf16_rtn)

    went_up = w_rtn_f32 > w_new
    went_down = w_rtn_f32 < w_new
    exact = ~went_up & ~went_down

    w_lo = torch.where(went_up, w_rtn_f32 - ulp_rtn, w_rtn_f32)
    w_hi = torch.where(went_down, w_rtn_f32 + ulp_rtn, w_rtn_f32)

    span = w_hi - w_lo
    frac = torch.where(span > 0, (w_new - w_lo) / span, torch.zeros_like(w_new))
    frac = frac.clamp(0.0, 1.0)

    rand = torch.rand_like(w_new)
    w_bf16_sr = torch.where(rand < frac, w_hi, w_lo).bfloat16()
    w_bf16_sr = torch.where(exact, w_bf16_rtn, w_bf16_sr)

    ulp_final = bf16_ulp(w_bf16_sr)
    residual_fp32 = w_new - w_bf16_sr.float()
    r_new = (residual_fp32 / ulp_final * 256.0).round().clamp(-128, 127).to(torch.int8)

    w_bf16.copy_(w_bf16_sr)
    r_int8.copy_(r_new)


def naive_bf16_accumulate(w_bf16: torch.Tensor, grad: torch.Tensor) -> None:
    """Baseline: just add gradient in bf16 (lossy)."""
    w_bf16.copy_((w_bf16.float() + grad.float()).bfloat16())
