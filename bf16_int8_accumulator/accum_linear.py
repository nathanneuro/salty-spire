"""Linear layer with configurable master-weight storage.

The `master` controls how weights are stored between forward passes:
  - 'fp32'      : standard fp32 nn.Parameter (baseline)
  - 'accum'     : bf16 + int8 residual accumulator (the scheme under test)
  - 'naive_bf16': plain bf16 (lossy baseline)

The `quant` controls whether the forward pass applies int4 quantization (STE)
to the weight. This is independent of master storage.

Training flow per step:
  1. `sync_from_master()` copies master → live fp32 `self.weight`
  2. forward pass uses self.weight (optionally int4-quantized via STE)
  3. backward populates self.weight.grad
  4. `apply_update(lr)` pushes -lr*grad back into the master
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from accumulator import (
    accumulate_with_residual,
    reconstruct_fp32,
    naive_bf16_accumulate,
)
from quant import quantize_int4_ste


class AccumLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant: str = "int4",
        master: str = "accum",
    ):
        super().__init__()
        assert quant in ("int4", "none")
        assert master in ("fp32", "accum", "naive_bf16")

        self.in_features = in_features
        self.out_features = out_features
        self.quant = quant
        self.master = master

        w = torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5

        if master == "accum":
            self.register_buffer("w_bf16", w.bfloat16().clone())
            self.register_buffer(
                "r_int8",
                torch.zeros(out_features, in_features, dtype=torch.int8),
            )
        elif master == "naive_bf16":
            self.register_buffer("w_bf16_naive", w.bfloat16().clone())

        # Live fp32 view used for forward/backward. For master='fp32' this IS
        # the master; for other modes it is synced each step.
        self.weight = nn.Parameter(w.clone())
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.sync_from_master()

    def sync_from_master(self) -> None:
        if self.master == "fp32":
            return
        with torch.no_grad():
            if self.master == "accum":
                self.weight.copy_(reconstruct_fp32(self.w_bf16, self.r_int8))
            else:  # naive_bf16
                self.weight.copy_(self.w_bf16_naive.float())

    def apply_update(self, lr: float) -> None:
        if self.weight.grad is None:
            return
        with torch.no_grad():
            delta = -lr * self.weight.grad
            if self.master == "accum":
                accumulate_with_residual(self.w_bf16, self.r_int8, delta)
            elif self.master == "naive_bf16":
                naive_bf16_accumulate(self.w_bf16_naive, delta)
            else:  # fp32
                self.weight.add_(delta)
            self.weight.grad.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self.quant == "int4":
            w = quantize_int4_ste(w)
        return F.linear(x, w, self.bias)
