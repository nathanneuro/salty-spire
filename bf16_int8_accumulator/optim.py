"""Adam optimizer that routes AccumLinear weight updates through master storage.

Design:
  - Adam moments (m, v) are always fp32 (we isolate the test to weight storage)
  - For AccumLinear weights: the Adam delta goes into the master storage
    (bf16+int8 accumulator, naive bf16, or fp32) instead of directly updating
    the live `self.weight`. The live weight will be re-synced from master before
    the next forward pass.
  - For all other parameters (LayerNorm, pos_embed, biases, small fp32 params):
    standard Adam in-place update.
"""

import torch

from accum_linear import AccumLinear
from accumulator import accumulate_with_residual, naive_bf16_accumulate


class AccumAdam:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.model = model
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0

        self.accum_linears = [
            m for m in model.modules() if isinstance(m, AccumLinear)
        ]
        accum_weight_ids = {id(m.weight) for m in self.accum_linears}

        # Everything else (LN, cls_token, pos_embed, expert_keys, biases of
        # AccumLinear, q_proj weight/bias, etc.) is a plain param.
        plain = []
        seen = set()
        for p in model.parameters():
            if id(p) in accum_weight_ids:
                continue
            if id(p) in seen:
                continue
            seen.add(id(p))
            plain.append(p)
        self.plain_params = plain

        self.state = {}
        for m in self.accum_linears:
            self.state[id(m.weight)] = {
                "m": torch.zeros_like(m.weight),
                "v": torch.zeros_like(m.weight),
            }
        for p in self.plain_params:
            self.state[id(p)] = {
                "m": torch.zeros_like(p),
                "v": torch.zeros_like(p),
            }

    def zero_grad(self):
        for m in self.accum_linears:
            if m.weight.grad is not None:
                m.weight.grad.zero_()
        for p in self.plain_params:
            if p.grad is not None:
                p.grad.zero_()

    def _adam_delta(self, grad, state, bc1, bc2):
        state["m"].mul_(self.b1).add_(grad, alpha=1 - self.b1)
        state["v"].mul_(self.b2).addcmul_(grad, grad, value=1 - self.b2)
        m_hat = state["m"] / bc1
        v_hat = state["v"] / bc2
        return -self.lr * m_hat / (v_hat.sqrt() + self.eps)

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t

        # AccumLinear weights: Adam delta routed through master storage
        for mod in self.accum_linears:
            if mod.weight.grad is None:
                continue
            g = mod.weight.grad
            delta = self._adam_delta(g, self.state[id(mod.weight)], bc1, bc2)
            if mod.master == "accum":
                accumulate_with_residual(mod.w_bf16, mod.r_int8, delta)
            elif mod.master == "naive_bf16":
                naive_bf16_accumulate(mod.w_bf16_naive, delta)
            else:  # fp32
                mod.weight.add_(delta)
            mod.weight.grad.zero_()

        # Everything else: plain Adam
        for p in self.plain_params:
            if p.grad is None:
                continue
            delta = self._adam_delta(p.grad, self.state[id(p)], bc1, bc2)
            p.add_(delta)
            p.grad.zero_()
