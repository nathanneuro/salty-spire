"""APOLLO-mini: memory-efficient Adam replacement.

Based on "APOLLO: SGD-like Memory, AdamW-level Performance" (Zhu et al. 2024).

Core idea: Adam's per-element normalization m/sqrt(v) is expensive in memory
(2x param count). APOLLO approximates the *per-tensor effective scaling* that
Adam applies by tracking moments in a low-rank projected subspace.

APOLLO-mini (this file): rank-1 projection + tensor-wise scale factor.

Algorithm per tensor W of shape (..., in_dim):
  state:
    proj: random unit vector of shape (in_dim,) -- fixed at init
    m:    rank-1 projected first moment, shape (...,) = W.shape[:-1]
    v:    rank-1 projected second moment, shape (...,)
  step:
    g_proj = (g * proj).sum(-1)                     # (...,)
    m = b1 * m + (1-b1) * g_proj
    v = b2 * v + (1-b2) * g_proj**2
    u_proj = m_hat / (sqrt(v_hat) + eps)            # Adam direction in subspace
    scale  = ||u_proj|| / (||g_proj|| + eps)        # tensor-wise scale factor
    delta  = -lr * scale * g                        # apply to full gradient

Memory per param: |W.shape[:-1]| * 2 + in_dim
  vs Adam's:      |W| * 2
  For W of shape (out, in): 2*out + in vs 2*out*in → ~in/2 smaller

For 1D params (biases, norms), we fall back to standard Adam because the
rank-1 projection is degenerate.

Also supports routing AccumLinear weight updates through their master storage
(bf16+int8 accumulator, naive bf16, or fp32).
"""

import torch

from accum_linear import AccumLinear
from accumulator import accumulate_with_residual, naive_bf16_accumulate


class AccumApolloMini:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        scale_cap: float = 64.0,
        per_channel: bool = True,
        variant: str = "mini",   # 'mini' = rank-1 m & v, 'hybrid' = full m, rank-1 v
        aux_loss_weight: float = 0.01,
        proj_seed: int = 0,
    ):
        self.model = model
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.scale_cap = scale_cap
        self.per_channel = per_channel
        self.variant = variant
        self.aux_loss_weight = aux_loss_weight
        self.t = 0

        self.accum_linears = [
            m for m in model.modules() if isinstance(m, AccumLinear)
        ]
        accum_ids = {id(m.weight) for m in self.accum_linears}

        plain = []
        seen = set()
        for p in model.parameters():
            if id(p) in accum_ids or id(p) in seen:
                continue
            seen.add(id(p))
            plain.append(p)
        self.plain_params = plain

        self._g = torch.Generator().manual_seed(proj_seed)
        self.state = {}
        for mod in self.accum_linears:
            self.state[id(mod.weight)] = self._make_state(mod.weight)
        for p in self.plain_params:
            self.state[id(p)] = self._make_state(p)

    def _make_state(self, p: torch.Tensor):
        if p.dim() >= 2:
            in_dim = p.shape[-1]
            out_shape = p.shape[:-1]
            proj = torch.randn(in_dim, generator=self._g) / (in_dim ** 0.5)
            if self.variant == "mini":
                # APOLLO-mini: rank-1 projected m and v, tensor/per-row scale.
                return {
                    "type": "apollo_mini",
                    "proj": proj,
                    "m": torch.zeros(out_shape),
                    "v": torch.zeros(out_shape),
                }
            elif self.variant == "hybrid":
                # APOLLO-hybrid: full momentum m, rank-1 projected v.
                # Half of Adam's state, still matches Adam's direction closely.
                return {
                    "type": "apollo_hybrid",
                    "proj": proj,
                    "m": torch.zeros_like(p),           # full
                    "v": torch.zeros(out_shape),        # rank-1
                }
            else:
                raise ValueError(f"unknown variant {self.variant}")
        else:
            return {
                "type": "adam",
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

    def _compute_delta(self, g: torch.Tensor, st: dict, bc1: float, bc2: float):
        if st["type"] == "apollo_mini":
            # Rank-1 projection of gradient onto fixed random direction.
            g_proj = (g * st["proj"]).sum(dim=-1)
            st["m"].mul_(self.b1).add_(g_proj, alpha=1 - self.b1)
            st["v"].mul_(self.b2).addcmul_(g_proj, g_proj, value=1 - self.b2)
            m_hat = st["m"] / bc1
            v_hat = st["v"] / bc2
            u_proj = m_hat / (v_hat.sqrt() + self.eps)

            if self.per_channel:
                denom = g_proj.abs().clamp(min=self.eps)
                mag = (u_proj.abs() / denom).clamp(max=self.scale_cap)
                sgn = torch.sign(u_proj * g_proj)
                sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
                scale = (mag * sgn).unsqueeze(-1)
                return -self.lr * scale * g
            else:
                u_norm = u_proj.norm()
                g_norm = g_proj.norm().clamp(min=self.eps)
                scale = (u_norm / g_norm).clamp(max=self.scale_cap)
                return -self.lr * scale * g

        elif st["type"] == "apollo_hybrid":
            # Full first moment, rank-1 projected second moment.
            # Direction follows Adam exactly; per-row sqrt(v) is estimated from
            # a rank-1 projection onto a fixed random direction.
            st["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
            g_proj = (g * st["proj"]).sum(dim=-1)
            st["v"].mul_(self.b2).addcmul_(g_proj, g_proj, value=1 - self.b2)
            m_hat = st["m"] / bc1
            v_hat = st["v"] / bc2
            # sqrt(v) is per-row; broadcast along last dim.
            denom = (v_hat.sqrt() + self.eps).unsqueeze(-1)
            return -self.lr * m_hat / denom

        else:
            st["m"].mul_(self.b1).add_(g, alpha=1 - self.b1)
            st["v"].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = st["m"] / bc1
            v_hat = st["v"] / bc2
            return -self.lr * m_hat / (v_hat.sqrt() + self.eps)

    @torch.no_grad()
    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t

        for mod in self.accum_linears:
            if mod.weight.grad is None:
                continue
            delta = self._compute_delta(
                mod.weight.grad, self.state[id(mod.weight)], bc1, bc2
            )
            if mod.master == "accum":
                accumulate_with_residual(mod.w_bf16, mod.r_int8, delta)
            elif mod.master == "naive_bf16":
                naive_bf16_accumulate(mod.w_bf16_naive, delta)
            else:
                mod.weight.add_(delta)
            mod.weight.grad.zero_()

        for p in self.plain_params:
            if p.grad is None:
                continue
            delta = self._compute_delta(p.grad, self.state[id(p)], bc1, bc2)
            p.add_(delta)
            p.grad.zero_()

    def memory_bytes(self) -> dict:
        """Return optimizer-state memory usage (just state tensors, no params)."""
        total = 0
        apollo = 0
        adam = 0
        for st in self.state.values():
            if st["type"] == "apollo":
                b = (
                    st["proj"].numel()
                    + st["m"].numel()
                    + st["v"].numel()
                ) * 4  # fp32
                apollo += b
                total += b
            else:
                b = (st["m"].numel() + st["v"].numel()) * 4
                adam += b
                total += b
        return {"total": total, "apollo_tensors": apollo, "adam_1d": adam}
