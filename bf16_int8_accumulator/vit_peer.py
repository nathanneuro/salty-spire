"""Tiny ViT with PEER (Parameter Efficient Expert Retrieval) FFN layers.

PEER replaces the standard transformer FFN with a top-k routed mixture of
"single neuron" experts. Each expert i is the pair (w_in[i, :], w_out[i, :]).
For input x, we:
  1. Score experts by q(x) · key_i
  2. Select top-k experts and softmax their scores → gates
  3. Apply only those experts: output = sum_k gate_i * GELU(x·w_in[i]) * w_out[i]

For testing we compute all expert pre-activations densely and mask to top-k.
This is wasteful but simple and fine for small num_experts.

The w_in and w_out matrices of PEER (which dominate the parameter count) go
through AccumLinear so we can test bf16+int8 master storage with int4 forward
quantization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from accum_linear import AccumLinear
from quant import quantize_int4_ste


class FFNLayer(nn.Module):
    """Standard 2-layer MLP, wraps AccumLinear for precision tests."""
    def __init__(self, dim: int, hidden: int, quant: str, master: str):
        super().__init__()
        self.fc1 = AccumLinear(dim, hidden, bias=True, quant=quant, master=master)
        self.fc2 = AccumLinear(hidden, dim, bias=True, quant=quant, master=master)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class PEERLayer(nn.Module):
    """Dense PEER (for debugging). Computes all expert activations then masks.

    Kept for comparison. For real sparse PEER use SparsePEERLayer.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        quant: str = "int4",
        master: str = "accum",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.q_proj = nn.Linear(d_model, d_model)
        self.expert_keys = nn.Parameter(
            torch.randn(num_experts, d_model) / math.sqrt(d_model)
        )
        self.w_in = AccumLinear(
            d_model, num_experts, bias=False, quant=quant, master=master
        )
        self.w_out = AccumLinear(
            num_experts, d_model, bias=False, quant=quant, master=master
        )
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        scores = F.linear(q, self.expert_keys)

        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_gates = F.softmax(topk_scores, dim=-1)

        hidden = self.w_in(x)
        gate = torch.zeros_like(hidden)
        gate.scatter_(-1, topk_idx, topk_gates.to(gate.dtype))
        hidden = F.gelu(hidden) * gate
        return self.w_out(hidden)


def _switch_load_balance_loss(
    scores: torch.Tensor,          # (N, E)  router logits
    topk_idx: torch.Tensor,        # (N, K)
    num_experts: int,
) -> torch.Tensor:
    """Switch Transformer load-balancing auxiliary loss.

    aux = E * <f, P>
      f[i] = fraction of top-k slots assigned to expert i
      P[i] = average router probability for expert i
    Minimized when both are uniform (= 1/E), giving aux = 1.
    """
    N, K = topk_idx.shape
    one_hot = F.one_hot(topk_idx, num_experts).float()      # (N, K, E)
    f = one_hot.sum(dim=(0, 1)) / (N * K)                   # (E,)
    P = F.softmax(scores, dim=-1).mean(dim=0)               # (E,)
    return num_experts * (f * P).sum()


class SparsePEERLayer(nn.Module):
    """True sparse PEER: each expert is a single neuron (rank-1 FFN slice).

    For each token x:
      1. Score all experts: s = q(x) · key_i
      2. Pick top-k: gates = softmax(top-k of s)
      3. Only evaluate the selected experts:
           h_k = GELU(x · expert_in[idx_k]) * gate_k
           out = sum_k h_k * expert_out[idx_k]
      4. Accumulate Switch-style load-balancing loss in self.last_aux_loss

    Compute cost is O(N * top_k * d_model) for expert eval, vs
    O(N * num_experts * d_model) for dense masked PEER. Memory for expert
    weights is still O(num_experts * d_model) regardless.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int,
        quant: str = "int4",
        master: str = "accum",
    ):
        super().__init__()
        assert top_k < num_experts, "sparse PEER needs top_k < num_experts"
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.quant = quant

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.expert_keys = nn.Parameter(
            torch.randn(num_experts, d_model) / math.sqrt(d_model)
        )
        # AccumLinear(d_model, num_experts, bias=False) has weight of shape
        # (num_experts, d_model), which is exactly "one d_model vector per expert".
        # We'll bypass its forward() and index into .weight directly.
        self.expert_in = AccumLinear(
            d_model, num_experts, bias=False, quant=quant, master=master
        )
        self.expert_out = AccumLinear(
            d_model, num_experts, bias=False, quant=quant, master=master
        )

        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        N = B * S
        x_flat = x.reshape(N, D)

        # ---- routing ----
        q = self.q_proj(x).reshape(N, D)
        scores = F.linear(q, self.expert_keys)              # (N, E)
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)  # (N, K)
        topk_gates = F.softmax(topk_scores, dim=-1)         # (N, K)

        # ---- load balancing loss (Switch) ----
        self.last_aux_loss = _switch_load_balance_loss(
            scores, topk_idx, self.num_experts
        )

        # ---- gather expert weights (with int4 STE quantization) ----
        w_in_full = self.expert_in.weight                   # (E, D)
        w_out_full = self.expert_out.weight                 # (E, D)
        if self.quant == "int4":
            w_in_full = quantize_int4_ste(w_in_full)
            w_out_full = quantize_int4_ste(w_out_full)

        w_in_sel = w_in_full[topk_idx]                      # (N, K, D)
        w_out_sel = w_out_full[topk_idx]                    # (N, K, D)

        # ---- sparse expert compute ----
        # h[n, k] = x[n] · w_in[topk_idx[n, k]]
        h = torch.einsum("nd,nkd->nk", x_flat, w_in_sel)    # (N, K)
        h = F.gelu(h) * topk_gates                          # (N, K)
        # out[n] = sum_k h[n, k] * w_out[topk_idx[n, k]]
        out = torch.einsum("nk,nkd->nd", h, w_out_sel)      # (N, D)

        return out.reshape(B, S, D)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, quant: str, master: str):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = AccumLinear(dim, dim * 3, bias=False, quant=quant, master=master)
        self.proj = AccumLinear(dim, dim, bias=True, quant=quant, master=master)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .view(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        top_k: int,
        quant: str,
        master: str,
        mlp_type: str = "peer",
        mlp_hidden_mult: int = 4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, quant, master)
        self.norm2 = nn.LayerNorm(dim)
        if mlp_type == "peer":
            self.mlp = PEERLayer(dim, num_experts, top_k, quant, master)
        elif mlp_type == "sparse_peer":
            self.mlp = SparsePEERLayer(dim, num_experts, top_k, quant, master)
        elif mlp_type == "ffn":
            self.mlp = FFNLayer(dim, dim * mlp_hidden_mult, quant, master)
        else:
            raise ValueError(f"unknown mlp_type={mlp_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PeerViT(nn.Module):
    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        num_classes: int = 10,
        dim: int = 48,
        depth: int = 2,
        num_heads: int = 4,
        num_experts: int = 128,
        top_k: int = 8,
        in_chans: int = 3,
        quant: str = "int4",
        master: str = "accum",
        mlp_type: str = "peer",
        mlp_hidden_mult: int = 4,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = in_chans * patch_size * patch_size

        self.patch_embed = AccumLinear(patch_dim, dim, quant=quant, master=master)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches + 1, dim) * 0.02
        )

        self.blocks = nn.ModuleList(
            [
                Block(dim, num_heads, num_experts, top_k, quant, master,
                      mlp_type=mlp_type, mlp_hidden_mult=mlp_hidden_mult)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = AccumLinear(dim, num_classes, quant=quant, master=master)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.patch_size
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)
        x = x.contiguous().view(B, C, -1, ps, ps).permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(B, self.n_patches, C * ps * ps)

        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return self.head(x[:, 0])

    def aux_loss(self) -> torch.Tensor:
        """Sum of load-balancing aux losses from all PEER layers."""
        total = None
        for m in self.modules():
            if isinstance(m, (PEERLayer, SparsePEERLayer)):
                total = m.last_aux_loss if total is None else total + m.last_aux_loss
        return total if total is not None else torch.tensor(0.0)


def sync_all(model: nn.Module) -> None:
    """Sync all AccumLinear weights from their master storage."""
    for m in model.modules():
        if isinstance(m, AccumLinear):
            m.sync_from_master()


def step_all(model: nn.Module, lr: float) -> None:
    """Plain SGD step: AccumLinear weights through the master, others direct."""
    accum_weight_ids = set()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, AccumLinear):
                m.apply_update(lr)
                accum_weight_ids.add(id(m.weight))
                if m.bias is not None and m.bias.grad is not None:
                    m.bias.sub_(lr * m.bias.grad)
                    m.bias.grad.zero_()
        for p in model.parameters():
            if id(p) not in accum_weight_ids and p.grad is not None:
                p.sub_(lr * p.grad)
                p.grad.zero_()


def count_params(model: nn.Module) -> dict:
    """Break down parameters by category."""
    n_accum_weight = 0
    n_accum_bias = 0
    n_other = 0
    for m in model.modules():
        if isinstance(m, AccumLinear):
            n_accum_weight += m.weight.numel()
            if m.bias is not None:
                n_accum_bias += m.bias.numel()
    all_params = sum(p.numel() for p in model.parameters())
    n_other = all_params - n_accum_weight - n_accum_bias
    return {
        "accum_weights": n_accum_weight,
        "accum_biases": n_accum_bias,
        "other": n_other,
        "total": all_params,
    }
