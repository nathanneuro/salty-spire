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


class FFNLayer(nn.Module):
    """Standard 2-layer MLP, wraps AccumLinear for precision tests."""
    def __init__(self, dim: int, hidden: int, quant: str, master: str):
        super().__init__()
        self.fc1 = AccumLinear(dim, hidden, bias=True, quant=quant, master=master)
        self.fc2 = AccumLinear(hidden, dim, bias=True, quant=quant, master=master)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class PEERLayer(nn.Module):
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

        # Query projection and retrieval keys stay in fp32 (small)
        self.q_proj = nn.Linear(d_model, d_model)
        self.expert_keys = nn.Parameter(
            torch.randn(num_experts, d_model) / math.sqrt(d_model)
        )

        # The large expert weight matrices -- these get int4 + accumulator
        self.w_in = AccumLinear(
            d_model, num_experts, bias=False, quant=quant, master=master
        )
        self.w_out = AccumLinear(
            num_experts, d_model, bias=False, quant=quant, master=master
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        q = self.q_proj(x)
        scores = F.linear(q, self.expert_keys)  # (B, S, num_experts)

        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)
        topk_gates = F.softmax(topk_scores, dim=-1)

        # Dense expert pre-activations, then mask to top-k
        hidden = self.w_in(x)  # (B, S, num_experts)
        gate = torch.zeros_like(hidden)
        gate.scatter_(-1, topk_idx, topk_gates.to(gate.dtype))
        hidden = F.gelu(hidden) * gate

        return self.w_out(hidden)


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
        # Patchify (B,C,H,W) -> (B, n_patches, C*ps*ps)
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)  # (B,C,H/ps,W/ps,ps,ps)
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
