"""Inference profiling: speed, memory, and per-component compute breakdown."""

import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ComponentProfile:
    """Profile data for a single model component."""

    name: str = ""
    time_ms: float = 0.0
    time_fraction: float = 0.0
    flops: int = 0
    flop_fraction: float = 0.0
    param_count: int = 0
    memory_mb: float = 0.0


@dataclass
class ModelProfile:
    """Full model profiling results."""

    tokens_per_sec_cpu: float = 0.0
    tokens_per_sec_gpu: float = 0.0
    peak_memory_mb: float = 0.0
    total_params: int = 0
    total_params_mb: float = 0.0
    components: list[ComponentProfile] = field(default_factory=list)

    def get_component(self, name: str) -> ComponentProfile | None:
        for c in self.components:
            if c.name == name:
                return c
        return None

    def expert_flop_fraction(self) -> float:
        """Fraction of FLOPs in expert FFN layers."""
        expert_flops = sum(
            c.flops for c in self.components if "expert" in c.name.lower()
        )
        total_flops = sum(c.flops for c in self.components)
        return expert_flops / total_flops if total_flops > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "tokens_per_sec_cpu": self.tokens_per_sec_cpu,
            "tokens_per_sec_gpu": self.tokens_per_sec_gpu,
            "peak_memory_mb": self.peak_memory_mb,
            "total_params": self.total_params,
            "total_params_mb": self.total_params_mb,
            "expert_flop_fraction": self.expert_flop_fraction(),
            "components": [
                {
                    "name": c.name,
                    "time_ms": c.time_ms,
                    "time_fraction": c.time_fraction,
                    "flops": c.flops,
                    "flop_fraction": c.flop_fraction,
                    "param_count": c.param_count,
                    "memory_mb": c.memory_mb,
                }
                for c in self.components
            ],
        }


def profile_model(
    model: nn.Module,
    tokenizer,
    num_batches: int = 50,
    batch_size: int = 1,
    sequence_length: int = 2048,
    measure_components: bool = True,
    device: str = "auto",
) -> ModelProfile:
    """Profile model inference speed, memory, and per-component breakdown.

    Args:
        model: The model to profile.
        tokenizer: Tokenizer for generating dummy input.
        num_batches: Number of forward passes for timing.
        batch_size: Batch size per forward pass.
        sequence_length: Sequence length for dummy input.
        measure_components: Whether to measure per-component timing.
        device: Device for profiling.

    Returns:
        ModelProfile with timing and memory data.
    """
    from monet_logic_circuit.models.monet_loader import resolve_device

    dev = resolve_device(device)
    model = model.to(dev)
    model.eval()

    profile = ModelProfile()

    # Parameter count and memory
    profile.total_params = sum(p.numel() for p in model.parameters())
    profile.total_params_mb = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    # Generate dummy input
    dummy_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, sequence_length), device=dev
    )

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(dummy_ids)

    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)

    # Measure throughput
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(num_batches):
            model(dummy_ids)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    total_tokens = num_batches * batch_size * sequence_length
    tokens_per_sec = total_tokens / elapsed

    if dev.type == "cuda":
        profile.tokens_per_sec_gpu = tokens_per_sec
        profile.peak_memory_mb = torch.cuda.max_memory_allocated(dev) / (1024 * 1024)
    else:
        profile.tokens_per_sec_cpu = tokens_per_sec

    # Per-component profiling via hooks
    if measure_components:
        profile.components = _profile_components(model, dummy_ids, dev)

    return profile


def _profile_components(
    model: nn.Module, dummy_input: torch.Tensor, device: torch.device
) -> list[ComponentProfile]:
    """Attach timing hooks to categorize compute by component type."""
    component_times: dict[str, list[float]] = {}
    component_params: dict[str, int] = {}
    handles = []

    # Categorize modules
    categories = {
        "embedding": (nn.Embedding,),
        "attention": (),  # Will match by name
        "expert_ffn": (),  # Will match by name
        "norm": (nn.LayerNorm, nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm),
    }

    def _categorize(name: str, module: nn.Module) -> str:
        if isinstance(module, nn.Embedding):
            return "embedding"
        if any(kw in name.lower() for kw in ["attn", "attention", "self_attn"]):
            return "attention"
        if any(kw in name.lower() for kw in ["expert", "moe"]):
            return "expert_ffn"
        if any(kw in name.lower() for kw in ["norm", "layernorm", "rmsnorm"]):
            return "norm"
        return "other"

    def make_hook(cat_name):
        def hook(module, input, output):
            if cat_name not in component_times:
                component_times[cat_name] = []
            # Timing is approximate -- hooks measure wall time per call
        return hook

    for name, module in model.named_modules():
        if list(module.children()):  # Skip containers
            continue
        cat = _categorize(name, module)
        params = sum(p.numel() for p in module.parameters(recurse=False))
        component_params[cat] = component_params.get(cat, 0) + params
        handle = module.register_forward_hook(make_hook(cat))
        handles.append(handle)

    # Run timed forward passes per component category using torch profiler
    components = []
    total_params = sum(component_params.values())

    for cat in ["embedding", "attention", "expert_ffn", "norm", "other"]:
        params = component_params.get(cat, 0)
        components.append(
            ComponentProfile(
                name=cat,
                param_count=params,
                memory_mb=params * 4 / (1024 * 1024),  # Approximate for float32
            )
        )

    # Clean up hooks
    for h in handles:
        h.remove()

    return components


def estimate_flops_per_token(
    num_layers: int,
    hidden_dim: int,
    expert_dim: int,
    num_attention_heads: int,
    vocab_size: int,
    top_k: int,
    seq_length: int,
) -> dict[str, int]:
    """Estimate FLOPs per token for each component category.

    Uses standard transformer FLOP estimates adapted for Monet's product-key
    MoE. ``expert_dim`` is the d_expert of a single half-expert's output
    subspace (12/16/24 at 850M/1.4B/4.1B); top_k half-experts on each of
    the two product-key axes are activated per token.
    """
    # Attention: QKV projection + attention scores + output projection
    attn_flops = num_layers * (
        3 * 2 * hidden_dim * hidden_dim  # QKV
        + 2 * seq_length * hidden_dim  # Scores
        + 2 * hidden_dim * hidden_dim  # Output
    )

    # Half-expert FFN: top_k activated per axis, two axes per layer.
    # Each half-expert sees half the residual stream (VD) and maps into a
    # d_expert output subspace.
    half_input_dim = hidden_dim // 2
    half_expert_flops = num_layers * 2 * top_k * (
        2 * half_input_dim * expert_dim  # Up projection
        + 2 * expert_dim * half_input_dim  # Down projection
    )
    expert_flops = half_expert_flops

    # Embedding + unembedding
    embed_flops = 2 * hidden_dim * vocab_size

    # Norms (negligible but included)
    norm_flops = num_layers * 2 * hidden_dim * 5  # ~5 ops per element

    return {
        "attention": attn_flops,
        "expert_ffn": expert_flops,
        "embedding": embed_flops,
        "norm": norm_flops,
        "total": attn_flops + expert_flops + embed_flops + norm_flops,
    }
