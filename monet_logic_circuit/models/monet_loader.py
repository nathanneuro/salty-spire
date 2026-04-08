"""Load pretrained Monet checkpoints and provide a uniform interface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class MonetConfig:
    """Configuration for a loaded Monet model."""

    checkpoint_path: str = ""
    num_layers: int = 0
    hidden_dim: int = 0
    num_experts_per_layer: int = 0
    expert_hidden_dim: int = 0
    num_attention_heads: int = 0
    vocab_size: int = 0
    max_seq_length: int = 2048
    top_k: int = 2  # Number of experts selected per token
    device: str = "auto"

    # Populated after loading
    total_experts: int = field(init=False, default=0)

    def __post_init__(self):
        self.total_experts = self.num_layers * self.num_experts_per_layer


def resolve_device(device: str) -> torch.device:
    """Resolve 'auto' device string to actual device."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def load_monet_model(
    checkpoint: str,
    device: str = "auto",
    dtype: Optional[torch.dtype] = None,
) -> tuple[nn.Module, MonetConfig]:
    """Load a pretrained Monet model from a checkpoint path or HF hub ID.

    Args:
        checkpoint: Path to local checkpoint directory or HuggingFace hub ID.
        device: Device to load onto ('auto', 'cpu', 'cuda', 'cuda:0', etc.).
        dtype: Optional dtype override. If None, uses checkpoint's native dtype.

    Returns:
        Tuple of (model, config).

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist locally and isn't
            a valid HF hub ID.
        ValueError: If checkpoint format is unrecognized.
    """
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint)

    # Try local path first
    if checkpoint_path.exists():
        return _load_local_checkpoint(checkpoint_path, device, dtype)

    # Fall back to HuggingFace hub
    return _load_from_hub(checkpoint, device, dtype)


def _load_local_checkpoint(
    path: Path,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> tuple[nn.Module, MonetConfig]:
    """Load from a local checkpoint directory."""
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found in checkpoint directory: {path}"
        )

    import json

    with open(config_path) as f:
        raw_config = json.load(f)

    config = _parse_config(raw_config, str(path))

    # Load model weights
    model = _build_model_from_config(config)
    weight_files = sorted(path.glob("*.safetensors")) or sorted(path.glob("*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {path}")

    state_dict = _load_state_dict(weight_files)
    model.load_state_dict(state_dict, strict=False)

    if dtype is not None:
        model = model.to(dtype=dtype)
    model = model.to(device=device)
    model.eval()

    return model, config


def _load_from_hub(
    hub_id: str,
    device: torch.device,
    dtype: Optional[torch.dtype],
) -> tuple[nn.Module, MonetConfig]:
    """Load from HuggingFace hub."""
    try:
        from transformers import AutoModelForCausalLM, AutoConfig
    except ImportError as e:
        raise ImportError(
            "transformers is required for loading from HuggingFace hub"
        ) from e

    hf_config = AutoConfig.from_pretrained(hub_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hub_id,
        config=hf_config,
        torch_dtype=dtype,
        device_map=device if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()

    config = _extract_config_from_hf(hf_config, hub_id)
    return model, config


def _parse_config(raw: dict, checkpoint_path: str) -> MonetConfig:
    """Parse a raw config dict into MonetConfig."""
    return MonetConfig(
        checkpoint_path=checkpoint_path,
        num_layers=raw.get("num_hidden_layers", raw.get("n_layer", 0)),
        hidden_dim=raw.get("hidden_size", raw.get("d_model", 0)),
        num_experts_per_layer=raw.get("num_local_experts", raw.get("num_experts", 0)),
        expert_hidden_dim=raw.get("intermediate_size", raw.get("d_ff", 0)),
        num_attention_heads=raw.get("num_attention_heads", raw.get("n_head", 0)),
        vocab_size=raw.get("vocab_size", 0),
        max_seq_length=raw.get("max_position_embeddings", 2048),
        top_k=raw.get("num_experts_per_tok", raw.get("top_k", 2)),
    )


def _extract_config_from_hf(hf_config, hub_id: str) -> MonetConfig:
    """Extract MonetConfig from a HuggingFace config object."""
    return MonetConfig(
        checkpoint_path=hub_id,
        num_layers=getattr(hf_config, "num_hidden_layers", 0),
        hidden_dim=getattr(hf_config, "hidden_size", 0),
        num_experts_per_layer=getattr(hf_config, "num_local_experts", 0),
        expert_hidden_dim=getattr(hf_config, "intermediate_size", 0),
        num_attention_heads=getattr(hf_config, "num_attention_heads", 0),
        vocab_size=getattr(hf_config, "vocab_size", 0),
        max_seq_length=getattr(hf_config, "max_position_embeddings", 2048),
        top_k=getattr(hf_config, "num_experts_per_tok", 2),
    )


def _build_model_from_config(config: MonetConfig) -> nn.Module:
    """Build an empty Monet model from config.

    This is a placeholder -- the actual architecture depends on which Monet
    variant we're loading. In practice this will dispatch to the correct
    model class based on config fields.
    """
    raise NotImplementedError(
        "Direct model construction from config not yet implemented. "
        "Use _load_from_hub() with trust_remote_code=True instead."
    )


def _load_state_dict(weight_files: list[Path]) -> dict:
    """Load and merge state dicts from one or more weight files."""
    from safetensors.torch import load_file as load_safetensors

    merged = {}
    for wf in weight_files:
        if wf.suffix == ".safetensors":
            merged.update(load_safetensors(str(wf)))
        else:
            merged.update(torch.load(str(wf), map_location="cpu", weights_only=True))
    return merged


def get_expert_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Extract all expert FFN modules from a Monet model.

    Returns a list of (name, module) tuples, where name encodes the
    layer index and expert index (e.g., 'layers.3.experts.7').

    This traverses the model looking for the MoE expert pattern. The exact
    attribute names depend on the Monet variant; this function tries common
    naming conventions.
    """
    experts = []
    for name, module in model.named_modules():
        # Common patterns for expert FFN modules in MoE architectures
        if any(
            pattern in name
            for pattern in [".experts.", ".expert_", ".moe.experts."]
        ):
            # Only grab leaf expert modules (the actual FFN, not containers)
            if not list(module.children()) or _is_ffn_block(module):
                experts.append((name, module))
    return experts


def get_router_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Extract all router/gating modules from a Monet model."""
    routers = []
    for name, module in model.named_modules():
        if any(
            pattern in name for pattern in [".gate.", ".router.", ".gating."]
        ):
            routers.append((name, module))
    return routers


def _is_ffn_block(module: nn.Module) -> bool:
    """Heuristic: does this module look like an FFN block?"""
    children = dict(module.named_children())
    has_linear = any(isinstance(c, nn.Linear) for c in children.values())
    return has_linear
