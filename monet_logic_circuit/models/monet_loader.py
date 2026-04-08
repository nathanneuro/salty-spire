"""Load pretrained Monet checkpoints and provide a uniform interface."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from monet_logic_circuit.models.registry import (
    Decomposition,
    MonetCheckpointInfo,
    lookup as registry_lookup,
    validate_decomposition,
)


@dataclass
class MonetConfig:
    """Configuration for a loaded Monet model.

    Monet uses a product-key expert decomposition: each layer has
    `num_half_experts_per_axis` half-experts on each of 2 axes, and the
    effective expert is the Cartesian product. Conversion operates at the
    half-expert granularity; see docs/model_selection.md.
    """

    checkpoint_path: str = ""
    decomposition: Decomposition = "vd"
    num_layers: int = 0
    hidden_dim: int = 0                  # d_model
    expert_dim: int = 0                  # d_expert (per half-expert subspace)
    num_half_experts_per_axis: int = 0   # N; effective experts = N^2
    num_attention_heads: int = 0
    vocab_size: int = 0
    max_seq_length: int = 2048
    top_k: int = 2  # Number of experts selected per token
    device: str = "auto"

    # Populated after loading
    total_half_experts: int = field(init=False, default=0)
    total_effective_experts: int = field(init=False, default=0)

    def __post_init__(self):
        self.total_half_experts = (
            self.num_layers * 2 * self.num_half_experts_per_axis
        )
        self.total_effective_experts = (
            self.num_layers * self.num_half_experts_per_axis ** 2
        )


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
    allowed_decompositions: Optional[list[Decomposition]] = None,
) -> tuple[nn.Module, MonetConfig]:
    """Load a pretrained Monet model from a checkpoint path or HF hub ID.

    Args:
        checkpoint: Path to local checkpoint directory or HuggingFace hub ID.
        device: Device to load onto ('auto', 'cpu', 'cuda', 'cuda:0', etc.).
        dtype: Optional dtype override. If None, uses checkpoint's native dtype.
        allowed_decompositions: Optional allow-list of decomposition variants
            (e.g. ["vd"]). If provided and the checkpoint is a known registry
            entry, its decomposition is validated against this list. The
            pipeline defaults to VD-only; see docs/model_selection.md.

    Returns:
        Tuple of (model, config).

    Raises:
        FileNotFoundError: If checkpoint path doesn't exist locally and isn't
            a valid HF hub ID.
        UnsupportedDecompositionError: If the checkpoint's decomposition is
            not in `allowed_decompositions`.
        ValueError: If checkpoint format is unrecognized.
    """
    device = resolve_device(device)
    checkpoint_path = Path(checkpoint)

    # If this is a known registry entry, validate decomposition up front.
    registry_info: Optional[MonetCheckpointInfo] = None
    try:
        registry_info = validate_decomposition(checkpoint, allowed_decompositions)
    except KeyError:
        # Unknown checkpoint (e.g. local path or custom finetune). Skip
        # registry-based validation — callers can still pin the allow-list
        # and we will fall back to a best-effort decomposition sniff after
        # the HF config is loaded.
        pass

    # Try local path first
    if checkpoint_path.exists():
        model, config = _load_local_checkpoint(checkpoint_path, device, dtype)
    else:
        # Fall back to HuggingFace hub
        model, config = _load_from_hub(checkpoint, device, dtype)

    if registry_info is not None:
        # Registry takes precedence over config-sniffing since it's the
        # ground truth for released checkpoints.
        config.decomposition = registry_info.decomposition
        config.expert_dim = registry_info.d_expert
        config.num_half_experts_per_axis = registry_info.num_half_experts_per_axis
    elif allowed_decompositions is not None:
        if config.decomposition not in allowed_decompositions:
            from monet_logic_circuit.models.registry import (
                UnsupportedDecompositionError,
            )
            raise UnsupportedDecompositionError(
                f"Checkpoint {checkpoint!r} has decomposition "
                f"{config.decomposition!r}, not in allowed set "
                f"{allowed_decompositions}."
            )

    return model, config


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


def _sniff_decomposition(raw_or_obj) -> Decomposition:
    """Best-effort decomposition sniff for unregistered checkpoints.

    Looks for an explicit `decomposition` field, then falls back to
    substring match on `model_type`/`architectures`. Defaults to "vd"
    (the pipeline's primary target) when no signal is available.
    """
    get = raw_or_obj.get if isinstance(raw_or_obj, dict) else (
        lambda k, default=None: getattr(raw_or_obj, k, default)
    )
    explicit = get("decomposition")
    if isinstance(explicit, str) and explicit.lower() in ("vd", "hd"):
        return explicit.lower()  # type: ignore[return-value]

    for field_name in ("model_type", "architectures"):
        value = get(field_name)
        if value is None:
            continue
        text = " ".join(value) if isinstance(value, (list, tuple)) else str(value)
        text = text.lower()
        if "-hd-" in text or "monethd" in text:
            return "hd"
        if "-vd-" in text or "monetvd" in text:
            return "vd"
    return "vd"


def _parse_config(raw: dict, checkpoint_path: str) -> MonetConfig:
    """Parse a raw config dict into MonetConfig."""
    return MonetConfig(
        checkpoint_path=checkpoint_path,
        decomposition=_sniff_decomposition(raw),
        num_layers=raw.get("num_hidden_layers", raw.get("n_layer", 0)),
        hidden_dim=raw.get("hidden_size", raw.get("d_model", 0)),
        expert_dim=raw.get("moe_expert_dim", raw.get("d_expert", 0)),
        num_half_experts_per_axis=raw.get(
            "moe_experts", raw.get("num_half_experts_per_axis", 0)
        ),
        num_attention_heads=raw.get("num_attention_heads", raw.get("n_head", 0)),
        vocab_size=raw.get("vocab_size", 0),
        max_seq_length=raw.get("max_position_embeddings", 2048),
        top_k=raw.get("num_experts_per_tok", raw.get("top_k", 2)),
    )


def _extract_config_from_hf(hf_config, hub_id: str) -> MonetConfig:
    """Extract MonetConfig from a HuggingFace config object."""
    return MonetConfig(
        checkpoint_path=hub_id,
        decomposition=_sniff_decomposition(hf_config),
        num_layers=getattr(hf_config, "num_hidden_layers", 0),
        hidden_dim=getattr(hf_config, "hidden_size", 0),
        expert_dim=getattr(hf_config, "moe_expert_dim", 0),
        num_half_experts_per_axis=getattr(hf_config, "moe_experts", 0),
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


def get_half_expert_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Extract all half-expert modules from a Monet model.

    Monet uses a product-key decomposition: each layer has 2N half-experts
    (N per product-key axis). Conversion targets these directly. Effective
    experts are recovered by composing half-experts according to the
    decomposition rule (VD: additive over axes; HD: compositional).

    Returns a list of (name, module) tuples, where name encodes the
    layer index, axis (0 or 1), and half-expert index within the axis.

    This traverses the model looking for the half-expert pattern. The exact
    attribute names depend on the Monet variant and implementation; this
    function tries common naming conventions.
    """
    half_experts = []
    for name, module in model.named_modules():
        # Common patterns for half-expert / moe-expert matrices in Monet.
        # The released HF implementations expose per-axis weight matrices;
        # adjust here as needed once we lock to a specific release.
        if any(
            pattern in name
            for pattern in [
                ".half_experts.",
                ".experts.",
                ".expert_",
                ".moe.experts.",
                ".moe.half_experts.",
            ]
        ):
            if not list(module.children()) or _is_ffn_block(module):
                half_experts.append((name, module))
    return half_experts


def get_router_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Extract all router/gating modules from a Monet model.

    Monet has two product-key routers per layer (one per axis of the
    decomposition). Both stay in float precision throughout the pipeline.
    """
    routers = []
    for name, module in model.named_modules():
        if any(
            pattern in name
            for pattern in [".gate.", ".router.", ".gating.", ".product_key."]
        ):
            routers.append((name, module))
    return routers


def _is_ffn_block(module: nn.Module) -> bool:
    """Heuristic: does this module look like an FFN block?"""
    children = dict(module.named_children())
    has_linear = any(isinstance(c, nn.Linear) for c in children.values())
    return has_linear
