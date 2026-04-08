"""Registry of known Monet checkpoints and their architectural parameters.

Used by the loader to validate that a requested checkpoint matches the
pipeline's supported configuration (currently VD-only) and to look up
architectural constants like d_expert and num_half_experts without
re-parsing config.json.

See docs/model_selection.md for the rationale behind the VD-only policy
and the recommended 850M -> 1.4B -> 4.1B progression.
"""

from dataclasses import dataclass
from typing import Literal


Decomposition = Literal["vd", "hd"]


@dataclass(frozen=True)
class MonetCheckpointInfo:
    """Architectural metadata for a released Monet checkpoint."""

    hub_id: str
    decomposition: Decomposition
    params_label: str          # "850M", "1.4B", "4.1B"
    d_model: int
    d_expert: int
    num_half_experts_per_axis: int  # N; effective experts per layer = N^2
    specialized: str | None = None   # "chat", "code", "vision", or None

    @property
    def num_half_experts_per_layer(self) -> int:
        """2N: total distinct half-experts per layer across both product-key axes."""
        return 2 * self.num_half_experts_per_axis

    @property
    def num_effective_experts_per_layer(self) -> int:
        """N^2: size of the Cartesian product of half-experts."""
        return self.num_half_experts_per_axis ** 2


# Canonical N = 512 half-experts per axis, per the paper.
_N = 512


CHECKPOINTS: dict[str, MonetCheckpointInfo] = {
    # --- Base language models ---
    "MonetLLM/monet-vd-850M-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/monet-vd-850M-100BT-hf",
        decomposition="vd",
        params_label="850M",
        d_model=1536,
        d_expert=12,
        num_half_experts_per_axis=_N,
    ),
    "MonetLLM/monet-hd-850M-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/monet-hd-850M-100BT-hf",
        decomposition="hd",
        params_label="850M",
        d_model=1536,
        d_expert=12,
        num_half_experts_per_axis=_N,
    ),
    "MonetLLM/monet-vd-1.4B-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/monet-vd-1.4B-100BT-hf",
        decomposition="vd",
        params_label="1.4B",
        d_model=2048,
        d_expert=16,
        num_half_experts_per_axis=_N,
    ),
    "MonetLLM/monet-hd-1.4B-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/monet-hd-1.4B-100BT-hf",
        decomposition="hd",
        params_label="1.4B",
        d_model=2048,
        d_expert=16,
        num_half_experts_per_axis=_N,
    ),
    "MonetLLM/monet-vd-4.1B-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/monet-vd-4.1B-100BT-hf",
        decomposition="vd",
        params_label="4.1B",
        d_model=3072,
        d_expert=24,
        num_half_experts_per_axis=_N,
    ),
    "MonetLLM/monet-hd-4.1B-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/monet-hd-4.1B-100BT-hf",
        decomposition="hd",
        params_label="4.1B",
        d_model=3072,
        d_expert=24,
        num_half_experts_per_axis=_N,
    ),
    # --- Specialized VD variants (1.4B) ---
    "MonetLLM/monet-vd-1.4B-100BT-chat-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/monet-vd-1.4B-100BT-chat-hf",
        decomposition="vd",
        params_label="1.4B",
        d_model=2048,
        d_expert=16,
        num_half_experts_per_axis=_N,
        specialized="chat",
    ),
    "MonetLLM/codemonet-vd-1.4B-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/codemonet-vd-1.4B-100BT-hf",
        decomposition="vd",
        params_label="1.4B",
        d_model=2048,
        d_expert=16,
        num_half_experts_per_axis=_N,
        specialized="code",
    ),
    "MonetLLM/visionmonet-vd-1.4B-100BT-hf": MonetCheckpointInfo(
        hub_id="MonetLLM/visionmonet-vd-1.4B-100BT-hf",
        decomposition="vd",
        params_label="1.4B",  # ~2B total including the vision tower
        d_model=2048,
        d_expert=16,
        num_half_experts_per_axis=_N,
        specialized="vision",
    ),
}


# Default primary development target.
DEFAULT_CHECKPOINT = "MonetLLM/monet-vd-850M-100BT-hf"


# Recommended scaling progression (see docs/model_selection.md).
SCALING_PROGRESSION: tuple[str, ...] = (
    "MonetLLM/monet-vd-850M-100BT-hf",
    "MonetLLM/monet-vd-1.4B-100BT-hf",
    "MonetLLM/monet-vd-4.1B-100BT-hf",
)


class UnsupportedDecompositionError(ValueError):
    """Raised when a checkpoint's decomposition is not in the allowed set."""


def lookup(hub_id: str) -> MonetCheckpointInfo:
    """Look up a known checkpoint by hub ID.

    Raises:
        KeyError: If hub_id is not in the registry.
    """
    if hub_id not in CHECKPOINTS:
        raise KeyError(
            f"Unknown Monet checkpoint: {hub_id!r}. "
            f"Known checkpoints: {sorted(CHECKPOINTS)}"
        )
    return CHECKPOINTS[hub_id]


def validate_decomposition(
    hub_id: str,
    allowed: list[Decomposition] | None,
) -> MonetCheckpointInfo:
    """Validate a checkpoint against an allowed-decomposition list.

    The conversion pipeline is VD-only by default (see
    docs/model_selection.md). HD checkpoints are accepted only when
    explicitly allowed, e.g. for late-stage comparison experiments.

    Args:
        hub_id: Checkpoint hub ID.
        allowed: List of allowed decomposition codes (e.g. ["vd"]). None
            means "no restriction".

    Returns:
        The checkpoint info if validation passes.

    Raises:
        KeyError: If hub_id is not in the registry.
        UnsupportedDecompositionError: If the checkpoint's decomposition
            is not in `allowed`.
    """
    info = lookup(hub_id)
    if allowed is None:
        return info
    if info.decomposition not in allowed:
        raise UnsupportedDecompositionError(
            f"Checkpoint {hub_id!r} uses decomposition "
            f"{info.decomposition!r}, which is not in the allowed set "
            f"{allowed}. The pipeline targets VD by default; see "
            f"docs/model_selection.md for rationale."
        )
    return info


def list_by_decomposition(decomposition: Decomposition) -> list[MonetCheckpointInfo]:
    """Return all registered checkpoints with the given decomposition."""
    return [c for c in CHECKPOINTS.values() if c.decomposition == decomposition]
