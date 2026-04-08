"""Storage and retrieval of per-half-expert (input, output) trace pairs.

Traces are the reference data for every distillation and verification step.
Collected once in Step 0 on the calibration set, reused in all subsequent
steps. Keyed by half-expert name (e.g. 'layer3_axis0_he17'); the store
itself is generic over the key string.
"""

from pathlib import Path

import torch


class ExpertTraceStore:
    """Persistent store for per-expert I/O traces using safetensors.

    Each expert's traces are stored as a separate file containing:
    - inputs: (N, hidden_dim) tensor of expert inputs
    - outputs: (N, hidden_dim) tensor of expert outputs

    where N is the number of tokens routed to that expert during calibration.
    """

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _expert_path(self, expert_name: str) -> Path:
        return self.base_dir / f"{expert_name}.safetensors"

    def save_traces(
        self, expert_name: str, inputs: torch.Tensor, outputs: torch.Tensor
    ):
        """Save input/output traces for a single expert."""
        from safetensors.torch import save_file

        path = self._expert_path(expert_name)
        save_file({"inputs": inputs.contiguous(), "outputs": outputs.contiguous()}, str(path))

    def load_traces(self, expert_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load input/output traces for a single expert.

        Returns:
            Tuple of (inputs, outputs) tensors.

        Raises:
            FileNotFoundError: If traces for this expert haven't been saved.
        """
        from safetensors.torch import load_file

        path = self._expert_path(expert_name)
        if not path.exists():
            raise FileNotFoundError(f"No traces found for expert '{expert_name}' at {path}")
        data = load_file(str(path))
        return data["inputs"], data["outputs"]

    def has_traces(self, expert_name: str) -> bool:
        return self._expert_path(expert_name).exists()

    def list_experts(self) -> list[str]:
        """List all experts with saved traces."""
        return [p.stem for p in sorted(self.base_dir.glob("*.safetensors"))]

    def get_trace_stats(self, expert_name: str) -> dict:
        """Get basic stats about stored traces without loading full tensors."""
        from safetensors import safe_open

        path = self._expert_path(expert_name)
        if not path.exists():
            return {}

        with safe_open(str(path), framework="pt") as f:
            input_shape = f.get_tensor("inputs").shape
            output_shape = f.get_tensor("outputs").shape

        return {
            "expert_name": expert_name,
            "num_samples": input_shape[0],
            "input_dim": input_shape[1] if len(input_shape) > 1 else 0,
            "output_dim": output_shape[1] if len(output_shape) > 1 else 0,
        }

    def load_all_traces(
        self, expert_names: list[str] | None = None
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Load traces for multiple experts at once.

        Args:
            expert_names: List of expert names. If None, loads all available.

        Returns:
            Dict mapping expert_name -> (inputs, outputs).
        """
        if expert_names is None:
            expert_names = self.list_experts()

        traces = {}
        for name in expert_names:
            if self.has_traces(name):
                traces[name] = self.load_traces(name)
        return traces
