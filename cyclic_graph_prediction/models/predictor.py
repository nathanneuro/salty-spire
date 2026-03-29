"""
Latent predictor: maps one node's representation to predict another node's latents.

Each directed edge (i -> j) in the graph has an associated predictor that
takes node i's latent and predicts node j's latent. Optionally includes
a learnable precision weight for precision-weighted prediction errors.
"""

import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """Predictor head for one directed edge in the graph.

    Maps source node's latent representation to predict target node's latent.
    Optionally includes learnable precision (inverse variance) weights for
    precision-weighted prediction error (Experiment 5).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        use_precision: bool = False,
    ):
        super().__init__()

        layers = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)

        # Precision weighting (Experiment 5)
        self.use_precision = use_precision
        if use_precision:
            # Log-precision to ensure positivity via exp()
            self.log_precision = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self, source_latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict target latent from source latent.

        Args:
            source_latent: [B, input_dim] from source node

        Returns:
            predicted_latent: [B, output_dim]
            precision: [output_dim] or None if not using precision weighting
        """
        predicted = self.mlp(source_latent)
        precision = self.log_precision.exp() if self.use_precision else None
        return predicted, precision


class PrecisionWeightedLoss(nn.Module):
    """Precision-weighted prediction error loss.

    L = 0.5 * sum(precision * (pred - target)^2) - 0.5 * sum(log(precision))

    The second term prevents the trivial solution of precision -> 0.
    This is the negative log-likelihood of a diagonal Gaussian with
    learned precision (inverse variance).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        precision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if precision is None:
            return nn.functional.mse_loss(prediction, target)

        sq_error = (prediction - target) ** 2
        # precision: [dim], broadcast over batch
        weighted_error = 0.5 * (precision * sq_error).mean()
        log_det = -0.5 * precision.log().sum() / prediction.shape[-1]
        return weighted_error + log_det
