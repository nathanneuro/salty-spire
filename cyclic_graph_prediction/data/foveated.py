"""
Foveated vision with saccadic eye movements (Experiment 9).

Models the retina as two streams:
- Fovea: high-resolution crop around the fixation point
- Periphery: full image, heavily blurred (low spatial frequency)

Saccades are modeled as a sequence of fixation points. Each fixation
produces a (fovea_crop, peripheral_blur) pair. Across a saccade sequence,
the model must integrate sharp local detail with blurry global context.

This maps naturally onto the graph prediction framework:
- Foveal node: sees sharp local crops, specializes in fine detail
- Peripheral node: sees blurry global view, specializes in layout/gist
- Higher nodes (V4, IT, parietal): predict/integrate both streams

The prediction task across saccades: from the current peripheral view +
previous foveal views, predict the next foveal view's latent. This is
exactly what the brain does — predictive coding across eye movements.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


# --- Saccade policies ---

class SaccadePolicy:
    """Base class for generating fixation sequences."""

    def __init__(self, image_size: int, fovea_size: int, seed: int | None = None):
        self.image_size = image_size
        self.fovea_size = fovea_size
        self.half_fovea = fovea_size // 2
        self.rng = np.random.RandomState(seed)

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        """Generate a sequence of (y, x) fixation coordinates."""
        raise NotImplementedError


class RandomSaccadePolicy(SaccadePolicy):
    """Uniformly random fixation points (null model for saccade patterns)."""

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        margin = self.half_fovea
        ys = self.rng.randint(margin, self.image_size - margin, size=num_fixations)
        xs = self.rng.randint(margin, self.image_size - margin, size=num_fixations)
        return list(zip(ys.tolist(), xs.tolist()))


class CenterBiasSaccadePolicy(SaccadePolicy):
    """Center-biased fixations with Gaussian jitter (realistic prior).

    Humans fixate near the center of images with a strong bias.
    """

    def __init__(self, image_size: int, fovea_size: int, sigma: float = 0.25, **kwargs):
        super().__init__(image_size, fovea_size, **kwargs)
        self.sigma = sigma  # fraction of image size

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        center = self.image_size / 2
        std = self.sigma * self.image_size
        margin = self.half_fovea

        fixations = []
        for _ in range(num_fixations):
            y = int(np.clip(self.rng.normal(center, std), margin, self.image_size - margin))
            x = int(np.clip(self.rng.normal(center, std), margin, self.image_size - margin))
            fixations.append((y, x))
        return fixations


class ScanpathSaccadePolicy(SaccadePolicy):
    """Sequential scanpath: starts near center, makes saccades of realistic length.

    Models the temporal structure of real saccade sequences:
    consecutive fixations are correlated (small saccades are more common).
    """

    def __init__(
        self,
        image_size: int,
        fovea_size: int,
        mean_saccade_length: float = 0.15,
        **kwargs,
    ):
        super().__init__(image_size, fovea_size, **kwargs)
        self.mean_saccade_length = mean_saccade_length  # fraction of image size

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        margin = self.half_fovea
        saccade_px = self.mean_saccade_length * self.image_size

        # Start near center
        y = self.image_size // 2
        x = self.image_size // 2
        fixations = [(y, x)]

        for _ in range(num_fixations - 1):
            # Random direction, exponential-distributed amplitude
            angle = self.rng.uniform(0, 2 * math.pi)
            amplitude = self.rng.exponential(saccade_px)
            dy = int(amplitude * math.sin(angle))
            dx = int(amplitude * math.cos(angle))
            y = int(np.clip(y + dy, margin, self.image_size - margin))
            x = int(np.clip(x + dx, margin, self.image_size - margin))
            fixations.append((y, x))

        return fixations


class SaliencyGuidedPolicy(SaccadePolicy):
    """Fixate on high-saliency regions computed from the image itself.

    Uses a lightweight saliency measure: local contrast (gradient magnitude)
    plus center bias. This approximates Itti-Koch saliency without requiring
    a separate neural network, making it fast and differentiable.

    The saliency map is computed once per image, then fixations are sampled
    from it sequentially with inhibition-of-return (IOR) to avoid refixating.
    """

    def __init__(
        self,
        image_size: int,
        fovea_size: int,
        center_bias_sigma: float = 0.3,
        ior_radius: float = 0.15,
        **kwargs,
    ):
        super().__init__(image_size, fovea_size, **kwargs)
        self.center_bias_sigma = center_bias_sigma
        self.ior_radius = ior_radius  # fraction of image size
        self._current_saliency = None

    def set_image(self, image: torch.Tensor):
        """Precompute saliency map for the current image.

        Args:
            image: [C, H, W] tensor (normalized)
        """
        self._current_saliency = self._compute_saliency(image)

    def _compute_saliency(self, image: torch.Tensor) -> np.ndarray:
        """Compute saliency as gradient magnitude + center bias.

        Captures edges, texture boundaries, and high-contrast regions —
        the features that drive bottom-up attention in the brain.
        """
        C, H, W = image.shape
        gray = image.mean(dim=0).numpy()  # [H, W]

        # Gradient magnitude (Sobel-like)
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gradient_mag = np.sqrt(gy ** 2 + gx ** 2)

        # Local contrast: std in 16x16 neighborhoods
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray, size=16)
        local_sq_mean = uniform_filter(gray ** 2, size=16)
        local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

        # Color contrast (channel variance at each pixel)
        if C >= 3:
            color_var = image.numpy().var(axis=0)  # [H, W]
        else:
            color_var = np.zeros_like(gray)

        # Combine cues
        saliency = gradient_mag + local_std + 0.5 * color_var

        # Center bias (Gaussian)
        cy, cx = H / 2, W / 2
        ys = np.arange(H)
        xs = np.arange(W)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        sigma = self.center_bias_sigma * H
        center_weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))

        saliency = saliency * (0.5 + 0.5 * center_weight)

        # Exclude margins where foveal crop can't fit
        margin = self.half_fovea
        saliency[:margin, :] = 0
        saliency[-margin:, :] = 0
        saliency[:, :margin] = 0
        saliency[:, -margin:] = 0

        return saliency

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        if self._current_saliency is None:
            # Fallback to center-bias if no image set
            center = self.image_size // 2
            return [(center, center)] * num_fixations

        saliency = self._current_saliency.copy()
        H, W = saliency.shape
        ior_px = int(self.ior_radius * self.image_size)

        fixations = []
        for _ in range(num_fixations):
            # Sample proportional to saliency
            flat = saliency.flatten()
            total = flat.sum()
            if total < 1e-10:
                # Saliency exhausted — fall back to random
                margin = self.half_fovea
                y = self.rng.randint(margin, H - margin)
                x = self.rng.randint(margin, W - margin)
            else:
                probs = flat / total
                idx = self.rng.choice(len(probs), p=probs)
                y, x = divmod(idx, W)

            fixations.append((int(y), int(x)))

            # Inhibition of return: suppress saliency around this fixation
            y_lo = max(0, y - ior_px)
            y_hi = min(H, y + ior_px)
            x_lo = max(0, x - ior_px)
            x_hi = min(W, x + ior_px)
            saliency[y_lo:y_hi, x_lo:x_hi] *= 0.1

        return fixations


class ObjectCenterPolicy(SaccadePolicy):
    """Fixate on salient object regions using simple foreground detection.

    Approximates the human tendency to fixate on objects rather than
    background. Uses a simple figure-ground separation based on
    contrast with the image border (background prior).

    First fixation: most salient foreground region.
    Subsequent: explore other foreground regions with IOR.
    """

    def __init__(
        self,
        image_size: int,
        fovea_size: int,
        border_width: int = 16,
        ior_radius: float = 0.2,
        **kwargs,
    ):
        super().__init__(image_size, fovea_size, **kwargs)
        self.border_width = border_width
        self.ior_radius = ior_radius
        self._current_objectness = None

    def set_image(self, image: torch.Tensor):
        """Compute objectness map from border-contrast heuristic."""
        C, H, W = image.shape
        gray = image.mean(dim=0).numpy()

        # Background model: mean color of border pixels
        bw = self.border_width
        border_pixels = np.concatenate([
            gray[:bw, :].flatten(),
            gray[-bw:, :].flatten(),
            gray[:, :bw].flatten(),
            gray[:, -bw:].flatten(),
        ])
        bg_mean = border_pixels.mean()
        bg_std = max(border_pixels.std(), 1e-6)

        # Objectness = how much each pixel differs from background
        objectness = np.abs(gray - bg_mean) / bg_std

        # Smooth to get blobs
        from scipy.ndimage import gaussian_filter
        objectness = gaussian_filter(objectness, sigma=8)

        # Suppress margins
        margin = self.half_fovea
        objectness[:margin, :] = 0
        objectness[-margin:, :] = 0
        objectness[:, :margin] = 0
        objectness[:, -margin:] = 0

        self._current_objectness = objectness

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        if self._current_objectness is None:
            center = self.image_size // 2
            return [(center, center)] * num_fixations

        H, W = self._current_objectness.shape
        objectness = self._current_objectness.copy()
        ior_px = int(self.ior_radius * self.image_size)
        fixations = []

        for _ in range(num_fixations):
            flat = objectness.flatten()
            total = flat.sum()
            if total < 1e-10:
                margin = self.half_fovea
                y = self.rng.randint(margin, H - margin)
                x = self.rng.randint(margin, W - margin)
            else:
                probs = flat / total
                idx = self.rng.choice(len(probs), p=probs)
                y, x = divmod(idx, W)

            fixations.append((int(y), int(x)))

            # IOR
            y_lo = max(0, y - ior_px)
            y_hi = min(H, y + ior_px)
            x_lo = max(0, x - ior_px)
            x_hi = min(W, x + ior_px)
            objectness[y_lo:y_hi, x_lo:x_hi] *= 0.05

        return fixations


class InformationGainPolicy(SaccadePolicy):
    """Fixate where the gap between peripheral and foveal information is largest.

    The intuition: the best place to look is where the blurry peripheral
    view is most uncertain — where foveating would provide the most
    new information. This approximates active inference / expected free
    energy minimization.

    Uses local entropy of the blurred image as a proxy for uncertainty.
    High peripheral entropy = unpredictable region = worth foveating.
    """

    def __init__(
        self,
        image_size: int,
        fovea_size: int,
        blur_sigma: float = 8.0,
        ior_radius: float = 0.15,
        **kwargs,
    ):
        super().__init__(image_size, fovea_size, **kwargs)
        self.blur_sigma = blur_sigma
        self.ior_radius = ior_radius
        self._uncertainty_map = None

    def set_image(self, image: torch.Tensor):
        """Compute uncertainty map: where is the peripheral view most uncertain?"""
        C, H, W = image.shape

        # Create blurred (peripheral) version
        ks = int(self.blur_sigma * 6) | 1
        blurred = _gaussian_blur(image.unsqueeze(0), ks, self.blur_sigma).squeeze(0)

        # Information gap: how much detail is lost by blurring?
        # High difference = region where blurring destroys a lot of info
        detail_loss = (image - blurred).abs().mean(dim=0).numpy()  # [H, W]

        # Local entropy of the blurred image (proxy for peripheral uncertainty)
        gray_blur = blurred.mean(dim=0).numpy()
        # Discretize to estimate local entropy
        from scipy.ndimage import uniform_filter
        local_mean = uniform_filter(gray_blur, size=16)
        local_sq = uniform_filter(gray_blur ** 2, size=16)
        local_var = np.maximum(local_sq - local_mean ** 2, 1e-8)
        # Gaussian entropy ∝ log(variance)
        local_entropy = 0.5 * np.log(local_var + 1e-8)

        # Combine: high detail loss AND high peripheral uncertainty
        uncertainty = detail_loss * np.maximum(local_entropy - local_entropy.min(), 0)

        # Suppress margins
        margin = self.half_fovea
        uncertainty[:margin, :] = 0
        uncertainty[-margin:, :] = 0
        uncertainty[:, :margin] = 0
        uncertainty[:, -margin:] = 0

        self._uncertainty_map = uncertainty

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        if self._uncertainty_map is None:
            center = self.image_size // 2
            return [(center, center)] * num_fixations

        H, W = self._uncertainty_map.shape
        uncertainty = self._uncertainty_map.copy()
        ior_px = int(self.ior_radius * self.image_size)
        fixations = []

        for _ in range(num_fixations):
            flat = uncertainty.flatten()
            total = flat.sum()
            if total < 1e-10:
                margin = self.half_fovea
                y = self.rng.randint(margin, H - margin)
                x = self.rng.randint(margin, W - margin)
            else:
                probs = flat / total
                idx = self.rng.choice(len(probs), p=probs)
                y, x = divmod(idx, W)

            fixations.append((int(y), int(x)))

            y_lo = max(0, y - ior_px)
            y_hi = min(H, y + ior_px)
            x_lo = max(0, x - ior_px)
            x_hi = min(W, x + ior_px)
            uncertainty[y_lo:y_hi, x_lo:x_hi] *= 0.1

        return fixations


class AttentionMapPolicy(SaccadePolicy):
    """Use a pretrained ViT's CLS-token attention to guide fixations.

    A pretrained ViT already knows "where to attend" for classification.
    The CLS token's attention over patches in the last (or a specified)
    layer gives a task-relevant saliency map — these are the regions
    the model considers most informative.

    This is analogous to the superior colliculus using cortical feedback
    (from FEF / prefrontal) to plan saccades. The attention map is
    essentially a learned saliency map trained on millions of images.

    Supports any timm ViT. The attention maps are extracted once per image
    at dataset construction time (or cached), not during training.
    """

    def __init__(
        self,
        image_size: int,
        fovea_size: int,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        layer_index: int = -1,
        head_reduction: str = "mean",
        ior_radius: float = 0.15,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(image_size, fovea_size, **kwargs)
        self.ior_radius = ior_radius
        self.temperature = temperature
        self.layer_index = layer_index
        self.head_reduction = head_reduction
        self._attention_map = None

        # Load pretrained ViT
        import timm
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.patch_size = self.model.patch_embed.patch_size
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]
        self.grid_size = image_size // self.patch_size

    @torch.no_grad()
    def set_image(self, image: torch.Tensor):
        """Extract CLS attention map from pretrained ViT.

        The CLS token attention in the last layer tells us which patches
        the model considers most task-relevant.
        """
        # image: [C, H, W] — add batch dim
        x = image.unsqueeze(0)

        # Forward through patch embedding + transformer blocks
        # We need attention weights, so we hook into the attention layers
        attn_weights = []

        def _hook(module, input, output):
            # timm attention modules store attn weights differently
            # For most timm ViTs, we can access via the module
            pass

        # Alternative: use timm's built-in feature extraction
        # Most timm ViTs support forward with attention output
        try:
            attn_map = self._extract_attention_timm(x)
        except Exception:
            # Fallback: use gradient-based attribution
            attn_map = self._extract_gradient_saliency(x)

        self._attention_map = attn_map

    def _extract_attention_timm(self, x: torch.Tensor) -> np.ndarray:
        """Extract CLS attention from timm ViT via manual forward pass."""
        B = x.shape[0]

        # Patch embed
        x_tokens = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(B, -1, -1)

        if hasattr(self.model, 'pos_embed'):
            x_tokens = torch.cat([cls_token, x_tokens], dim=1)
            x_tokens = x_tokens + self.model.pos_embed
        else:
            x_tokens = torch.cat([cls_token, x_tokens], dim=1)

        if hasattr(self.model, 'pos_drop'):
            x_tokens = self.model.pos_drop(x_tokens)

        # Run through transformer blocks, capturing attention at target layer
        blocks = self.model.blocks
        target_idx = self.layer_index % len(blocks)

        for i, block in enumerate(blocks):
            if i == target_idx:
                # Extract attention from this block
                attn_out = self._forward_block_with_attention(block, x_tokens)
                x_tokens, attn = attn_out
            else:
                x_tokens = block(x_tokens)

        # attn: [B, num_heads, num_tokens, num_tokens]
        # CLS attention over patches: attn[:, :, 0, 1:] (CLS attending to patches)
        cls_attn = attn[0, :, 0, 1:]  # [num_heads, num_patches]

        if self.head_reduction == "mean":
            cls_attn = cls_attn.mean(dim=0)
        elif self.head_reduction == "max":
            cls_attn = cls_attn.max(dim=0).values
        else:
            cls_attn = cls_attn.mean(dim=0)

        # Reshape to 2D grid
        grid = cls_attn.reshape(self.grid_size, self.grid_size).numpy()

        # Upsample to image resolution
        from scipy.ndimage import zoom
        scale = self.image_size / self.grid_size
        attn_map = zoom(grid, scale, order=1)

        # Suppress margins
        margin = self.half_fovea
        attn_map[:margin, :] = 0
        attn_map[-margin:, :] = 0
        attn_map[:, :margin] = 0
        attn_map[:, -margin:] = 0

        return attn_map

    def _forward_block_with_attention(self, block, x):
        """Forward through a timm Block, returning (output, attention_weights)."""
        # Most timm blocks: x = x + attn(norm1(x)); x = x + mlp(norm2(x))
        residual = x
        x_norm = block.norm1(x)

        # Access the attention module
        attn_module = block.attn
        B, N, C = x_norm.shape
        qkv = attn_module.qkv(x_norm).reshape(
            B, N, 3, attn_module.num_heads, C // attn_module.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_weights = (q @ k.transpose(-2, -1)) * attn_module.scale
        attn_weights = attn_weights.softmax(dim=-1)

        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        attn_out = attn_module.proj(attn_out)
        if hasattr(attn_module, 'proj_drop'):
            attn_out = attn_module.proj_drop(attn_out)

        x = residual + attn_out
        x = x + block.mlp(block.norm2(x))
        return x, attn_weights

    def _extract_gradient_saliency(self, x: torch.Tensor) -> np.ndarray:
        """Fallback: gradient-based saliency if attention extraction fails."""
        x_input = x.clone().requires_grad_(True)
        output = self.model(x_input)
        # Backprop from max logit
        max_logit = output.max()
        max_logit.backward()

        # Gradient magnitude as saliency
        grad = x_input.grad[0].abs().mean(dim=0).numpy()

        margin = self.half_fovea
        grad[:margin, :] = 0
        grad[-margin:, :] = 0
        grad[:, :margin] = 0
        grad[:, -margin:] = 0
        return grad

    def generate_fixations(self, num_fixations: int) -> list[tuple[int, int]]:
        if self._attention_map is None:
            center = self.image_size // 2
            return [(center, center)] * num_fixations

        H, W = self._attention_map.shape
        # Apply temperature: higher temp = more exploratory, lower = more greedy
        attn = self._attention_map.copy()
        if self.temperature != 1.0:
            attn = np.power(np.maximum(attn, 0), 1.0 / self.temperature)

        ior_px = int(self.ior_radius * self.image_size)
        fixations = []

        for _ in range(num_fixations):
            flat = attn.flatten()
            total = flat.sum()
            if total < 1e-10:
                margin = self.half_fovea
                y = self.rng.randint(margin, H - margin)
                x = self.rng.randint(margin, W - margin)
            else:
                probs = flat / total
                idx = self.rng.choice(len(probs), p=probs)
                y, x = divmod(idx, W)

            fixations.append((int(y), int(x)))

            # IOR
            y_lo = max(0, y - ior_px)
            y_hi = min(H, y + ior_px)
            x_lo = max(0, x - ior_px)
            x_hi = min(W, x + ior_px)
            attn[y_lo:y_hi, x_lo:x_hi] *= 0.1

        return fixations


SACCADE_POLICIES = {
    "random": RandomSaccadePolicy,
    "center_bias": CenterBiasSaccadePolicy,
    "scanpath": ScanpathSaccadePolicy,
    "saliency": SaliencyGuidedPolicy,
    "object_center": ObjectCenterPolicy,
    "information_gain": InformationGainPolicy,
    "attention_map": AttentionMapPolicy,
}

# Policies that need set_image() called before generate_fixations()
IMAGE_ADAPTIVE_POLICIES = {
    "saliency", "object_center", "information_gain", "attention_map",
}


# --- Foveated image processing ---

def extract_foveal_crop(
    image: torch.Tensor,
    fixation: tuple[int, int],
    fovea_size: int,
    output_size: int = 64,
) -> torch.Tensor:
    """Extract a high-resolution crop centered on the fixation point.

    Args:
        image: [C, H, W]
        fixation: (y, x) center of fixation
        fovea_size: size of the crop in pixels (before resize)
        output_size: final output resolution

    Returns:
        [C, output_size, output_size] — sharp foveal view
    """
    C, H, W = image.shape
    y, x = fixation
    half = fovea_size // 2

    # Clamp to image boundaries
    y1 = max(0, y - half)
    y2 = min(H, y + half)
    x1 = max(0, x - half)
    x2 = min(W, x + half)

    crop = image[:, y1:y2, x1:x2]

    # Resize to standard output size
    crop = F.interpolate(
        crop.unsqueeze(0), size=(output_size, output_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    return crop


def create_peripheral_view(
    image: torch.Tensor,
    blur_sigma: float = 8.0,
    output_size: int = 64,
) -> torch.Tensor:
    """Create a blurred peripheral view of the full image.

    Models the low spatial frequency information available in peripheral
    vision — layout, gist, coarse color/luminance.

    Args:
        image: [C, H, W]
        blur_sigma: Gaussian blur sigma (higher = more peripheral)
        output_size: final output resolution

    Returns:
        [C, output_size, output_size] — blurry global view
    """
    # Gaussian blur
    kernel_size = int(blur_sigma * 6) | 1  # ensure odd
    blurred = _gaussian_blur(image.unsqueeze(0), kernel_size, blur_sigma).squeeze(0)

    # Downsample to output size
    blurred = F.interpolate(
        blurred.unsqueeze(0), size=(output_size, output_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    return blurred


def create_foveated_view(
    image: torch.Tensor,
    fixation: tuple[int, int],
    fovea_size: int = 64,
    blur_sigma: float = 8.0,
) -> torch.Tensor:
    """Create a single foveated image: sharp at fixation, blurry elsewhere.

    This produces a full-resolution image with spatially varying blur,
    like the actual retinal image. Sharp within fovea_size of the fixation
    point, progressively blurred toward the periphery.

    Args:
        image: [C, H, W]
        fixation: (y, x) fixation point
        fovea_size: diameter of sharp foveal region
        blur_sigma: max blur in far periphery

    Returns:
        [C, H, W] foveated image
    """
    C, H, W = image.shape
    y_fix, x_fix = fixation

    # Create distance map from fixation point
    ys = torch.arange(H, dtype=torch.float32)
    xs = torch.arange(W, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dist = torch.sqrt((yy - y_fix) ** 2 + (xx - x_fix) ** 2)

    # Eccentricity-dependent blur: no blur inside fovea, linear ramp outside
    half_fov = fovea_size / 2
    max_dist = math.sqrt(H**2 + W**2) / 2
    # Normalized eccentricity: 0 at fovea edge, 1 at image corner
    eccentricity = ((dist - half_fov).clamp(min=0) / (max_dist - half_fov)).clamp(max=1)

    # Create progressively blurred versions
    num_blur_levels = 5
    blurred_versions = [image]
    for i in range(1, num_blur_levels):
        sigma = blur_sigma * i / num_blur_levels
        ks = int(sigma * 6) | 1
        blurred_versions.append(
            _gaussian_blur(image.unsqueeze(0), ks, sigma).squeeze(0)
        )

    # Blend based on eccentricity
    result = image.clone()
    for i in range(1, num_blur_levels):
        lo = (i - 1) / num_blur_levels
        hi = i / num_blur_levels
        mask = ((eccentricity >= lo) & (eccentricity < hi)).float().unsqueeze(0)
        alpha = (eccentricity - lo) / (hi - lo + 1e-8)
        alpha = alpha.clamp(0, 1).unsqueeze(0) * mask
        result = result * (1 - alpha) + blurred_versions[i] * alpha

    # Clamp the farthest periphery
    far_mask = (eccentricity >= (num_blur_levels - 1) / num_blur_levels).float().unsqueeze(0)
    result = result * (1 - far_mask) + blurred_versions[-1] * far_mask

    return result


def _gaussian_blur(
    x: torch.Tensor, kernel_size: int, sigma: float
) -> torch.Tensor:
    """Apply Gaussian blur to a [B, C, H, W] tensor."""
    channels = x.shape[1]

    # 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 2D via two 1D convolutions
    kernel_h = kernel_1d.reshape(1, 1, 1, -1).expand(channels, -1, -1, -1)
    kernel_v = kernel_1d.reshape(1, 1, -1, 1).expand(channels, -1, -1, -1)

    pad_h = kernel_size // 2
    x = F.pad(x, [pad_h, pad_h, 0, 0], mode="reflect")
    x = F.conv2d(x, kernel_h, groups=channels)
    x = F.pad(x, [0, 0, pad_h, pad_h], mode="reflect")
    x = F.conv2d(x, kernel_v, groups=channels)
    return x


# --- Dataset ---

class FoveatedSaccadeDataset(Dataset):
    """Dataset that produces saccade sequences with foveal + peripheral views.

    Each sample is a sequence of fixations on one image. At each fixation:
    - foveal_crop: high-res crop around fixation point
    - peripheral_view: blurred full image (same across fixations)
    - foveated_image: full image with spatially varying blur at this fixation
    - fixation_coords: (y, x) normalized to [0, 1]

    Labels:
    - identity_label: object class
    - saccade_direction: angle from previous fixation (for motion probing)
    """

    def __init__(
        self,
        root: str,
        image_size: int = 224,
        fovea_size: int = 64,
        fovea_output_size: int = 64,
        peripheral_output_size: int = 64,
        peripheral_blur_sigma: float = 8.0,
        num_fixations: int = 4,
        saccade_policy: str = "scanpath",
        train: bool = True,
        dataset_name: str = "cifar100",
        seed: int | None = None,
    ):
        self.image_size = image_size
        self.fovea_size = fovea_size
        self.fovea_output_size = fovea_output_size
        self.peripheral_output_size = peripheral_output_size
        self.peripheral_blur_sigma = peripheral_blur_sigma
        self.num_fixations = num_fixations

        base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if dataset_name == "cifar100":
            self.dataset = datasets.CIFAR100(
                root=root, train=train, transform=base_transform, download=True,
            )
            self.num_classes = 100
        elif dataset_name == "cifar10":
            self.dataset = datasets.CIFAR10(
                root=root, train=train, transform=base_transform, download=True,
            )
            self.num_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.saccade_policy_name = saccade_policy
        policy_cls = SACCADE_POLICIES[saccade_policy]
        self.saccade_gen = policy_cls(
            image_size=image_size, fovea_size=fovea_size, seed=seed,
        )
        self._is_adaptive = saccade_policy in IMAGE_ADAPTIVE_POLICIES

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image, identity_label = self.dataset[idx]  # [C, H, W]

        # For content-driven policies, compute saliency/objectness from image
        if self._is_adaptive:
            self.saccade_gen.set_image(image)

        # Generate saccade sequence
        fixations = self.saccade_gen.generate_fixations(self.num_fixations)

        # Peripheral view (constant across fixations — the blurry "gist")
        peripheral = create_peripheral_view(
            image, self.peripheral_blur_sigma, self.peripheral_output_size,
        )

        # Per-fixation foveal crops and foveated images
        foveal_crops = []
        foveated_images = []
        norm_fixations = []
        saccade_directions = []

        for i, (y, x) in enumerate(fixations):
            # High-res foveal crop
            crop = extract_foveal_crop(
                image, (y, x), self.fovea_size, self.fovea_output_size,
            )
            foveal_crops.append(crop)

            # Full foveated image (sharp at fixation, blurry elsewhere)
            foveated = create_foveated_view(
                image, (y, x), self.fovea_size, self.peripheral_blur_sigma,
            )
            foveated_images.append(foveated)

            # Normalized fixation coordinates
            norm_fixations.append(
                torch.tensor([y / self.image_size, x / self.image_size])
            )

            # Saccade direction from previous fixation
            if i > 0:
                prev_y, prev_x = fixations[i - 1]
                dy = y - prev_y
                dx = x - prev_x
                angle = math.atan2(dy, dx)
                saccade_directions.append(angle)
            else:
                saccade_directions.append(0.0)

        return {
            "image": image,                                        # [C, H, W]
            "peripheral": peripheral,                              # [C, out, out]
            "foveal_crops": torch.stack(foveal_crops),             # [num_fix, C, out, out]
            "foveated_images": torch.stack(foveated_images),       # [num_fix, C, H, W]
            "fixation_coords": torch.stack(norm_fixations),        # [num_fix, 2]
            "saccade_directions": torch.tensor(saccade_directions),# [num_fix]
            "identity_label": identity_label,
        }
