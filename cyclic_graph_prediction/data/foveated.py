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


SACCADE_POLICIES = {
    "random": RandomSaccadePolicy,
    "center_bias": CenterBiasSaccadePolicy,
    "scanpath": ScanpathSaccadePolicy,
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

        policy_cls = SACCADE_POLICIES[saccade_policy]
        self.saccade_gen = policy_cls(
            image_size=image_size, fovea_size=fovea_size, seed=seed,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image, identity_label = self.dataset[idx]  # [C, H, W]

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
