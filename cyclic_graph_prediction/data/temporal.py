"""
Temporal / video dataset for ventral-dorsal stream experiments (Experiment 7).

Produces frame pairs with known spatial transformations, enabling
identity vs. motion probing to test stream specialization.

Approach: given a static image dataset, generate synthetic "video" by
applying controlled spatial transforms to create frame pairs. The identity
label is the original class; the motion label is the transform type.
This lets us test on CIFAR-100 without requiring actual video data.

For real video data, also supports loading frame pairs from video datasets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np


# --- Spatial transform catalog (the "motion" labels) ---

TRANSFORMS = {
    0: "static",          # no change
    1: "translate_right",  # shift right
    2: "translate_up",     # shift up
    3: "zoom_in",          # center crop + resize (looming)
    4: "zoom_out",         # pad + resize (receding)
    5: "rotate_cw",        # clockwise rotation
    6: "rotate_ccw",       # counter-clockwise rotation
    7: "shear_h",          # horizontal shear
}

NUM_MOTION_CLASSES = len(TRANSFORMS)


def apply_transform(
    image: torch.Tensor, transform_id: int, magnitude: float = 0.15
) -> torch.Tensor:
    """Apply a spatial transform to simulate frame-to-frame motion.

    Args:
        image: [C, H, W] tensor
        transform_id: which transform to apply
        magnitude: strength of the transform

    Returns:
        Transformed [C, H, W] tensor
    """
    C, H, W = image.shape
    m = magnitude

    if transform_id == 0:  # static
        return image.clone()

    elif transform_id == 1:  # translate right
        shift = int(W * m)
        out = torch.zeros_like(image)
        out[:, :, shift:] = image[:, :, : W - shift]
        return out

    elif transform_id == 2:  # translate up
        shift = int(H * m)
        out = torch.zeros_like(image)
        out[:, : H - shift, :] = image[:, shift:, :]
        return out

    elif transform_id == 3:  # zoom in (looming)
        crop = int(H * m)
        cropped = image[:, crop : H - crop, crop : W - crop]
        return F.interpolate(
            cropped.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)

    elif transform_id == 4:  # zoom out (receding)
        pad = int(H * m)
        padded = F.pad(image, [pad, pad, pad, pad], mode="constant", value=0)
        return F.interpolate(
            padded.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)

    elif transform_id == 5:  # rotate clockwise
        angle = magnitude * 30  # degrees
        theta = torch.tensor(
            [[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
             [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]],
            dtype=image.dtype,
        )
        grid = F.affine_grid(
            theta.unsqueeze(0), [1, C, H, W], align_corners=False
        )
        return F.grid_sample(
            image.unsqueeze(0), grid, align_corners=False
        ).squeeze(0)

    elif transform_id == 6:  # rotate counter-clockwise
        angle = -magnitude * 30
        theta = torch.tensor(
            [[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
             [np.sin(np.radians(angle)),  np.cos(np.radians(angle)), 0]],
            dtype=image.dtype,
        )
        grid = F.affine_grid(
            theta.unsqueeze(0), [1, C, H, W], align_corners=False
        )
        return F.grid_sample(
            image.unsqueeze(0), grid, align_corners=False
        ).squeeze(0)

    elif transform_id == 7:  # horizontal shear
        theta = torch.tensor(
            [[1, m, 0],
             [0, 1, 0]],
            dtype=image.dtype,
        )
        grid = F.affine_grid(
            theta.unsqueeze(0), [1, C, H, W], align_corners=False
        )
        return F.grid_sample(
            image.unsqueeze(0), grid, align_corners=False
        ).squeeze(0)

    return image.clone()


class TemporalPairDataset(Dataset):
    """Produces frame pairs with identity and motion labels.

    Each sample returns:
        - frame1: [C, H, W] original image
        - frame2: [C, H, W] transformed image (simulated next frame)
        - identity_label: object category (int)
        - motion_label: transform type (int)

    This enables probing ventral nodes for identity and dorsal nodes for motion.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 224,
        train: bool = True,
        dataset_name: str = "cifar100",
        transform_magnitude: float = 0.15,
        seed: int | None = None,
    ):
        self.image_size = image_size
        self.transform_magnitude = transform_magnitude
        self.rng = np.random.RandomState(seed)

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
            self.num_identity_classes = 100
        elif dataset_name == "cifar10":
            self.dataset = datasets.CIFAR10(
                root=root, train=train, transform=base_transform, download=True,
            )
            self.num_identity_classes = 10
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        frame1, identity_label = self.dataset[idx]

        # Random motion transform
        motion_label = self.rng.randint(0, NUM_MOTION_CLASSES)
        frame2 = apply_transform(frame1, motion_label, self.transform_magnitude)

        return {
            "frame1": frame1,
            "frame2": frame2,
            "identity_label": identity_label,
            "motion_label": motion_label,
        }


class DualStreamInputBuilder:
    """Builds per-node inputs for the dual-stream graph from frame pairs.

    Different input strategies:
    - "both_frames": all nodes see concatenated [frame1, frame2] as 6-channel input
    - "frame1_only": all nodes see frame1 (identity only, no motion signal)
    - "temporal_diff": all nodes see [frame1, frame2 - frame1] (explicit motion)
    - "per_stream": ventral nodes get frame1, dorsal nodes get frame_diff
      (this biases the result — use "both_frames" for the pure topology test)
    """

    def __init__(
        self,
        node_ids: list[int],
        strategy: str = "both_frames",
        ventral_node_ids: list[int] | None = None,
        dorsal_node_ids: list[int] | None = None,
    ):
        self.node_ids = node_ids
        self.strategy = strategy
        self.ventral_node_ids = ventral_node_ids or []
        self.dorsal_node_ids = dorsal_node_ids or []

    def build_inputs(
        self, frame1: torch.Tensor, frame2: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Build per-node input tensors from a frame pair.

        Args:
            frame1: [B, C, H, W]
            frame2: [B, C, H, W]

        Returns:
            {node_id: input_tensor} — input shape depends on strategy
        """
        if self.strategy == "both_frames":
            # Stack temporally: [B, 2C, H, W] — all nodes see both frames
            combined = torch.cat([frame1, frame2], dim=1)
            return {nid: combined for nid in self.node_ids}

        elif self.strategy == "frame1_only":
            return {nid: frame1 for nid in self.node_ids}

        elif self.strategy == "temporal_diff":
            diff = frame2 - frame1
            combined = torch.cat([frame1, diff], dim=1)
            return {nid: combined for nid in self.node_ids}

        elif self.strategy == "per_stream":
            # Ventral gets frame1 (appearance), dorsal gets frame1+diff (motion)
            diff = frame2 - frame1
            motion_input = torch.cat([frame1, diff], dim=1)
            inputs = {}
            for nid in self.node_ids:
                if nid in self.dorsal_node_ids:
                    inputs[nid] = motion_input
                else:
                    inputs[nid] = frame1
            return inputs

        raise ValueError(f"Unknown strategy: {self.strategy}")
