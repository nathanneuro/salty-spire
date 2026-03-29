"""
Patch masking utilities for multi-node input partitioning.

Each node in the graph sees a different subset of input patches. This module
generates complementary masks so that collectively all patches are covered
but each node has a partial view, creating a mutual-information incentive
for cross-node prediction.
"""

import torch
import numpy as np


class PatchMaskGenerator:
    """Generate patch-level masks for image inputs.

    Divides an image into a grid of patches and assigns each patch to one
    or more nodes, ensuring each node sees a specified fraction of patches.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_nodes: int = 4,
        visible_fraction: float = 0.25,
        overlap: bool = False,
        seed: int | None = None,
    ):
        self.num_patches_per_side = image_size // patch_size
        self.total_patches = self.num_patches_per_side ** 2
        self.num_nodes = num_nodes
        self.visible_fraction = visible_fraction
        self.overlap = overlap
        self.rng = np.random.RandomState(seed)

    def __call__(self) -> list[torch.Tensor]:
        """Generate a set of boolean masks, one per node.

        Returns:
            List of [total_patches] boolean tensors. True = visible to that node.
        """
        num_visible = int(self.total_patches * self.visible_fraction)

        if not self.overlap:
            # Partition patches across nodes without overlap
            perm = self.rng.permutation(self.total_patches)
            masks = []
            for i in range(self.num_nodes):
                mask = torch.zeros(self.total_patches, dtype=torch.bool)
                start = i * num_visible
                end = min(start + num_visible, self.total_patches)
                indices = perm[start:end]
                mask[indices] = True
                masks.append(mask)
        else:
            # Each node gets an independent random subset (may overlap)
            masks = []
            for _ in range(self.num_nodes):
                indices = self.rng.choice(
                    self.total_patches, size=num_visible, replace=False
                )
                mask = torch.zeros(self.total_patches, dtype=torch.bool)
                mask[torch.from_numpy(indices)] = True
                masks.append(mask)

        return masks


def create_node_masks(
    batch_size: int,
    image_size: int = 224,
    patch_size: int = 16,
    num_nodes: int = 4,
    visible_fraction: float = 0.25,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    """Create a batch of masks for all nodes.

    For simplicity, uses the same mask partition for the entire batch
    (different across calls due to random state).

    Returns:
        List of [B, C, H, W] float tensors that can be multiplied with images
        to zero out invisible patches.
    """
    gen = PatchMaskGenerator(
        image_size=image_size,
        patch_size=patch_size,
        num_nodes=num_nodes,
        visible_fraction=visible_fraction,
    )
    patch_masks = gen()  # list of [total_patches] bool

    n = image_size // patch_size
    spatial_masks = []
    for pmask in patch_masks:
        # Reshape to 2D grid and upsample to pixel resolution
        grid = pmask.float().reshape(1, 1, n, n)
        # Repeat each patch to cover patch_size x patch_size pixels
        pixel_mask = grid.repeat_interleave(patch_size, dim=2).repeat_interleave(
            patch_size, dim=3
        )
        # Expand to batch
        pixel_mask = pixel_mask.expand(batch_size, 3, image_size, image_size)
        spatial_masks.append(pixel_mask.to(device))

    return spatial_masks
