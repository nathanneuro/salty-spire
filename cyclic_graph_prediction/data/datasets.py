"""
Dataset wrappers for cyclic graph prediction experiments.

Wraps standard vision datasets to produce masked multi-view inputs suitable
for the prediction graph, where each node sees a different subset of patches.
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .masking import PatchMaskGenerator


class MaskedMultiViewDataset(Dataset):
    """Wraps a standard image dataset to produce per-node masked views.

    Each __getitem__ returns:
        - The original image tensor
        - A dict mapping node_id -> masked image tensor
    """

    def __init__(
        self,
        root: str,
        num_nodes: int = 4,
        visible_fraction: float = 0.25,
        image_size: int = 224,
        patch_size: int = 16,
        train: bool = True,
        dataset_name: str = "cifar100",
    ):
        self.num_nodes = num_nodes
        self.image_size = image_size
        self.patch_size = patch_size

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
                root=root, train=train, transform=base_transform, download=True
            )
        elif dataset_name == "cifar10":
            self.dataset = datasets.CIFAR10(
                root=root, train=train, transform=base_transform, download=True
            )
        elif dataset_name == "imagenet":
            split = "train" if train else "val"
            self.dataset = datasets.ImageFolder(
                root=f"{root}/{split}", transform=base_transform
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.mask_gen = PatchMaskGenerator(
            image_size=image_size,
            patch_size=patch_size,
            num_nodes=num_nodes,
            visible_fraction=visible_fraction,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[int, torch.Tensor], int]:
        image, label = self.dataset[idx]  # [C, H, W]

        # Generate masks for this sample
        patch_masks = self.mask_gen()  # list of [total_patches] bool

        n = self.image_size // self.patch_size
        masked_views = {}
        for node_id, pmask in enumerate(patch_masks):
            grid = pmask.float().reshape(1, n, n)
            pixel_mask = grid.repeat_interleave(self.patch_size, dim=1).repeat_interleave(
                self.patch_size, dim=2
            )  # [1, H, W]
            masked_views[node_id] = image * pixel_mask

        return image, masked_views, label
