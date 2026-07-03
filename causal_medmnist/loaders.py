import os
from functools import cache

import medmnist
import numpy as np
from medmnist import INFO

_ROOT = os.path.expanduser("~/.medmnist")


@cache
def _load(key, size=28):
    os.makedirs(_ROOT, exist_ok=True)
    dataset_class = getattr(medmnist, INFO[key]["python_class"])
    dataset = dataset_class(root=_ROOT, split="train", download=True, size=size)
    images = dataset.imgs.astype(np.float64) / 255.0
    return images, dataset.labels.flatten()


def load_class_pools(key, healthy_label, disease_label, size=28):
    """Return (healthy_images, disease_images) as (count, height, width) arrays in [0, 1]."""
    images, labels = _load(key, size)
    return images[labels == healthy_label], images[labels == disease_label]


def split_pool(pool, split):
    """Split a pool into train or test halves of disjoint images."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    half = len(pool) // 2
    return pool[half:] if split == "test" else pool[:half]
