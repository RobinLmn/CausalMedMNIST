import os
from functools import cache

import medmnist
import numpy as np
from medmnist import INFO

_ROOT = os.path.expanduser("~/.medmnist")
SPLITS = ("train", "val", "test")


@cache
def _load(key, size=28):
    os.makedirs(_ROOT, exist_ok=True)
    dataset_class = getattr(medmnist, INFO[key]["python_class"])
    images, labels, splits = [], [], []
    for split in SPLITS:
        dataset = dataset_class(root=_ROOT, split=split, download=True, size=size)
        images.append(dataset.imgs.astype(np.float64) / 255.0)
        label = dataset.labels
        labels.append(label.flatten() if label.shape[1] == 1 else label)
        splits.append(np.full(len(dataset.labels), split))
    return np.concatenate(images), np.concatenate(labels), np.concatenate(splits)


def load_class_pools(key, healthy_label, disease_label, split=None, size=28):
    """Return (healthy_images, disease_images) as (count, height, width) arrays in [0, 1]."""
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    images, labels, splits = _load(key, size)
    if split is not None:
        keep = splits == split
        images, labels = images[keep], labels[keep]

    if labels.ndim == 1:
        return images[labels == healthy_label], images[labels == disease_label]

    return images[labels.sum(1) == 0], images[labels[:, disease_label] == 1]


def medmnist_loader(key, healthy_label, disease_label, size=28):
    """Build a loader `(split) -> (healthy_images, disease_images)` for MedMNIST datasets."""
    def load(split=None):
        return load_class_pools(key, healthy_label, disease_label, split=split, size=size)
    return load
