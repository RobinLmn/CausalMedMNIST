from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .perturbations.base import Perturbation


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    loader: Callable[..., tuple[np.ndarray, np.ndarray]]
    channels: int
    covariate_dimension: int
    prior: Callable | None
    perturbation: Callable[..., Perturbation]
    treatment_coefficients: np.ndarray
    outcome_coefficients: np.ndarray


REGISTRY: dict[str, DatasetConfig] = {}


def register(config):
    if config.key in REGISTRY:
        raise ValueError(f"Dataset {config.key!r} is already registered")
    REGISTRY[config.key] = config
    return config


def get_config(dataset):
    if isinstance(dataset, DatasetConfig):
        return dataset
    if dataset not in REGISTRY:
        raise KeyError(f"Unknown dataset {dataset!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[dataset]
