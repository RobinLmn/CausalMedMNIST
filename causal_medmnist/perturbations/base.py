from abc import ABC, abstractmethod

import numpy as np


class Perturbation(ABC):
    """A synthetic perturbation fitted from data and applied to baseline images."""

    @abstractmethod
    def fit(self, healthy, disease, baselines, rng):
        """Sample `self.mask`, the per-unit disease signature, from the class pools and baselines."""

    def apply(self, magnitude, rng) -> np.ndarray:
        """Scale the fitted mask by each unit's magnitude and add per-pixel noise."""
        noise = rng.normal(0.0, self.noise_sigma, size=(len(magnitude), *self.mask.shape[1:]))
        return magnitude.reshape((-1,) + (1,) * (self.mask.ndim - 1)) * self.mask + noise
