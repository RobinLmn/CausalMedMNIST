from abc import ABC, abstractmethod

import numpy as np


class Perturbation(ABC):
    """A disease-signature generator that turns a per-unit magnitude into image changes."""

    @abstractmethod
    def fit(self, healthy, disease) -> "Perturbation":
        """Learn the disease signature from the two real class pools."""

    @abstractmethod
    def apply(self, baselines, magnitude, rng) -> np.ndarray:
        """Return the per-unit signature field to add onto the baseline images."""
