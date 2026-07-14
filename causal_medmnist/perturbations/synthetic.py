from abc import abstractmethod

import numpy as np

from .base import Perturbation


class SyntheticPerturbation(Perturbation):
    """A hand-authored lesion drawn onto each baseline instead of learned from a disease class.

    Args:
        prior: Unused
        noise_sigma: Standard deviation of the additive per-pixel noise.
    """

    def __init__(self, prior=None, noise_sigma=0.02):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.mask = None

    @abstractmethod
    def draw(self, baseline, rng):
        """Render the lesion for one baseline, or return `None` if it cannot be drawn on that image."""

    def fit(self, healthy, disease, baselines, rng):
        self.mask = np.zeros_like(baselines, dtype=float)
        for i, baseline in enumerate(baselines):
            drawn = self.draw(baseline, rng)
            if drawn is not None:
                self.mask[i] = drawn
