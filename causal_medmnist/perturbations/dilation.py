import numpy as np
from scipy.ndimage import affine_transform

from .base import Perturbation


class DilationPerturbation(Perturbation):
    """Morphological perturbation that enlarges a structure an image already contains.

    Args:
        prior: Location-prior `(healthy, disease, ...) -> (H, W)` marking the structure to enlarge.
        noise_sigma: Standard deviation of the additive per-pixel noise.
        strength: Reference outward scale factor of the structure (the magnitude dial scales the residual).
    """

    def __init__(self, prior=None, noise_sigma=0.025, strength=0.4):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.strength = strength
        self.mask = None

    def fit(self, healthy, disease, baselines, rng):
        region = self.prior(healthy, disease) if self.prior is not None else np.ones(baselines.shape[1:3])
        region = region / (region.max() + 1e-12)
        ys, xs = np.nonzero(region > 0.5)
        center = np.array([ys.mean(), xs.mean()])
        matrix = np.eye(2) / (1.0 + self.strength)
        offset = center - matrix @ center

        self.mask = np.zeros_like(baselines, dtype=float)
        for i, baseline in enumerate(baselines):
            structure = baseline * region
            self.mask[i] = affine_transform(structure, matrix, offset=offset, order=1) - structure
