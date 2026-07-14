import numpy as np

from .base import Perturbation


class DesaturationPerturbation(Perturbation):
    """Colour perturbation that drains a lesion's colour toward gray.

    Args:
        prior: Location-prior callable `(healthy, disease, ...) -> (H, W) | None` for the lesion region.
        noise_sigma: Standard deviation of the additive per-pixel noise.
    """

    def __init__(self, prior=None, noise_sigma=0.025):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.mask = None

    def fit(self, healthy, disease, baselines, rng):
        region = self.prior(healthy, disease) if self.prior is not None else np.ones(baselines.shape[1:3])
        pigment = np.clip(1.0 - baselines.mean(-1), 0.0, 1.0)
        weight = (region * pigment)[..., None]

        luminance = baselines.mean(-1, keepdims=True)
        self.mask = weight * (luminance - baselines)
