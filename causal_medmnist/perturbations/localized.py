import numpy as np

from ..prior import spatial_smooth
from .base import Perturbation


class LocalizedPerturbation(Perturbation):
    """Localized perturbation learned from the mean contrast between a healthy and a disease class.

    Args:
        prior: Location-prior, callable `(healthy, disease, signed, n, rng) -> array | None`. `None` leaves the raw contrast.
        noise_sigma: Standard deviation of the additive per-pixel noise.
        signed: If True, keep the signed contrast, otherwise only keep the positive part.
    """

    def __init__(self, prior=None, noise_sigma=0.025, signed=False):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.signed = signed
        self.mask = None

    def fit(self, healthy, disease, baselines, rng):
        n = len(baselines)
        prior = self.prior(healthy, disease, signed=self.signed, n=n, rng=rng) if self.prior is not None else None
        difference = spatial_smooth(disease.mean(0) - healthy.mean(0), 1.0)

        if prior is not None and prior.ndim == 3:
            self.mask = np.stack([self._mask(difference, prior[i]) for i in range(n)])
        else:
            self.mask = self._mask(difference, prior)[None]

    def _mask(self, difference, prior):
        if prior is not None:
            difference = difference * (prior[..., None] if difference.ndim == 3 else prior)

        kept = difference if self.signed else np.maximum(difference, 0.0)
        magnitude = np.abs(kept)
        spatial = magnitude.sum(-1) if magnitude.ndim == 3 else magnitude
        threshold = float(np.percentile(spatial[spatial > 0], 85))
        region = spatial >= threshold
        kept = np.where(region[..., None] if difference.ndim == 3 else region, kept, 0.0)

        mask = spatial_smooth(kept, 1.0)
        return mask / (np.abs(mask).max() + 1e-12)
