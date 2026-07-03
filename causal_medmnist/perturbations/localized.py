import numpy as np
from scipy.ndimage import gaussian_filter, label

from ..prior import derived_prior, gaussian_prior
from .base import Perturbation


class LocalizedPerturbation(Perturbation):
    """Localized perturbation learned from the mean contrast between a healthy and a disease class.

    Args:
        prior: Location prior that weights the contrast: "auto" derives one from data, a `dict` builds a Gaussian prior, `None` leaves the
            raw contrast.
        noise_sigma: Standard deviation of the additive per-pixel noise.
        select: If set, split the mask into its connected regions and activate only a random subset per unit. Either an int for a fixed number of 
            regions, or a `(low, high)` tuple for a per-unit count.  `None` uses the whole mask for all samples.
    """

    def __init__(self, prior=None, noise_sigma=0.025, select=None):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.select = select
        self.mask = None
        self._components = None
        self._cache_baselines = None
        self._cache_masks = None

    def fit(self, healthy, disease) -> "LocalizedPerturbation":
        if self.prior == "auto":
            prior = derived_prior(healthy, disease)
        elif isinstance(self.prior, dict):
            prior = gaussian_prior(healthy.shape[1:], **self.prior)
        else:
            prior = None

        self.mask = build_contrast_mask(healthy, disease, prior=prior)
        if self.select is not None:
            self._components = _connected_regions(self.mask)
        return self

    def apply(self, baselines, magnitude, rng):
        if self.mask is None:
            raise RuntimeError("LocalizedPerturbation must be fit before apply is called")

        if self.select is None:
            field = self.mask[None]
        else:
            if baselines is not self._cache_baselines:
                self._cache_masks = self._select_masks(len(baselines), rng)
                self._cache_baselines = baselines
            field = self._cache_masks

        noise = rng.normal(0.0, self.noise_sigma, size=baselines.shape)
        return magnitude[:, None, None] * field + noise

    def _select_masks(self, n, rng):
        components = self._components
        num = len(components)
        if isinstance(self.select, int):
            counts = np.full(n, min(self.select, num))
        else:
            low, high = self.select
            counts = rng.integers(low, min(high, num) + 1, size=n)

        masks = np.empty((n, *self.mask.shape))
        for i in range(n):
            chosen = rng.choice(num, size=counts[i], replace=False)
            combined = components[chosen].sum(0)
            masks[i] = combined / (combined.max() + 1e-12)
        return masks


def _connected_regions(mask, threshold=0.05):
    """Split a mask into its connected regions, returned as a (regions, height, width) stack."""
    labels, count = label(mask > threshold)
    if count == 0:
        return mask[None]
    return np.stack([mask * (labels == i) for i in range(1, count + 1)])


def build_contrast_mask(healthy, disease, prior=None, percentile=85.0, smooth_sigma=1.0):
    """Learn a disease-signature mask from the contrast between two class pools.

    Args:
        healthy: (count, height, width) images of the healthy class.
        disease: (count, height, width) images of the disease class.
        prior: Optional (height, width) anatomical prior to weight the signature by.
        percentile: Keep pixels above this percentile of positive values.
        smooth_sigma: Gaussian smoothing applied to the thresholded mask.

    Returns:
        A (height, width) mask in [0, 1], normalized to peak at 1.
    """
    difference = gaussian_filter(disease.mean(0) - healthy.mean(0), sigma=1.0)
    if prior is not None:
        difference = difference * prior

    threshold = float(np.percentile(difference[difference > 0], percentile))
    mask = gaussian_filter(np.where(difference >= threshold, difference, 0.0), sigma=smooth_sigma)
    mask = mask / (mask.max() + 1e-12)

    return mask
