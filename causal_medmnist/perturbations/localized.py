import numpy as np
from scipy.ndimage import gaussian_filter

from ..prior import derived_prior, gaussian_prior
from .base import Perturbation


class LocalizedPerturbation(Perturbation):
    """Localized perturbation learned from the mean contrast between a healthy and a disease class.

    Args:
        prior: Location prior that weights the contrast: "auto" derives one from data, a `dict` builds
            a Gaussian prior, `None` leaves the raw contrast.
        noise_sigma: Standard deviation of the additive per-pixel noise.
    """

    def __init__(self, prior=None, noise_sigma=0.025):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.mask = None

    def fit(self, healthy, disease) -> "LocalizedPerturbation":
        if self.prior == "auto":
            prior = derived_prior(healthy, disease)
        elif isinstance(self.prior, dict):
            prior = gaussian_prior(healthy.shape[1:], **self.prior)
        else:
            prior = None

        self.mask = build_contrast_mask(healthy, disease, prior=prior)
        return self

    def apply(self, baselines, magnitude, rng):
        if self.mask is None:
            raise RuntimeError("LocalizedPerturbation must be fit before apply is called")

        noise = rng.normal(0.0, self.noise_sigma, size=baselines.shape)
        return magnitude[:, None, None] * self.mask[None] + noise


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
