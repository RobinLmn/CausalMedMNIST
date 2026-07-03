from ..masks import build_contrast_mask, gaussian_prior, derived_prior
from .base import Perturbation


class ClassContrastPerturbation(Perturbation):
    """Additive lesion learned from the mean contrast between a healthy and a disease class."""

    def __init__(self, prior=None, noise_sigma=0.025):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.mask = None

    def fit(self, healthy, disease) -> "ClassContrastPerturbation":
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
            raise RuntimeError("ClassContrastPerturbation must be fit before apply is called")
        
        noise = rng.normal(0.0, self.noise_sigma, size=baselines.shape)
        return magnitude[:, None, None] * self.mask[None] + noise
