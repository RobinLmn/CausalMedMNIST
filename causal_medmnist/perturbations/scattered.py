import numpy as np
from scipy.ndimage import gaussian_filter

from ..prior import derived_prior, gaussian_prior
from .base import Perturbation


class ScatteredPerturbation(Perturbation):
    """Perturbation from scattered focal deposits sampled over a disease-relevant support region.

    Args:
        prior: Location prior gating the support; "auto" derives one from data, a dict builds a manual Gaussian, None leaves the raw contrast.
        n_deposits: Deposits per unit. Either an int for a fixed count, or a `(low, high)` tuple to draw a per-unit count uniformly.
        deposit_sigma: Radius of each Gaussian deposit. Either a scalar for a fixed radius, or a `(low, high)` tuple to draw each radius uniformly.
        noise_sigma: Standard deviation of the additive per-pixel noise.
    """

    def __init__(self, prior=None, n_deposits=(3, 10), deposit_sigma=(0.8, 1.8), noise_sigma=0.025):
        self.prior = prior
        self.n_deposits = n_deposits
        self.deposit_sigma = deposit_sigma
        self.noise_sigma = noise_sigma
        self.support = None
        self._cache_baselines = None
        self._cache_templates = None

    def fit(self, healthy, disease) -> "ScatteredPerturbation":
        support = np.maximum(gaussian_filter(disease.mean(0) - healthy.mean(0), sigma=1.0), 0.0)

        if self.prior == "auto":
            support = support * derived_prior(healthy, disease)
        elif isinstance(self.prior, dict):
            support = support * gaussian_prior(healthy.shape[1:], **self.prior)

        self.support = support / support.sum()
        return self

    def apply(self, baselines, magnitude, rng):
        if self.support is None:
            raise RuntimeError("ScatteredPerturbation must be fit before apply is called")

        if baselines is not self._cache_baselines:
            self._cache_templates = self._sample_templates(baselines, rng)
            self._cache_baselines = baselines

        noise = rng.normal(0.0, self.noise_sigma, size=baselines.shape)
        return magnitude[:, None, None] * self._cache_templates + noise

    def _sample_templates(self, baselines, rng):
        n = len(baselines)
        height, width = baselines.shape[1:]
        rows, columns = np.ogrid[0:height, 0:width]

        if isinstance(self.n_deposits, int):
            counts = np.full(n, self.n_deposits)
        else:
            low, high = self.n_deposits
            counts = rng.integers(low, high + 1, size=n)
        max_deposits = int(counts.max())

        centers = rng.choice(self.support.size, size=(n, max_deposits), p=self.support.ravel())
        center_rows, center_columns = np.unravel_index(centers, (height, width))

        if isinstance(self.deposit_sigma, (int, float)):
            sigmas = np.full((n, max_deposits), float(self.deposit_sigma))
        else:
            low, high = self.deposit_sigma
            sigmas = rng.uniform(low, high, size=(n, max_deposits))

        templates = np.zeros((n, height, width))
        for k in range(max_deposits):
            active = (k < counts)[:, None, None]
            dr = rows[None] - center_rows[:, k][:, None, None]
            dc = columns[None] - center_columns[:, k][:, None, None]
            templates += active * np.exp(-(dr**2 + dc**2) / (2.0 * sigmas[:, k][:, None, None] ** 2))

        peak = templates.max(axis=(1, 2), keepdims=True)
        return templates / np.where(peak > 0.0, peak, 1.0)
