import numpy as np

from ..prior import spatial_smooth
from .base import Perturbation


class ScatteredPerturbation(Perturbation):
    """Perturbation from scattered focal deposits sampled over a disease-relevant support region.

    Args:
        prior: Location-prior, callable `(healthy, disease, n, rng) -> (H, W) | None` for the support; `None` leaves the raw contrast.
        n_deposits: `(low, high)` range for the number of deposits per unit (inclusive), drawn uniformly.
        deposit_sigma: `(low, high)` range for each deposit's radius, drawn uniformly.
        noise_sigma: Standard deviation of the additive per-pixel noise.
        color: `(low, high)` pair of RGB tints; each deposit's colour is drawn uniformly between them. `None` leaves grayscale bright deposits.
        temperature: Sharpen (`<1`) or flatten (`>1`) the support before sampling; `1.0` leaves it unchanged.
    """

    def __init__(
        self,
        prior=None,
        n_deposits=(3, 10),
        deposit_sigma=(0.8, 1.8),
        noise_sigma=0.025,
        color=None,
        temperature=1.0,
    ):
        self.prior = prior
        self.n_deposits = n_deposits
        self.deposit_sigma = deposit_sigma
        self.noise_sigma = noise_sigma
        self.color = color
        self.temperature = temperature
        self.support = None
        self.mask = None

    def fit(self, healthy, disease, baselines, rng):
        contrast = spatial_smooth(disease.mean(0) - healthy.mean(0), 1.0)
        support = np.abs(contrast).sum(-1) if contrast.ndim == 3 else np.maximum(contrast, 0.0)

        if self.prior is not None:
            support = support * self.prior(healthy, disease, n=len(baselines), rng=rng)

        support = support ** (1.0 / self.temperature)
        self.support = support / support.sum()
        self.mask = self._sample_templates(baselines, rng)

    def _sample_templates(self, baselines, rng):
        n = len(baselines)
        height, width = baselines.shape[1:3]
        rows, columns = np.ogrid[0:height, 0:width]

        counts = rng.integers(self.n_deposits[0], self.n_deposits[1] + 1, size=n)
        max_deposits = int(counts.max())

        centers = rng.choice(self.support.size, size=(n, max_deposits), p=self.support.ravel())
        center_rows, center_columns = np.unravel_index(centers, (height, width))
        sigmas = rng.uniform(self.deposit_sigma[0], self.deposit_sigma[1], size=(n, max_deposits))

        if self.color is not None:
            low, high = self.color
            colors = low + rng.uniform(0.0, 1.0, size=(n, max_deposits, 1)) * (high - low)

        templates = np.zeros((n, *baselines.shape[1:]))
        for k in range(max_deposits):
            dr = rows[None] - center_rows[:, k][:, None, None]
            dc = columns[None] - center_columns[:, k][:, None, None]
            bump = (k < counts)[:, None, None] * np.exp(-(dr**2 + dc**2) / (2.0 * sigmas[:, k][:, None, None] ** 2))
            if self.color is not None:
                templates += bump[..., None] * colors[:, k][:, None, None, :]
            else:
                templates += bump

        peak = np.abs(templates).max(axis=tuple(range(1, templates.ndim)), keepdims=True)
        return templates / np.where(peak > 0.0, peak, 1.0)
