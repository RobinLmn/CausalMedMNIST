import numpy as np
from scipy.ndimage import binary_fill_holes, label, map_coordinates

from .base import Perturbation


def _radial_warp(field, center, radius, rng, growth, irregularity):
    """Push the boundary of a region outward per angle and resample the field over the new coordinates."""
    cy, cx = center
    yy, xx = np.mgrid[: field.shape[0], : field.shape[1]].astype(float)
    r = np.hypot(yy - cy, xx - cx) + 1e-6
    theta = np.arctan2(yy - cy, xx - cx)

    profile = np.zeros_like(theta)
    for harmonic in rng.choice(np.arange(3, 12), size=7, replace=False):
        profile += rng.uniform(-1, 1) / harmonic * np.cos(harmonic * theta + rng.uniform(0, 2 * np.pi))
    profile /= np.abs(profile).max() + 1e-9

    window = np.exp(-((np.maximum(r - radius, 0.0) / (0.7 * radius)) ** 2))
    scale = 1.0 + (growth + irregularity * profile) * window
    coords = np.stack([cy + (r / scale) * np.sin(theta), cx + (r / scale) * np.cos(theta)])

    if field.ndim == 3:
        return np.stack([map_coordinates(field[..., k], coords, order=1, mode="nearest") for k in range(field.shape[-1])], axis=-1)
    return map_coordinates(field, coords, order=1, mode="nearest")


def _region(support):
    ys, xs = np.nonzero(support)
    return (ys.mean(), xs.mean()), np.sqrt(support.sum() / np.pi)


def _segment(image):
    """Largest dark, filled blob containing (or nearest to) the image centre; None if too small."""
    gray = image.mean(-1) if image.ndim == 3 else image
    dark = binary_fill_holes(gray < np.percentile(gray, 40))
    labels, count = label(dark)
    if count == 0:
        return None
    component = labels[gray.shape[0] // 2, gray.shape[1] // 2]
    if component == 0:
        component = 1 + int(np.argmax([(labels == i).sum() for i in range(1, count + 1)]))
    lesion = labels == component
    return lesion if lesion.sum() >= 6 else None


class DistortionPerturbation(Perturbation):
    """Morphological perturbation that warps a lesion's own boundary outward (growth + irregular margin).

    Args:
        prior: Unused
        noise_sigma: Standard deviation of the additive per-pixel noise.
        growth: Uniform outward expansion of the boundary.
        irregularity: Amplitude of the per-angle wobble that ragged-edges the boundary.
    """

    def __init__(self, prior=None, noise_sigma=0.025, growth=0.35, irregularity=0.7):
        self.prior = prior
        self.noise_sigma = noise_sigma
        self.growth = growth
        self.irregularity = irregularity
        self.mask = None

    def fit(self, healthy, disease, baselines, rng):
        self.mask = np.zeros_like(baselines, dtype=float)
        for i, baseline in enumerate(baselines):
            lesion = _segment(baseline)
            if lesion is not None:
                self.mask[i] = _radial_warp(baseline, *_region(lesion), rng, self.growth, self.irregularity) - baseline
