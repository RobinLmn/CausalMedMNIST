import numpy as np
from scipy.ndimage import gaussian_filter


def spatial_smooth(image, sigma):
    """Gaussian-smooth spatially; a trailing channel axis (RGB) is left unblurred."""
    return gaussian_filter(image, (sigma, sigma, 0) if image.ndim == 3 else sigma)


def gaussian_prior(shape, center, sigma):
    """A 2D Gaussian bump over an image grid, used as a location prior.

    Args:
        shape: (height, width) of the image.
        center: (row, column) of the bump's peak.
        sigma: (row_spread, column_spread) of the bump.

    Returns:
        A (height, width) array in [0, 1] peaking at `center`.
    """
    rows, columns = np.ogrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-0.5 * (((rows - center[0]) / sigma[0]) ** 2 + ((columns - center[1]) / sigma[1]) ** 2))


def derived_prior(healthy, disease, signed=False, n=None, rng=None):
    """Derive a location prior from data by moment-matching a Gaussian.

    Args:
        healthy: (count, height, width) images of the healthy class.
        disease: (count, height, width) images of the disease class.
        signed: If True, fit to the contrast magnitude rather than its positive part.
        n, rng: Unused

    Returns:
        A (height, width) Gaussian prior in [0, 1], peaking at the contrast's center of mass.
    """
    contrast = disease.mean(0) - healthy.mean(0)
    if contrast.ndim == 3:
        contrast = np.abs(contrast).sum(-1)
    elif signed:
        contrast = np.abs(contrast)
    positive = np.maximum(gaussian_filter(contrast, sigma=1.0), 0.0)

    weights = positive / positive.sum()
    rows = np.arange(positive.shape[0])
    columns = np.arange(positive.shape[1])

    center = ((weights.sum(1) * rows).sum(), (weights.sum(0) * columns).sum())
    sigma = (np.sqrt((weights.sum(1) * (rows - center[0]) ** 2).sum()), np.sqrt((weights.sum(0) * (columns - center[1]) ** 2).sum()))
    return gaussian_prior(positive.shape, center, sigma)


def unilateral_prior(healthy, disease, signed=False, n=None, rng=None):
    """Per-unit prior that keeps only the left or right half of a derived prior: one side drawn per unit."""
    base = derived_prior(healthy, disease, signed=signed)
    half = base.shape[1] // 2
    left, right = base.copy(), base.copy()
    left[:, half:] = 0.0
    right[:, :half] = 0.0
    sides = np.stack([left, right])
    return sides[rng.integers(0, 2, size=n)]
