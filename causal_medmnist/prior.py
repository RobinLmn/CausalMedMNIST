import numpy as np
from scipy.ndimage import gaussian_filter


def spatial_smooth(image, sigma):
    """Gaussian-smooth spatially; a trailing channel axis (RGB) is left unblurred."""
    return gaussian_filter(image, (sigma, sigma, 0) if image.ndim == 3 else sigma)


def _gaussian_bump(shape, center, sigma):
    rows, columns = np.ogrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-0.5 * (((rows - center[0]) / sigma[0]) ** 2 + ((columns - center[1]) / sigma[1]) ** 2))


def gaussian_prior(healthy, disease, signed=False, n=None, rng=None, *, center, sigma):
    """Location prior that is a fixed 2D Gaussian bump at `center` with spread `sigma`.

    Bind `center` and `sigma` in the config, e.g. `prior=partial(gaussian_prior, center=(14, 13), sigma=(3, 3))`.

    Args:
        healthy, disease: Class pools; only `healthy`'s image shape is used.
        signed, n, rng: Accepted for the shared prior signature but unused.
        center: (row, column) of the bump's peak.
        sigma: (row_spread, column_spread) of the bump.

    Returns:
        A (height, width) array in [0, 1] peaking at `center`.
    """
    return _gaussian_bump(healthy.shape[1:3], center, sigma)


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
    return _gaussian_bump(positive.shape, center, sigma)


def dark_region_prior(healthy, disease, signed=False, n=None, rng=None):
    """Location prior favouring the darker (lower-intensity) regions of the healthy image."""
    brightness = gaussian_filter(healthy.mean(0), sigma=1.0)
    dark = np.maximum(brightness.max() - brightness, 0.0)
    return dark / (dark.max() + 1e-12)


def unilateral_prior(healthy, disease, signed=False, n=None, rng=None):
    """Per-unit prior that keeps only the left or right half of a derived prior: one side drawn per unit."""
    base = derived_prior(healthy, disease, signed=signed)
    half = base.shape[1] // 2
    left, right = base.copy(), base.copy()
    left[:, half:] = 0.0
    right[:, :half] = 0.0
    sides = np.stack([left, right])
    return sides[rng.integers(0, 2, size=n)]
