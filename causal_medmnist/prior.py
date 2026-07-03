import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_prior(shape, center, sigma):
    """A 2D Gaussian bump over an image grid, used as an anatomical location prior.

    Args:
        shape: (height, width) of the image.
        center: (row, column) of the bump's peak.
        sigma: (row_spread, column_spread) of the bump.

    Returns:
        A (height, width) array in [0, 1] peaking at `center`.
    """
    rows, columns = np.ogrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-0.5 * (((rows - center[0]) / sigma[0]) ** 2 + ((columns - center[1]) / sigma[1]) ** 2))


def derived_prior(healthy, disease, margin=3):
    """Derive an anatomical location prior from data by moment-matching a Gaussian.

    Fits an axis-aligned Gaussian whose center and per-axis spread are the first and second
    moments of the (border-masked) positive contrast between the disease and healthy pools, so
    the prior localizes itself to wherever the two classes differ, with no hand-tuning.

    Args:
        healthy: (count, height, width) images of the healthy class.
        disease: (count, height, width) images of the disease class.
        margin: Width of the image border zeroed before fitting, to suppress framing artifacts
            that would otherwise inflate the fitted spread.

    Returns:
        A (height, width) Gaussian prior in [0, 1], peaking at the contrast's center of mass.
    """
    positive = np.maximum(gaussian_filter(disease.mean(0) - healthy.mean(0), sigma=1.0), 0.0)

    if margin:
        positive[:margin], positive[-margin:], positive[:, :margin], positive[:, -margin:] = 0.0, 0.0, 0.0, 0.0

    weights = positive / positive.sum()
    rows = np.arange(positive.shape[0])
    columns = np.arange(positive.shape[1])
    center = ((weights.sum(1) * rows).sum(), (weights.sum(0) * columns).sum())
    sigma = (np.sqrt((weights.sum(1) * (rows - center[0]) ** 2).sum()), np.sqrt((weights.sum(0) * (columns - center[1]) ** 2).sum()),)
    
    return gaussian_prior(positive.shape, center, sigma)


def build_contrast_mask(healthy, disease, prior=None, percentile=85.0, smooth_sigma=1.0):
    """Learn a disease-signature mask from the contrast between two class pools.

    Takes the smoothed mean difference between the disease and healthy pools, keeps
    the positive part, optionally weights it by an anatomical prior, thresholds to the
    strongest pixels, and smooths the result.

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
    positive = np.maximum(difference, 0.0)

    if prior is not None:
        positive = positive * prior

    positive_values = positive[positive > 0]
    threshold = float(np.percentile(positive_values, percentile))
    mask = gaussian_filter(np.where(positive >= threshold, positive, 0.0), sigma=smooth_sigma)
    mask = np.maximum(mask, 0.0)
    mask = mask / (mask.max() + 1e-12)

    return mask
