import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, label

from ..datasets import DatasetConfig, register
from ..loaders import medmnist_loader
from ..perturbations.synthetic import SyntheticPerturbation

BLOODMNIST_IMMATURE_GRANULOCYTE_LABEL = 3


def segment_cell(image, stain_threshold=0.05, brightness_threshold=0.6, min_nucleus_pixels=15, cytoplasm_width=3):
    """Segment the leukocyte of a blood smear into its nucleus and cytoplasm masks.

    Args:
        image: An (H, W, 3) RGB blood smear in [0, 1].
        stain_threshold: Minimum blue-minus-red channel excess for a pixel to count as stained.
        brightness_threshold: Maximum mean intensity for a pixel to count as stained.
        min_nucleus_pixels: Stained components smaller than this are debris, not a nucleus.
        cytoplasm_width: Thickness of the cytoplasm ring around the nucleus, in pixels.

    Returns:
        A `(nucleus, cytoplasm)` pair of boolean masks, or `(None, None)` if no nucleus was found.
    """
    purple = image[..., 2] - image[..., 0]
    stained = binary_fill_holes((purple > stain_threshold) & (image.mean(-1) < brightness_threshold))
    components, count = label(stained)
    if count == 0:
        return None, None

    center = np.array(image.shape[:2]) / 2.0
    nearest, nearest_distance = None, np.inf
    for i in range(1, count + 1):
        rows, columns = np.nonzero(components == i)
        if len(rows) < min_nucleus_pixels:
            continue
        distance = np.hypot(rows.mean() - center[0], columns.mean() - center[1])
        if distance < nearest_distance:
            nearest, nearest_distance = i, distance

    if nearest is None:
        return None, None

    nucleus = components == nearest
    cytoplasm = binary_dilation(nucleus, iterations=cytoplasm_width) & ~binary_dilation(nucleus, iterations=1)
    return nucleus, cytoplasm


class AuerRodPerturbation(SyntheticPerturbation):
    """Auer rods, the pathognomonic finding of acute myeloid leukaemia.

    Args:
        prior: Unused.
        noise_sigma: Standard deviation of the additive per-pixel noise.
        max_draw_failures: Largest fraction of baselines whose cell may fail to segment before `fit` raises.
        rod_counts: The possible numbers of rods in one cell.
        rod_probabilities: Probability of each entry of `rod_counts`. Most blasts carry a single rod.
        rod_length: Half-length of a rod, in pixels.
        rod_width: Half-width of a rod, in pixels: the distance over which it fades back to the baseline.
        reach: How far past the nucleus, in pixels, a rod may extend.
        color: RGB colour of a fully saturated rod.
        min_cytoplasm_pixels: Cells whose cytoplasm ring is smaller than this are declined.
    """

    def __init__(
        self,
        prior=None,
        noise_sigma=0.02,
        max_draw_failures=0.02,
        rod_counts=(1, 2, 3),
        rod_probabilities=(0.70, 0.22, 0.08),
        rod_length=1.6,
        rod_width=1.0,
        reach=4,
        color=(0.40, 0.12, 0.45),
        min_cytoplasm_pixels=8,
    ):
        super().__init__(prior=prior, noise_sigma=noise_sigma)
        self.rod_counts = np.asarray(rod_counts)
        self.rod_probabilities = np.asarray(rod_probabilities)
        self.rod_length = rod_length
        self.rod_width = rod_width
        self.reach = reach
        self.color = np.asarray(color)
        self.min_cytoplasm_pixels = min_cytoplasm_pixels

    def draw(self, baseline, rng):
        nucleus, cytoplasm = segment_cell(baseline)
        if cytoplasm is None or cytoplasm.sum() < self.min_cytoplasm_pixels:
            return None

        seed_rows, seed_columns = np.nonzero(cytoplasm)
        rows, columns = np.mgrid[: baseline.shape[0], : baseline.shape[1]].astype(float)
        within_cell = binary_dilation(nucleus, iterations=self.reach)

        mask = np.zeros_like(baseline)
        for _ in range(int(rng.choice(self.rod_counts, p=self.rod_probabilities))):
            seed = int(rng.integers(len(seed_rows)))
            row, column = seed_rows[seed], seed_columns[seed]
            angle = rng.uniform(0.0, np.pi)

            along = (columns - column) * np.cos(angle) + (rows - row) * np.sin(angle)
            across = -(columns - column) * np.sin(angle) + (rows - row) * np.cos(angle)
            rod = np.clip(1.0 - np.abs(across) / self.rod_width, 0.0, 1.0) * (np.abs(along) < self.rod_length) * within_cell

            mask = np.where(rod[..., None] > 0.0, rod[..., None] * (self.color - baseline), mask)
        return mask

BLOOD_AUER_ROD = register(
    DatasetConfig(
        key="bloodmnist_auer_rod",
        loader=medmnist_loader("bloodmnist", BLOODMNIST_IMMATURE_GRANULOCYTE_LABEL, BLOODMNIST_IMMATURE_GRANULOCYTE_LABEL),
        channels=3,
        covariate_dimension=6,
        prior=None,
        perturbation=AuerRodPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
