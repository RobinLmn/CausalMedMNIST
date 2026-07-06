import numpy as np

from ..datasets import DatasetConfig, register
from ..perturbations.localized import LocalizedPerturbation
from ..prior import unilateral_prior

PNEUMONIAMNIST_NORMAL_LABEL = 0
PNEUMONIAMNIST_PNEUMONIA_LABEL = 1

PNEUMONIA = register(
    DatasetConfig(
        key="pneumoniamnist",
        classes=(PNEUMONIAMNIST_NORMAL_LABEL, PNEUMONIAMNIST_PNEUMONIA_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=unilateral_prior,
        perturbation=LocalizedPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
