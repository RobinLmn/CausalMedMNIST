import numpy as np

from ..datasets import DatasetConfig, register
from ..perturbations.class_contrast import ClassContrastPerturbation

OCTMNIST_CNV_LABEL = 0
OCTMNIST_DME_LABEL = 1
OCTMNIST_DRUSEN_LABEL = 2
OCTMNIST_NORMAL_LABEL = 3

OCTMNIST_CNV = register(
    DatasetConfig(
        key="octmnist_cnv",
        source="octmnist",
        classes=(OCTMNIST_NORMAL_LABEL, OCTMNIST_CNV_LABEL),
        channels=1,
        covariate_dimension=6,
        prior="auto",
        perturbation=ClassContrastPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

OCTMNIST_DME = register(
    DatasetConfig(
        key="octmnist_dme",
        source="octmnist",
        classes=(OCTMNIST_NORMAL_LABEL, OCTMNIST_DME_LABEL),
        channels=1,
        covariate_dimension=6,
        prior="auto",
        perturbation=ClassContrastPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
