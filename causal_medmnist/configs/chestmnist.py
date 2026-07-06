import numpy as np

from ..datasets import DatasetConfig, register
from ..perturbations.localized import LocalizedPerturbation
from ..perturbations.scattered import ScatteredPerturbation
from ..prior import derived_prior

CHESTMNIST_NO_FINDING_LABEL = 0
CHESTMNIST_EFFUSION_LABEL = 2
CHESTMNIST_MASS_LABEL = 4
CHESTMNIST_NODULE_LABEL = 5

CHEST_EFFUSION = register(
    DatasetConfig(
        key="chestmnist_effusion",
        source="chestmnist",
        classes=(CHESTMNIST_NO_FINDING_LABEL, CHESTMNIST_EFFUSION_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=LocalizedPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

CHEST_MASS = register(
    DatasetConfig(
        key="chestmnist_mass",
        source="chestmnist",
        classes=(CHESTMNIST_NO_FINDING_LABEL, CHESTMNIST_MASS_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=LocalizedPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

CHEST_NODULE = register(
    DatasetConfig(
        key="chestmnist_nodule",
        source="chestmnist",
        classes=(CHESTMNIST_NO_FINDING_LABEL, CHESTMNIST_NODULE_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=ScatteredPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
