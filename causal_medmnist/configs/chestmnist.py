from functools import partial

import numpy as np

from ..datasets import DatasetConfig, register
from ..loaders import medmnist_loader
from ..perturbations.dilation import DilationPerturbation
from ..perturbations.localized import LocalizedPerturbation
from ..perturbations.scattered import ScatteredPerturbation
from ..prior import dark_region_prior, derived_prior, gaussian_prior, unilateral_prior

CHESTMNIST_NO_FINDING_LABEL = 0
CHESTMNIST_CARDIOMEGALY_LABEL = 1
CHESTMNIST_EFFUSION_LABEL = 2
CHESTMNIST_MASS_LABEL = 4
CHESTMNIST_NODULE_LABEL = 5

CHEST_EFFUSION = register(
    DatasetConfig(
        key="chestmnist_effusion",
        loader=medmnist_loader("chestmnist", CHESTMNIST_NO_FINDING_LABEL, CHESTMNIST_EFFUSION_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=unilateral_prior,
        perturbation=LocalizedPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

CHEST_MASS = register(
    DatasetConfig(
        key="chestmnist_mass",
        loader=medmnist_loader("chestmnist", CHESTMNIST_NO_FINDING_LABEL, CHESTMNIST_MASS_LABEL),
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
        loader=medmnist_loader("chestmnist", CHESTMNIST_NO_FINDING_LABEL, CHESTMNIST_NODULE_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=dark_region_prior,
        perturbation=ScatteredPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

CHEST_CARDIOMEGALY = register(
    DatasetConfig(
        key="chestmnist_cardiomegaly",
        loader=medmnist_loader("chestmnist", CHESTMNIST_NO_FINDING_LABEL, CHESTMNIST_CARDIOMEGALY_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=partial(gaussian_prior, center=(17, 16), sigma=(5, 4)),
        perturbation=DilationPerturbation,
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
