from functools import partial

import numpy as np

from ..datasets import DatasetConfig, register
from ..perturbations.localized import LocalizedPerturbation
from ..perturbations.scattered import ScatteredPerturbation
from ..prior import derived_prior

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
        prior=derived_prior,
        perturbation=LocalizedPerturbation,
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
        prior=derived_prior,
        perturbation=partial(LocalizedPerturbation, signed=True),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

OCTMNIST_DRUSEN = register(
    DatasetConfig(
        key="octmnist_drusen",
        source="octmnist",
        classes=(OCTMNIST_NORMAL_LABEL, OCTMNIST_DRUSEN_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=partial(ScatteredPerturbation, temperature=0.5),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
