from functools import partial

import numpy as np

from ..datasets import DatasetConfig, register
from ..loaders import medmnist_loader
from ..perturbations.composite import SequentialPerturbation
from ..perturbations.distortion import DistortionPerturbation
from ..perturbations.localized import LocalizedPerturbation
from ..prior import derived_prior

BREASTMNIST_MALIGNANT_LABEL = 0
BREASTMNIST_NORMAL_BENIGN_LABEL = 1

BREAST = register(
    DatasetConfig(
        key="breastmnist",
        loader=medmnist_loader("breastmnist", BREASTMNIST_NORMAL_BENIGN_LABEL, BREASTMNIST_MALIGNANT_LABEL),
        channels=1,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=partial(
            SequentialPerturbation,
            parts=[
                partial(LocalizedPerturbation, signed=True),
                partial(DistortionPerturbation, growth=0.06, irregularity=0.18),
            ],
        ),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
