from functools import partial

import numpy as np

from ..datasets import DatasetConfig, register
from ..loaders import medmnist_loader
from ..perturbations.composite import CompositePerturbation
from ..perturbations.desaturation import DesaturationPerturbation
from ..perturbations.distortion import DistortionPerturbation
from ..prior import derived_prior

DERMAMNIST_MELANOMA_LABEL = 4
DERMAMNIST_NEVUS_LABEL = 5

DERMA = register(
    DatasetConfig(
        key="dermamnist",
        loader=medmnist_loader("dermamnist", DERMAMNIST_NEVUS_LABEL, DERMAMNIST_MELANOMA_LABEL),
        channels=3,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=partial(CompositePerturbation, parts=[DesaturationPerturbation, DistortionPerturbation]),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)
