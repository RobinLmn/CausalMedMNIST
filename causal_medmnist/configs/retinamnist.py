from functools import partial

import numpy as np

from ..datasets import DatasetConfig, register
from ..perturbations.scattered import ScatteredPerturbation
from ..prior import derived_prior

RETINAMNIST_HEALTHY_LABEL = 0
RETINAMNIST_MILD_LABEL = 1
RETINAMNIST_MODERATE_LABEL = 2
RETINAMNIST_SEVERE_LABEL = 3
RETINAMNIST_PROLIFERATIVE_LABEL = 4
RETINAMNIST_DISEASE_LABEL = RETINAMNIST_SEVERE_LABEL

RETINA_MILD = register(
    DatasetConfig(
        key="retinamnist_mild",
        source="retinamnist",
        classes=(RETINAMNIST_HEALTHY_LABEL, RETINAMNIST_MILD_LABEL),
        channels=3,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=partial(ScatteredPerturbation, color=np.array([[-0.15, -0.9, -0.8], [-0.5, -1.0, -0.9]])),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

RETINA_MODERATE = register(
    DatasetConfig(
        key="retinamnist_moderate",
        source="retinamnist",
        classes=(RETINAMNIST_HEALTHY_LABEL, RETINAMNIST_MODERATE_LABEL),
        channels=3,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=partial(ScatteredPerturbation, color=np.array([[-0.15, -0.9, -0.8], [-0.5, -1.0, -0.9]])),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

RETINA_SEVERE = register(
    DatasetConfig(
        key="retinamnist_severe",
        source="retinamnist",
        classes=(RETINAMNIST_HEALTHY_LABEL, RETINAMNIST_SEVERE_LABEL),
        channels=3,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=partial(ScatteredPerturbation, color=np.array([[-0.15, -0.9, -0.8], [-0.5, -1.0, -0.9]])),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

RETINA_PROLIFERATIVE = register(
    DatasetConfig(
        key="retinamnist_proliferative",
        source="retinamnist",
        classes=(RETINAMNIST_HEALTHY_LABEL, RETINAMNIST_PROLIFERATIVE_LABEL),
        channels=3,
        covariate_dimension=6,
        prior=derived_prior,
        perturbation=partial(ScatteredPerturbation, color=np.array([[-0.15, -0.9, -0.8], [-0.5, -1.0, -0.9]])),
        treatment_coefficients=np.array([0.90, -0.75, 0.55, -0.40, 0.30, -0.20]),
        outcome_coefficients=np.array([0.60, -0.50, 0.40, -0.30, 0.20, -0.10]),
    )
)

RETINA = RETINA_SEVERE
