from dataclasses import dataclass

import numpy as np

from .datasets import get_config
from .loaders import load_class_pools, split_pool
from .utils import default, sigmoid


@dataclass
class Sample:
    """A generated causal dataset with ground-truth potential outcomes.

    Attributes:
        X: (n, covariate_dimension) confounders.
        Y: (n, height, width) observed images.
        A: (n,) binary treatment assignments.
        propensity: (n,) treatment probabilities P(A=1 | X).
        Y0: (n, height, width) control potential outcomes.
        Y1: (n, height, width) treated potential outcomes.
    """

    X: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    propensity: np.ndarray
    Y0: np.ndarray | None = None
    Y1: np.ndarray | None = None


class Scenario:
    """A causal scenario from which datasets can be generated.

    Wraps a MedMNIST dataset in causal model where confounders `X` drive both the treatment `A` and
    the outcome `Y`. Draws samples with known ground-truth potential outcomes via the `generate`
    method.

    Args:
        dataset: A registered dataset configuration or its key string.
        effect_strength: Strength of the treatment effect; `0.0` is the null with no effect.
        confounding_strength: Scales the treatment and outcome coefficients; `0.0` is a randomized
            trial with no confounding.
        propensity_clipping: (low, high) bounds on P(A=1 | X); enforces overlap.
        covariate_dimension: Dimension of the confounders `X`; defaults to the dataset's value.
        treatment_coefficients: Override for the treatment weights; defaults to a decaying
            alternating-sign vector.
        outcome_coefficients: Override for the outcome weights; defaults to a decaying
            alternating-sign vector.
        center_effect: If True, center the potential outcomes so the mean effect is zero and the
            signal lives in the heterogeneity; if False, keep a genuine non-zero average effect.
        seed: Default random seed; `generate(seed=...)` can override it per draw.

    Examples:
        >>> scenario = Scenario(OCTMNIST, effect_strength=0.6)
        >>> sample = scenario.generate(n=1000, seed=0)
    """

    def __init__(
        self,
        dataset,
        *,
        effect_strength=0.6,
        confounding_strength=1.0,
        propensity_clipping=(0.07, 0.93),
        covariate_dimension=None,
        treatment_coefficients=None,
        outcome_coefficients=None,
        center_effect=True,
        seed=None,
    ):
        self.config = get_config(dataset)
        self.effect_strength = effect_strength
        self.confounding_strength = confounding_strength
        self.propensity_clipping = propensity_clipping
        self.covariate_dimension = default(covariate_dimension, self.config.covariate_dimension)
        self.treatment_coefficients = default(treatment_coefficients, self.config.treatment_coefficients)
        self.outcome_coefficients = default(outcome_coefficients, self.config.outcome_coefficients)
        self.center_effect = center_effect
        self.seed = seed
        self._perturbation = None

        if len(self.treatment_coefficients) != self.covariate_dimension:
            raise ValueError(f"treatment_coefficients has length {len(self.treatment_coefficients)}, expected: {self.covariate_dimension}")
        if len(self.outcome_coefficients) != self.covariate_dimension:
            raise ValueError(f"outcome_coefficients has length {len(self.outcome_coefficients)}, expected: {self.covariate_dimension}")

    def generate(self, n, *, seed=None, split="train"):
        """Sample a dataset from this scenario's causal model.

        Draws confounders `X`, assigns treatment `A` given `X`, generates potential outcomes `Y0`
        and `Y1` at the configured effect strength, and selects the observed `Y` from the treatment
        received.

        Args:
            n: Sample size, the number of units to generate.
            seed: Random seed for this draw, overriding the scenario's seed.
            split: Which half of the source image pool to draw baselines from, "train" or "test".
                Use "test" for held-out evaluation.

        Returns:
            A `Sample` holding `X`, `A`, `Y`, `propensity`, and the oracle outcomes `Y0` and `Y1`.
        """
        rng = np.random.default_rng(self.seed if seed is None else seed)
        alpha = self.treatment_coefficients * self.confounding_strength
        beta = self.outcome_coefficients * self.confounding_strength

        X = rng.normal(0, 1, size=(n, self.covariate_dimension))
        propensity = np.clip(sigmoid(X @ alpha), *self.propensity_clipping)
        A = rng.binomial(1, propensity)
        Y0, Y1 = self._potential_outcomes(X, beta, split, rng, n)

        select = A.reshape((-1,) + (1,) * (Y0.ndim - 1))
        Y = np.where(select == 1, Y1, Y0)
        return Sample(X=X, Y=Y, A=A, propensity=propensity, Y0=Y0, Y1=Y1)

    def _fitted_perturbation(self, healthy, disease):
        if self._perturbation is None:
            self._perturbation = self.config.perturbation(prior=self.config.prior).fit(healthy, disease)
        return self._perturbation

    def _potential_outcomes(self, X, beta, split, rng, n, q_min=0.10, q_max=0.40, tau=0.20):
        config = self.config
        healthy, disease = load_class_pools(config.source or config.key, *config.classes)
        pool = split_pool(np.arange(len(healthy)), split)

        if n > len(pool):
            raise ValueError(f"n={n} exceeds the {len(pool)} available baselines for split={split!r}")
        
        baselines = healthy[rng.choice(pool, size=n, replace=False)]
        perturbation = self._fitted_perturbation(healthy, disease)

        q = q_min + (q_max - q_min) * sigmoid(X @ beta)
        S = rng.binomial(1, q).astype(float)
        J0 = np.exp(tau * rng.normal(size=n) - 0.5 * tau**2)
        J1 = np.exp(tau * rng.normal(size=n) - 0.5 * tau**2)
        theta = self.effect_strength

        R0 = perturbation.apply(baselines, q * theta * J0, rng)
        R1 = perturbation.apply(baselines, S * theta * J1, rng)

        if self.center_effect:
            delta_bar = (R1 - R0).mean(axis=0)
            R0 = R0 + 0.5 * delta_bar
            R1 = R1 - 0.5 * delta_bar

        return (baselines + R0).astype(np.float32), (baselines + R1).astype(np.float32)
