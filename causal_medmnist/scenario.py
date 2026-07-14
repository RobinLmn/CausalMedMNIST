from dataclasses import dataclass

import numpy as np

from .datasets import get_config
from .utils import sigmoid


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

    Wraps a MedMNIST dataset in model where confounders `X` cause both the treatment `A` and the outcome `Y`. Draws samples and returns ground-truth
    potential outcomes with the `generate` method.

    Args:
        dataset: A dataset configuration or its key string.
        effect_strength: Strength of the treatment effect. `0.0` represents the null hypothesis with no treatment effect.
        confounding_strength: Scales the treatment and outcome coefficients. `0.0` is a randomized trial with no confounding.
        propensity_clipping: (low, high) clipping bounds on the propensity to enforce overlap.
        distributional_effect: If True, the treatment effect is built to be distributional (mean-matched). Otherwise, the treatment effect lies in the mean.
        center_effect: If True, the potential outcomes are re-centered to ensure a mean effect of zero. Only used for distributional effects.
        seed: Default random seed.

    Examples:
        >>> scenario = Scenario(OCTMNIST_DME, effect_strength=0.6)
        >>> sample = scenario.generate(n=1000, seed=0)
    """

    def __init__(
        self,
        dataset,
        *,
        effect_strength=0.6,
        confounding_strength=1.0,
        propensity_clipping=(0.07, 0.93),
        distributional_effect=True,
        center_effect=True,
        seed=None,
    ):
        self.config = get_config(dataset)
        self.effect_strength = effect_strength
        self.confounding_strength = confounding_strength
        self.propensity_clipping = propensity_clipping
        self.distributional_effect = distributional_effect
        self.center_effect = center_effect
        self.seed = seed
        self.perturbation = self.config.perturbation(prior=self.config.prior)

        if len(self.config.treatment_coefficients) != self.config.covariate_dimension:
            raise ValueError(f"treatment_coefficients has length {len(self.config.treatment_coefficients)}, expected: {self.config.covariate_dimension}")

        if len(self.config.outcome_coefficients) != self.config.covariate_dimension:
            raise ValueError(f"outcome_coefficients has length {len(self.config.outcome_coefficients)}, expected: {self.config.covariate_dimension}")

    def generate(self, n, *, seed=None, split="train", replace=False):
        """Sample a dataset from this scenario's causal model.

        Draws confounders `X`, assigns treatment `A` given `X`, generates potential outcomes `Y0`
        and `Y1` at the configured effect strength, and selects the observed `Y` from the treatment
        received.

        Args:
            n: Sample size, the number of units to generate.
            seed: Random seed for this draw, overriding the scenario's seed.
            split: Which MedMNIST split to draw baselines from, "train", "val", or "test".
            replace: If True, sample baselines with replacement so `n` may exceed the split size.

        Returns:
            A `Sample` holding `X`, `A`, `Y`, `propensity`, and the oracle outcomes `Y0` and `Y1`.
        """
        rng = np.random.default_rng(self.seed if seed is None else seed)
        alpha = self.config.treatment_coefficients * self.confounding_strength
        beta = self.config.outcome_coefficients * self.confounding_strength

        X = rng.normal(0, 1, size=(n, self.config.covariate_dimension))
        propensity = np.clip(sigmoid(X @ alpha), *self.propensity_clipping)
        A = rng.binomial(1, propensity)
        Y0, Y1 = self._potential_outcomes(X, beta, split, replace, rng, n)

        select = A.reshape((-1,) + (1,) * (Y0.ndim - 1))
        Y = np.where(select == 1, Y1, Y0)
        return Sample(X=X, Y=Y, A=A, propensity=propensity, Y0=Y0, Y1=Y1)

    def _potential_outcomes(self, X, beta, split, replace, rng, n, q_min=0.10, q_max=0.40, tau=0.20):
        healthy, _ = self.config.loader(split)
        if not replace and n > len(healthy):
            raise ValueError(f"N={n} exceeds the number of available images for split={split!r} (available: {len(healthy)}). Set replace=True or reduce N.")

        baselines = healthy[rng.choice(len(healthy), size=n, replace=replace)]

        healthy_tr, disease_tr = self.config.loader("train")
        self.perturbation.fit(healthy_tr, disease_tr, baselines, rng)

        J1 = np.exp(tau * rng.normal(size=n) - 0.5 * tau**2)
        theta = self.effect_strength

        if not self.distributional_effect:
            R0 = self.perturbation.apply(np.zeros(n), rng)
            R1 = self.perturbation.apply(theta * J1, rng)
        else:
            q = q_min + (q_max - q_min) * sigmoid(X @ beta)
            S = rng.binomial(1, q).astype(float)
            J0 = np.exp(tau * rng.normal(size=n) - 0.5 * tau**2)

            R0 = self.perturbation.apply(q * theta * J0, rng)
            R1 = self.perturbation.apply(S * theta * J1, rng)

            if self.center_effect:
                delta_bar = (R1 - R0).mean(axis=0)
                R0 = R0 + 0.5 * delta_bar
                R1 = R1 - 0.5 * delta_bar

        return (baselines + R0).astype(np.float32), (baselines + R1).astype(np.float32)
