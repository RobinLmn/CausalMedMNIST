from .base import Perturbation


class CompositePerturbation(Perturbation):
    """Composite perturbatio as the sum of additive perturbations.

    Args:
        parts: Collection of perturbations to add.
        prior: Location-prior passed to every part.
        noise_sigma: Standard deviation of the additive per-pixel noise.
    """

    def __init__(self, parts, prior=None, noise_sigma=0.025):
        self.parts = [part(prior=prior) for part in parts]
        self.noise_sigma = noise_sigma
        self.mask = None

    def fit(self, healthy, disease, baselines, rng):
        for part in self.parts:
            part.fit(healthy, disease, baselines, rng)
        self.mask = sum(part.mask for part in self.parts)


class SequentialPerturbation(Perturbation):
    """Apply perturbations in sequence, each fitted on the result of the previous one.

    Args:
        parts: Perturbations to be applied in order.
        prior: Location-prior passed to every part.
        noise_sigma: Standard deviation of the additive per-pixel noise.
    """

    def __init__(self, parts, prior=None, noise_sigma=0.025):
        self.parts = [part(prior=prior) for part in parts]
        self.noise_sigma = noise_sigma
        self.mask = None

    def fit(self, healthy, disease, baselines, rng):
        current = baselines
        for part in self.parts:
            part.fit(healthy, disease, current, rng)
            current = current + part.mask
        self.mask = current - baselines
