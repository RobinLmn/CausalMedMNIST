from dataclasses import replace

import numpy as np
import pytest

import causal_medmnist as cm

CONFIGS = [
    cm.OCTMNIST_CNV,
    cm.OCTMNIST_DME,
    cm.OCTMNIST_DRUSEN,
    cm.PNEUMONIA,
    cm.CHEST_EFFUSION,
    cm.CHEST_MASS,
    cm.CHEST_NODULE,
    cm.CHEST_CARDIOMEGALY,
    cm.DERMA,
    cm.BREAST,
    cm.RETINA_MILD,
    cm.RETINA_MODERATE,
    cm.RETINA_SEVERE,
    cm.RETINA_PROLIFERATIVE,
    cm.BLOOD_AUER_ROD,
]
CONFIG_IDS = [
    "cnv",
    "dme",
    "drusen",
    "pneumonia",
    "chest_effusion",
    "chest_mass",
    "chest_nodule",
    "chest_cardiomegaly",
    "derma",
    "breast",
    "retina_mild",
    "retina_moderate",
    "retina_severe",
    "retina_proliferative",
    "blood_auer_rod",
]


@pytest.fixture(params=CONFIGS, ids=CONFIG_IDS, scope="module")
def config(request):
    return request.param


@pytest.fixture(scope="module")
def sample(config):
    return cm.Scenario(config, effect_strength=0.6).generate(n=200, seed=0)


def test_shapes(sample):
    assert sample.X.shape == (200, 6)
    assert sample.A.shape == (200,)
    assert sample.Y.shape[:3] == (200, 28, 28)
    assert sample.propensity.shape == (200,)
    assert sample.Y0.shape == sample.Y.shape
    assert sample.Y1.shape == sample.Y.shape


def test_observed_is_float32(sample):
    assert sample.Y.dtype == np.float32


def test_oracle_present(sample):
    assert sample.Y0 is not None
    assert sample.Y1 is not None


def test_treatment_is_binary(sample):
    assert set(np.unique(sample.A)).issubset({0, 1})


def test_propensity_within_overlap_bounds(sample):
    low, high = 0.07, 0.93
    assert sample.propensity.min() >= low
    assert sample.propensity.max() <= high


def test_observed_matches_potential_outcomes(sample):
    select = sample.A.reshape((-1,) + (1,) * (sample.Y1.ndim - 1))
    expected = np.where(select == 1, sample.Y1, sample.Y0)
    assert np.array_equal(sample.Y, expected)


def test_determinism(config):
    a = cm.Scenario(config).generate(n=50, seed=0)
    b = cm.Scenario(config).generate(n=50, seed=0)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.A, b.A)
    assert np.array_equal(a.Y, b.Y)


def test_different_seed_differs(config):
    a = cm.Scenario(config).generate(n=50, seed=0)
    b = cm.Scenario(config).generate(n=50, seed=1)
    assert not np.array_equal(a.Y, b.Y)


def test_string_and_object_selector_match(config):
    a = cm.Scenario(config).generate(n=50, seed=0)
    b = cm.Scenario(config.key).generate(n=50, seed=0)
    assert np.array_equal(a.Y, b.Y)


def test_effect_grows_with_strength(config):
    weak = cm.Scenario(config, effect_strength=0.0).generate(n=200, seed=0)
    strong = cm.Scenario(config, effect_strength=0.6).generate(n=200, seed=0)
    assert (strong.Y1 - strong.Y0).std() > (weak.Y1 - weak.Y0).std()


def test_effect_is_centered(config):
    sample = cm.Scenario(config, effect_strength=0.6).generate(n=300, seed=0)
    average_effect = (sample.Y1 - sample.Y0).mean(axis=0)
    assert np.abs(average_effect).max() < 1e-6


def test_uncentered_effect_has_nonzero_mean(config):
    sample = cm.Scenario(config, effect_strength=0.6, center_effect=False).generate(n=300, seed=0)
    average_effect = (sample.Y1 - sample.Y0).mean(axis=0)
    assert np.abs(average_effect).max() > 1e-3


def test_randomized_when_no_confounding(config):
    sample = cm.Scenario(config, confounding_strength=0.0).generate(n=200, seed=0)
    assert np.allclose(sample.propensity, 0.5)


def test_covariate_dimension_override():
    config = replace(
        cm.OCTMNIST_DME,
        covariate_dimension=3,
        treatment_coefficients=np.array([0.5, -0.3, 0.2]),
        outcome_coefficients=np.array([0.4, -0.2, 0.1]),
    )
    assert cm.Scenario(config).generate(n=20, seed=0).X.shape == (20, 3)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        cm.Scenario("not_a_dataset")


def test_coefficient_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cm.Scenario(replace(cm.OCTMNIST_DME, covariate_dimension=4))


def test_bad_split_raises():
    with pytest.raises(ValueError):
        cm.Scenario(cm.OCTMNIST_DME).generate(n=5, seed=0, split="validation")


def test_n_too_large_raises():
    with pytest.raises(ValueError):
        cm.Scenario(cm.OCTMNIST_DME).generate(n=10**9, seed=0)


@pytest.mark.parametrize(
    "grade",
    [cm.RETINA_MILD, cm.RETINA_MODERATE, cm.RETINA_SEVERE, cm.RETINA_PROLIFERATIVE],
    ids=["mild", "moderate", "severe", "proliferative"],
)
def test_retina_grades_generate(grade):
    sample = cm.Scenario(grade, effect_strength=0.4).generate(n=100, seed=0, replace=True)
    assert sample.Y.shape == (100, 28, 28, 3)
    assert np.isfinite(sample.Y).all()
    assert np.abs((sample.Y1 - sample.Y0).mean(axis=0)).max() < 1e-6
