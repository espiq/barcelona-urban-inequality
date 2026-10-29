"""Unit tests for the statistical building blocks, using cases with known answers."""

import numpy as np
import pandas as pd
import pytest

from bcn.inequality import (
    beta_convergence,
    concentration_index,
    theil_decomposition,
    theil_t,
    weighted_gini,
    weighted_quantile,
)
from bcn.spatial import fdr_threshold


def test_gini_perfect_equality_is_zero():
    assert weighted_gini([5, 5, 5, 5]) == pytest.approx(0.0)


def test_gini_matches_closed_form():
    # For n equal-weight units, one holding everything: G = (n - 1) / n.
    assert weighted_gini([0, 0, 0, 1]) == pytest.approx(0.75)


def test_gini_weights_equal_replication():
    # A weight of 3 must be identical to repeating the observation 3 times.
    assert weighted_gini([1, 4], [3, 1]) == pytest.approx(weighted_gini([1, 1, 1, 4]))


def test_gini_scale_invariant():
    x = np.array([10.0, 20, 35, 80])
    assert weighted_gini(x) == pytest.approx(weighted_gini(x * 7.3))


def test_weighted_quantile_median():
    assert weighted_quantile([1, 2, 3], [1, 1, 1], 0.5) == pytest.approx(2.0)


def test_concentration_index_signs_and_bounds():
    income = [10, 20, 30, 40]
    assert concentration_index([5, 5, 5, 5], income) == pytest.approx(0.0)
    assert concentration_index([4, 3, 2, 1], income) < 0  # concentrated among the poor
    assert concentration_index([1, 2, 3, 4], income) > 0
    # Everything held by the richest unit: CI = 1 - 1/n for equal weights.
    assert concentration_index([0, 0, 0, 1], income) == pytest.approx(0.75)


def test_concentration_index_equals_gini_when_ranked_by_itself():
    x = np.array([3.0, 8, 1, 12, 5])
    w = np.array([2.0, 1, 4, 1, 3])
    assert concentration_index(x, x, w) == pytest.approx(weighted_gini(x, w))


def test_theil_zero_under_equality():
    assert theil_t([3, 3, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_theil_decomposition_is_exact():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.lognormal(10, 0.4, 200),
            "w": rng.integers(100, 2000, 200),
            "g": rng.integers(0, 8, 200),
        }
    )
    d = theil_decomposition(df, "y", "w", "g")
    assert d["between"] + d["within"] == pytest.approx(d["total"], rel=1e-10)
    assert 0 <= d["between_share"] <= 1


def test_beta_convergence_recovers_known_beta():
    rng = np.random.default_rng(1)
    y0 = pd.Series(rng.lognormal(10, 0.3, 2000))
    true_beta, years = -0.02, 8
    growth = 0.3 + true_beta * np.log(y0) + rng.normal(0, 0.002, len(y0))
    y1 = y0 * np.exp(growth * years)
    res = beta_convergence(y0, y1, years)
    assert res.beta == pytest.approx(true_beta, abs=1e-3)
    assert res.summary_row("x")["verdict"] == "convergence"
    assert res.half_life_years > 0


def test_beta_divergence_has_no_half_life():
    y0 = pd.Series(np.linspace(1e4, 5e4, 50))
    y1 = y0 * (y0 / y0.mean()) ** 0.2 * 1.3
    res = beta_convergence(y0, y1, 5)
    assert res.beta > 0
    assert res.summary_row("x")["verdict"] == "divergence"
    assert np.isnan(res.half_life_years)


def test_fdr_threshold_benjamini_hochberg():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # Classic textbook example: BH at 5% rejects the first two hypotheses.
    assert fdr_threshold(p, 0.05) == pytest.approx(0.008)
    assert fdr_threshold([0.5, 0.9], 0.05) == 0.0
