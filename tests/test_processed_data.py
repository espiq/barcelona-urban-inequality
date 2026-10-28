"""Data-contract tests on the processed tables committed to the repository."""

import pandas as pd
import pytest

from bcn.config import DATA_PROCESSED, FIRST_YEAR, LAST_YEAR

pytestmark = pytest.mark.skipif(
    not (DATA_PROCESSED / "barri_panel.csv").exists(),
    reason="processed data not built; run `python -m bcn`",
)


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(DATA_PROCESSED / "barri_panel.csv")


@pytest.fixture(scope="module")
def sections():
    return pd.read_csv(DATA_PROCESSED / "section_panel.csv")


def test_panel_is_balanced(panel):
    counts = panel.groupby("year").barri_code.nunique()
    assert list(counts.index) == list(range(FIRST_YEAR, LAST_YEAR + 1))
    assert (counts == 73).all()
    assert not panel.duplicated(["barri_code", "year"]).any()


def test_sections_unique_and_complete(sections):
    assert not sections.duplicated(["section_id", "year"]).any()
    assert sections.groupby("year").size().eq(1068).all()


def test_barri_population_sums_sections(panel, sections):
    a = panel.groupby("year").population.sum()
    b = sections.groupby("year").population.sum()
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_barri_income_within_section_range(panel, sections):
    rng = sections.groupby(["barri_code", "year"]).income_pc.agg(["min", "max"])
    m = panel.set_index(["barri_code", "year"]).join(rng)
    assert (m.income_pc >= m["min"] - 1e-6).all()
    assert (m.income_pc <= m["max"] + 1e-6).all()


def test_shares_are_valid(panel):
    for col in ("share_foreign", "share_non_eu"):
        assert panel[col].between(0, 1).all()
