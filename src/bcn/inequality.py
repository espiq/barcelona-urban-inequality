"""Inequality and convergence metrics for grouped (per-capita) income data.

All functions take *group means* with *population weights*, i.e. they measure
inequality between areas, treating every resident of an area as earning the
area mean. This is the standard approach for small-area income data, and it
gives a lower bound on interpersonal inequality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def weighted_gini(values, weights=None) -> float:
    """Gini coefficient of ``values`` with frequency ``weights`` (0 = equality)."""
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cum_w = np.cumsum(w)
    cum_xw = np.cumsum(x * w)
    # Area under the Lorenz curve via the trapezoid rule.
    lorenz = np.concatenate([[0.0], cum_xw / cum_xw[-1]])
    pop = np.concatenate([[0.0], cum_w / cum_w[-1]])
    return float(1 - np.sum((pop[1:] - pop[:-1]) * (lorenz[1:] + lorenz[:-1])))


def weighted_quantile(values, weights, q: float) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cdf, x))


def concentration_index(outcome, rank_by, weights=None) -> float:
    """Concentration index of ``outcome`` across units ranked by ``rank_by``.

    CI = 2 * cov_w(h, R) / mean_w(h), where R is the weighted fractional rank
    of each unit in the ``rank_by`` distribution (Kakwani, Wagstaff & van
    Doorslaer, 1997). CI lies in [-1, 1]: negative values mean the outcome is
    concentrated among low-``rank_by`` (e.g. poorer) units, 0 means no
    income-related inequality.
    """
    h = np.asarray(outcome, dtype=float)
    x = np.asarray(rank_by, dtype=float)
    w = np.ones_like(h) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x, kind="stable")
    h, w = h[order], w[order]
    share = w / w.sum()
    rank = np.cumsum(share) - 0.5 * share  # fractional rank at each unit's midpoint
    mu = np.sum(share * h)
    cov = np.sum(share * (h - mu) * (rank - np.sum(share * rank)))
    return float(2 * cov / mu)


def theil_t(values, weights) -> float:
    """Theil T index (GE(1)) for group means ``values`` with population ``weights``."""
    y = np.asarray(values, dtype=float)
    p = np.asarray(weights, dtype=float)
    mu = np.average(y, weights=p)
    s = p * y / np.sum(p * y)  # income shares
    return float(np.sum(s * np.log(y / mu)))


def theil_decomposition(df: pd.DataFrame, value: str, weight: str, group: str) -> dict:
    """Exact decomposition T = T_between + T_within for a grouping.

    T_within = sum_g S_g * T_g, where S_g is group g's income share.
    T_between is the Theil index of the group means.
    """
    total = theil_t(df[value], df[weight])
    inc = df[value] * df[weight]
    grp = df.assign(_inc=inc).groupby(group)
    shares = grp["_inc"].sum() / inc.sum()
    t_g = grp.apply(lambda d: theil_t(d[value], d[weight]), include_groups=False)
    within = float((shares * t_g).sum())
    means = grp["_inc"].sum() / grp[weight].sum()
    between = theil_t(means.values, grp[weight].sum().values)
    return {
        "total": total,
        "between": between,
        "within": within,
        "between_share": between / total,
    }


def inequality_by_year(sections: pd.DataFrame) -> pd.DataFrame:
    """City-wide spatial inequality indicators per year, from census sections.

    The Theil index is decomposed hierarchically:
    districts (10) -> barris within districts (73) -> sections within barris (1,068).
    """
    rows = []
    for year, d in sections.groupby("year"):
        y, w = d["income_pc"], d["population"]
        t_district = theil_decomposition(d, "income_pc", "population", "district_code")
        t_barri = theil_decomposition(d, "income_pc", "population", "barri_code")
        rows.append(
            {
                "year": year,
                "mean_income": np.average(y, weights=w),
                "gini_sections": weighted_gini(y, w),
                "p90_p10": weighted_quantile(y, w, 0.9) / weighted_quantile(y, w, 0.1),
                "cv": np.sqrt(np.cov(y, aweights=w)) / np.average(y, weights=w),
                "sd_log_income": np.sqrt(np.cov(np.log(y), aweights=w)),
                "theil_total": t_district["total"],
                # Three-level nested decomposition (shares sum to 1).
                "theil_between_districts": t_district["between"],
                "theil_between_barris_within_districts": t_barri["between"] - t_district["between"],
                "theil_within_barris": t_barri["within"],
            }
        )
    return pd.DataFrame(rows)


@dataclass
class BetaConvergence:
    beta: float
    se: float
    pvalue: float
    r2: float
    n: int
    implied_speed: float  # annual convergence rate lambda
    half_life_years: float
    model: object

    def summary_row(self, label: str) -> dict:
        if self.pvalue >= 0.05:
            verdict = "no significant relation"
        else:
            verdict = "convergence" if self.beta < 0 else "divergence"
        return {
            "specification": label,
            "verdict": verdict,
            "beta": self.beta,
            "se_robust": self.se,
            "p_value": self.pvalue,
            "r2": self.r2,
            "n": self.n,
            "speed_pct_per_year": 100 * self.implied_speed,
            "half_life_years": self.half_life_years,
        }


def beta_convergence(
    initial: pd.Series,
    final: pd.Series,
    years: int,
    controls: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    cluster: pd.Series | None = None,
) -> BetaConvergence:
    """Absolute/conditional beta-convergence (Barro & Sala-i-Martin, 1992).

    (1/T) * ln(y_T / y_0) = a + beta * ln(y_0) + controls + e

    beta < 0 indicates that poorer areas grow faster. The implied speed of
    convergence is lambda = -ln(1 + beta*T) / T and the half-life ln(2)/lambda.
    """
    growth = np.log(final / initial) / years
    X = pd.DataFrame({"log_initial": np.log(initial)})
    if controls is not None:
        X = X.join(controls)
    X = sm.add_constant(X, has_constant="add")
    model = sm.WLS(growth, X, weights=weights) if weights is not None else sm.OLS(growth, X)
    if cluster is not None:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": cluster})
    else:
        fit = model.fit(cov_type="HC3")
    b = fit.params["log_initial"]
    # Speed and half-life are only defined under convergence (beta < 0).
    speed = -np.log(1 + b * years) / years if -1 < b * years < 0 else np.nan
    return BetaConvergence(
        beta=float(b),
        se=float(fit.bse["log_initial"]),
        pvalue=float(fit.pvalues["log_initial"]),
        r2=float(fit.rsquared),
        n=int(fit.nobs),
        implied_speed=float(speed),
        half_life_years=float(np.log(2) / speed) if speed > 0 else np.nan,
        model=fit,
    )


def growth_level_diagnostics(sections: pd.DataFrame) -> pd.DataFrame:
    """Year-on-year audit of the section income series.

    For a stable statistical method, annual growth should be roughly
    uncorrelated with the previous level and rankings should be persistent.
    A large |corr(growth, level)| in a single year flags a possible
    methodological break (or a genuine, highly redistributive shock).
    """
    w = sections.pivot(index="section_id", columns="year", values="income_pc")
    rows = []
    for prev, year in zip(w.columns[:-1], w.columns[1:]):
        g = np.log(w[year] / w[prev])
        rows.append(
            {
                "year": int(year),
                "mean_growth_pct": 100 * (np.exp(g.mean()) - 1),
                "sd_growth_pp": 100 * g.std(),
                "corr_growth_vs_prev_level": g.corr(np.log(w[prev])),
                "rank_persistence_spearman": w[year].corr(w[prev], method="spearman"),
            }
        )
    return pd.DataFrame(rows)
