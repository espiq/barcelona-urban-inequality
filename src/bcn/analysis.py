"""End-to-end analysis: reads ``data/processed`` and writes every figure,
table and headline number used in the report.

Run with ``python -m bcn.analysis`` (after ``bcn.download`` and ``bcn.clean``).
"""

from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler

from . import viz
from .config import DATA_PROCESSED, FIRST_YEAR, LAST_YEAR, ROOT, TABLES
from .inequality import (
    beta_convergence,
    growth_level_diagnostics,
    inequality_by_year,
    weighted_gini,
)
from .models import (
    lm_diagnostics,
    ols_robust,
    select_k,
    spatial_coef_table,
    spatial_fit_stats,
    spatial_regressions,
    typology,
    vif_table,
    zscore,
)
from .reporting import save_results, save_table
from .spatial import global_moran, local_moran, queen_weights

log = logging.getLogger(__name__)
T = LAST_YEAR - FIRST_YEAR
RESULTS: dict = {}


def _table(df: pd.DataFrame, name: str, floatfmt: str = ".3f") -> None:
    save_table(df, name, TABLES, floatfmt)


def load():
    barris = gpd.read_file(DATA_PROCESSED / "barris.geojson")
    secs = gpd.read_file(DATA_PROCESSED / "sections.geojson")
    sections = pd.read_csv(DATA_PROCESSED / "section_panel.csv")
    panel = pd.read_csv(DATA_PROCESSED / "barri_panel.csv")
    cross = pd.read_csv(DATA_PROCESSED / "barri_cross_section.csv")
    huts = gpd.read_file(DATA_PROCESSED / "tourist_flats.geojson")
    return barris, secs, sections, panel, cross, huts


# =========================================================================== 1. levels
def fig_income_maps(barris, secs, sections, cross):
    last = sections[sections.year == LAST_YEAR].set_index("section_id")
    s = secs.set_index("section_id").join(last[["income_pc"]])
    b = barris.merge(cross[["barri_code", "income_pc"]], on="barri_code")

    vmin, vmax = s.income_pc.quantile([0.02, 0.98])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, gdf, title in [
        (axes[0], b, "73 neighbourhoods (barris)"),
        (axes[1], s, "1,068 census sections"),
    ]:
        gdf.plot(
            "income_pc",
            ax=ax,
            cmap=viz.SEQUENTIAL,
            vmin=vmin,
            vmax=vmax,
            edgecolor=viz.SURFACE,
            linewidth=0.3 if len(gdf) > 100 else 0.8,
        )
        barris.boundary.plot(ax=ax, color=viz.TEXT_2, linewidth=0.25)
        ax.set_title(title, fontsize=10, color=viz.TEXT_2, fontweight="normal")
        viz.map_axes(ax)
    sm = plt.cm.ScalarMappable(cmap=viz.SEQUENTIAL, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.02, aspect=50)
    cb.set_label(f"Household disposable income per person, {LAST_YEAR} (EUR)")
    cb.outline.set_visible(False)
    fig.suptitle(
        f"Where Barcelona's income lives ({LAST_YEAR})",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    viz.save(fig, "01_income_maps")

    top = cross.nlargest(5, "income_pc")[["barri", "district", "income_pc"]]
    bot = cross.nsmallest(5, "income_pc")[["barri", "district", "income_pc"]]
    RESULTS["richest_barri"] = top.iloc[0].to_dict()
    RESULTS["poorest_barri"] = bot.iloc[0].to_dict()
    RESULTS["richest_poorest_ratio"] = float(top.iloc[0].income_pc / bot.iloc[0].income_pc)
    _table(
        pd.concat([top.assign(group="Top 5"), bot.assign(group="Bottom 5")]),
        "top_bottom_barris",
        ".0f",
    )


def fig_income_trajectories(panel):
    first = panel[panel.year == FIRST_YEAR].set_index("barri_code")
    q = pd.qcut(first.income_pc, 5, labels=[f"Q{i}" for i in range(1, 6)])
    p = panel.assign(quintile=panel.barri_code.map(q))
    p["inc_w"] = p.income_pc * p.population
    g = p.groupby(["quintile", "year"], observed=True)
    agg = (g.inc_w.sum() / g.population.sum()).rename("income").reset_index()
    agg["index"] = agg.groupby("quintile", observed=True).income.transform(
        lambda s: 100 * s / s.iloc[0]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    shades = [viz.BLUE_RAMP[i] for i in (1, 2, 3, 4, 6)]
    ends = {0: {}, 1: {}}
    for (name, d), c in zip(agg.groupby("quintile", observed=True), shades):
        for i, col in enumerate(("income", "index")):
            axes[i].plot(d.year, d[col], color=c, marker="o", ms=4)
            ends[i][name] = (d.year.iloc[-1], d[col].iloc[-1])
    axes[0].set_title("Mean disposable income per person (EUR, nominal)")
    axes[1].set_title(f"Indexed, {FIRST_YEAR} = 100")
    axes[1].axhline(100, color=viz.MUTED, lw=0.8)
    for i in (0, 1):
        viz.label_line_ends(axes[i], ends[i])
    for ax in axes:
        ax.axvspan(2019.5, 2020.5, color=viz.NEUTRAL, alpha=0.6, lw=0)
        ax.set_xticks(range(FIRST_YEAR, LAST_YEAR + 1))
    axes[0].text(
        2020, axes[0].get_ylim()[1], "COVID-19", ha="center", va="top", fontsize=8, color=viz.TEXT_2
    )
    fig.suptitle(
        f"Income by neighbourhood quintile (barris ranked by {FIRST_YEAR} income)",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    viz.caption(
        fig,
        "Q1 = poorest fifth of barris in 2015, Q5 = richest. Population-weighted "
        "means. Values are nominal; the shaded band marks 2020.",
    )
    viz.save(fig, "02_income_trajectories")
    last = agg[agg.year == LAST_YEAR].set_index("quintile")["index"]
    RESULTS["growth_q1_pct"] = float(last["Q1"] - 100)
    RESULTS["growth_q5_pct"] = float(last["Q5"] - 100)


# =========================================================================== 2. inequality
def fig_inequality(sections):
    ineq = inequality_by_year(sections)
    _table(ineq, "inequality_by_year", ".4f")
    RESULTS["gini_first"] = float(ineq.gini_sections.iloc[0])
    RESULTS["gini_last"] = float(ineq.gini_sections.iloc[-1])
    RESULTS["p90p10_first"] = float(ineq.p90_p10.iloc[0])
    RESULTS["p90p10_last"] = float(ineq.p90_p10.iloc[-1])
    shares = ineq[
        ["theil_between_districts", "theil_between_barris_within_districts", "theil_within_barris"]
    ].div(ineq.theil_total, axis=0)
    RESULTS["theil_shares_last"] = shares.iloc[-1].round(4).to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(ineq.year, ineq.gini_sections, marker="o", color=viz.CATEGORICAL[0])
    axes[0].set_title("Gini between census sections")
    axes[1].plot(ineq.year, ineq.p90_p10, marker="o", color=viz.CATEGORICAL[0])
    axes[1].set_title("P90 / P10 ratio")
    labels = ["Between districts", "Between barris, within district", "Within barris"]
    colors = [viz.BLUE_RAMP[5], viz.BLUE_RAMP[3], viz.BLUE_RAMP[1]]
    axes[2].stackplot(
        ineq.year,
        shares.T.values,
        colors=colors,
        labels=labels,
        edgecolor=viz.SURFACE,
        linewidth=1.5,
    )
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Theil index decomposition (share)")
    axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, fontsize=8)
    for ax in axes:
        ax.set_xticks(range(FIRST_YEAR, LAST_YEAR + 1, 2))
    for ax in axes[:2]:
        ax.axvspan(2019.5, 2020.5, color=viz.NEUTRAL, alpha=0.6, lw=0)
    fig.suptitle(
        "Spatial income inequality over time", x=0.02, ha="left", fontsize=14, fontweight="semibold"
    )
    viz.caption(
        fig,
        "Computed on 1,068 census sections, weighted by residents. The Theil "
        "decomposition is exact and nested: districts > barris > sections.",
    )
    viz.save(fig, "03_inequality_over_time")
    return ineq


def series_audit(sections):
    diag = growth_level_diagnostics(sections)
    _table(diag, "series_audit", ".3f")
    RESULTS["series_audit"] = diag.round(3).to_dict(orient="records")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    colors = [
        viz.CATEGORICAL[1] if abs(v) > 0.4 else viz.CATEGORICAL[0]
        for v in diag.corr_growth_vs_prev_level
    ]
    bars = axes[0].bar(
        diag.year,
        diag.corr_growth_vs_prev_level,
        color=colors,
        width=0.7,
        edgecolor=viz.SURFACE,
        linewidth=2,
    )
    axes[0].bar_label(bars, fmt="%+.2f", fontsize=8, color=viz.TEXT_2, padding=2)
    axes[0].axhline(0, color=viz.TEXT_2, lw=0.8)
    for y in (-0.4, 0.4):
        axes[0].axhline(y, color=viz.MUTED, lw=0.8, ls="--")
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("corr(annual growth, previous level), 1,068 sections")
    axes[0].grid(axis="x", visible=False)
    bars = axes[1].bar(
        diag.year,
        diag.mean_growth_pct,
        color=viz.CATEGORICAL[0],
        width=0.7,
        edgecolor=viz.SURFACE,
        linewidth=2,
    )
    axes[1].bar_label(bars, fmt="%+.1f", fontsize=8, color=viz.TEXT_2, padding=2)
    axes[1].axhline(0, color=viz.TEXT_2, lw=0.8)
    axes[1].set_title("Mean annual income growth (%)")
    axes[1].grid(axis="x", visible=False)
    for ax in axes:
        ax.set_xticks(diag.year)
    fig.suptitle(
        "Data audit: are year-on-year changes plausible?",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    viz.caption(
        fig,
        "Orange bars (|r| > 0.4) flag years where growth is strongly tied to the "
        "starting level - consistent with a revision of the estimation method or "
        "a strongly redistributive shock. Ranks remain very persistent "
        "(Spearman > 0.98 every year).",
    )
    viz.save(fig, "11_series_audit")


def convergence(sections, panel, cross):
    s0 = sections[sections.year == FIRST_YEAR].set_index("section_id")
    s1 = sections[sections.year == LAST_YEAR].set_index("section_id")
    common = s0.index.intersection(s1.index)
    s0, s1 = s0.loc[common], s1.loc[common]
    controls = pd.DataFrame({"share_non_eu_0": s0.pop_non_eu / s0.population}, index=common)
    dist_fe = pd.get_dummies(s0.district_code, prefix="d", drop_first=True, dtype=float)

    specs = {
        "Barris, absolute (n=73)": beta_convergence(
            cross.set_index("barri_code").income_pc_first,
            cross.set_index("barri_code").income_pc,
            T,
        ),
        "Sections, absolute (n=1,068)": beta_convergence(
            s0.income_pc, s1.income_pc, T, cluster=s0.barri_code
        ),
        "Sections, + non-EU share + district FE": beta_convergence(
            s0.income_pc, s1.income_pc, T, controls=controls.join(dist_fe), cluster=s0.barri_code
        ),
    }
    tab = pd.DataFrame([m.summary_row(k) for k, m in specs.items()])
    _table(tab, "beta_convergence", ".4f")
    RESULTS["beta_convergence"] = tab.to_dict(orient="records")

    # Robustness: the conclusion must not hinge on one (possibly revised) year.
    wide = sections.pivot(index="section_id", columns="year", values="income_pc")
    clusters = sections.drop_duplicates("section_id").set_index("section_id").barri_code
    rows = []
    for a, b_ in [
        (2015, 2019),
        (2015, 2020),
        (2015, 2022),
        (2015, 2023),
        (2019, 2023),
        (2021, 2023),
    ]:
        m = beta_convergence(wide[a], wide[b_], b_ - a, cluster=clusters.loc[wide.index])
        rows.append({"window": f"{a}-{b_}", **m.summary_row("Sections, absolute")})
    win = pd.DataFrame(rows).drop(columns=["specification"])
    _table(win, "beta_convergence_by_window", ".4f")
    RESULTS["beta_by_window"] = win[["window", "verdict", "beta", "p_value"]].to_dict(
        orient="records"
    )

    fig, ax = plt.subplots(figsize=(8.5, 6))
    x_s, g_s = np.log(s0.income_pc), np.log(s1.income_pc / s0.income_pc) / T
    ax.scatter(x_s, 100 * g_s, s=9, color=viz.MUTED, alpha=0.45, lw=0, label="Census section")
    xb = np.log(cross.income_pc_first)
    gb = np.log(cross.income_pc / cross.income_pc_first) / T
    ax.scatter(
        xb,
        100 * gb,
        s=46,
        color=viz.CATEGORICAL[0],
        edgecolor=viz.SURFACE,
        lw=1.2,
        label="Barri",
        zorder=3,
    )
    for _, m, c in [
        ("Sections, absolute (n=1,068)", specs["Sections, absolute (n=1,068)"], viz.TEXT_2),
        ("Barris, absolute (n=73)", specs["Barris, absolute (n=73)"], viz.CATEGORICAL[0]),
    ]:
        xx = np.linspace(x_s.min(), x_s.max(), 50)
        ax.plot(xx, 100 * (m.model.params["const"] + m.beta * xx), color=c, lw=2, ls="--")
    lab = cross.assign(g=gb, x=xb)
    picks = pd.concat([lab.nlargest(1, "g"), lab.nlargest(1, "x"), lab.nsmallest(1, "x")])
    for _, r in picks.drop_duplicates("barri_code").iterrows():
        right = r.x > xb.median()
        ax.annotate(
            r.barri,
            (r.x, 100 * r.g),
            xytext=(-8 if right else 8, 8),
            textcoords="offset points",
            ha="right" if right else "left",
            fontsize=8,
            color=viz.TEXT_2,
            arrowprops={"arrowstyle": "-", "color": viz.MUTED, "lw": 0.6},
        )
    ax.set_xlabel(f"log income per person, {FIRST_YEAR}")
    ax.set_ylabel(f"Annual income growth {FIRST_YEAR}-{LAST_YEAR} (%)")
    b = specs["Barris, absolute (n=73)"]
    verdict = b.summary_row("")["verdict"]
    ax.set_title(
        f"Beta-convergence test across barris: beta = {b.beta:+.4f} "
        f"(p = {b.pvalue:.3f}) -> {verdict}"
    )
    ax.legend(loc="upper right")
    viz.caption(
        fig,
        "Negative slope = poorer areas grow faster (catch-up); positive slope = "
        "divergence. Dashed lines: OLS fits. Robust (HC3) / barri-clustered SEs.",
    )
    viz.save(fig, "04_beta_convergence")
    return specs


def covid_shock(sections):
    w = sections.pivot(index="section_id", columns="year", values="income_pc")
    pop = sections[sections.year == 2019].set_index("section_id").population
    d = pd.DataFrame(
        {
            "decile": pd.qcut(w[2019], 10, labels=range(1, 11)),
            "shock": 100 * (w[2020] / w[2019] - 1),
            "recovery": 100 * (w[LAST_YEAR] / w[2020] - 1),
            "pop": pop,
        }
    )
    agg = (
        d.groupby("decile", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "shock": np.average(g.shock, weights=g["pop"]),
                    "recovery": np.average(g.recovery, weights=g["pop"]),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    _table(agg, "covid_shock_by_decile", ".2f")
    RESULTS["covid_shock_d1"] = float(agg.shock.iloc[0])
    RESULTS["covid_shock_d10"] = float(agg.shock.iloc[-1])
    RESULTS["corr_shock_recovery"] = float(d[["shock", "recovery"]].corr().iloc[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
    for ax, col, title, c in [
        (axes[0], "shock", "Shock: income change 2019 -> 2020 (%)", viz.CATEGORICAL[1]),
        (
            axes[1],
            "recovery",
            f"Recovery: income change 2020 -> {LAST_YEAR} (%)",
            viz.CATEGORICAL[0],
        ),
    ]:
        bars = ax.bar(
            agg.decile.astype(int),
            agg[col],
            color=c,
            width=0.78,
            edgecolor=viz.SURFACE,
            linewidth=2,
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, color=viz.TEXT_2, padding=2)
        ax.axhline(0, color=viz.TEXT_2, lw=0.8)
        ax.set_title(title)
        ax.set_xticks(range(1, 11))
        ax.set_xlabel("Census-section decile of 2019 income (1 = poorest)")
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Who absorbed the COVID-19 income shock?",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    viz.caption(fig, "Population-weighted mean change per decile of census sections.")
    viz.save(fig, "05_covid_shock")


# =========================================================================== 3. spatial
def spatial_analysis(barris, secs, sections, panel):
    secs = secs.sort_values("section_id").reset_index(drop=True)
    w_s = queen_weights(secs, "section_id")
    b = barris.sort_values("barri_code").reset_index(drop=True)
    w_b = queen_weights(b, "barri_code")

    rows = []
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        ys = sections[sections.year == year].set_index("section_id").loc[secs.section_id]
        yb = panel[panel.year == year].set_index("barri_code").loc[b.barri_code]
        rows.append({"year": year, "level": "sections", **global_moran(np.log(ys.income_pc), w_s)})
        rows.append({"year": year, "level": "barris", **global_moran(np.log(yb.income_pc), w_b)})
    moran = pd.DataFrame(rows)
    _table(moran, "global_moran", ".4f")
    RESULTS["moran_sections_last"] = float(
        moran.query("level == 'sections' and year == @LAST_YEAR").moran_i.iloc[0]
    )
    RESULTS["moran_barris_last"] = float(
        moran.query("level == 'barris' and year == @LAST_YEAR").moran_i.iloc[0]
    )

    s1 = sections[sections.year == LAST_YEAR].set_index("section_id").loc[secs.section_id]
    s0 = sections[sections.year == FIRST_YEAR].set_index("section_id").loc[secs.section_id]
    lisa_lvl = local_moran(np.log(s1.income_pc), w_s)
    lisa_grw = local_moran(np.log(s1.income_pc / s0.income_pc), w_s)
    RESULTS["lisa_level_counts"] = lisa_lvl.cluster.value_counts().to_dict()
    RESULTS["lisa_growth_counts"] = lisa_grw.cluster.value_counts().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4))
    for ax, lisa, title in [
        (axes[0], lisa_lvl, f"Income level, {LAST_YEAR}"),
        (axes[1], lisa_grw, f"Income growth, {FIRST_YEAR}-{LAST_YEAR}"),
    ]:
        g = secs.assign(cluster=lisa.cluster.values)
        g.plot(color=g.cluster.map(viz.LISA_COLORS), ax=ax, edgecolor=viz.SURFACE, lw=0.2)
        barris.boundary.plot(ax=ax, color=viz.TEXT_2, lw=0.3)
        ax.set_title(title, fontsize=11)
        viz.map_axes(ax)
    handles = [Patch(color=c, label=k) for k, c in viz.LISA_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        "Local spatial clusters (LISA) of income across census sections",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    viz.caption(
        fig,
        "Local Moran's I on log values, Queen contiguity, 9,999 permutations, "
        "significance at 5% after Benjamini-Hochberg FDR correction. "
        "High-High = rich area surrounded by rich areas, etc.",
    )
    viz.save(fig, "06_lisa_clusters")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for (lvl, d), c in zip(moran.groupby("level"), viz.CATEGORICAL):
        ax.plot(d.year, d.moran_i, marker="o", color=c)
        ax.annotate(
            lvl.capitalize(),
            (d.year.iloc[-1], d.moran_i.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=viz.TEXT_2,
        )
    ax.set_ylim(0, 1)
    ax.set_xticks(range(FIRST_YEAR, LAST_YEAR + 1))
    ax.set_title("Global Moran's I of log income (all p < 0.001)")
    ax.set_ylabel("Moran's I")
    viz.save(fig, "07_global_moran")
    return w_b


# =========================================================================== 4. correlates
FEATURE_LABELS = {
    f"share_tertiary_{LAST_YEAR}": "Tertiary education (25+)",
    "share_non_eu": "Non-EU nationals",
    "log_cadastral_m2": "Cadastral value per m2 (log)",
    "log_tourist_intensity": "Tourist-flat places per 1,000 res. (log1p)",
    "log_density": "Population density (log)",
}


def correlates(barris, cross, w_b):
    df = cross.sort_values("barri_code").reset_index(drop=True).copy()
    df["log_income"] = np.log(df.income_pc)
    df["log_cadastral_m2"] = np.log(df.cadastral_value_m2)
    df["log_tourist_intensity"] = np.log1p(df.tourist_places_per_1000)
    df["log_density"] = np.log(df.density_per_km2)
    feats = list(FEATURE_LABELS)

    corr = df[["log_income", *feats]].corr().round(3)
    corr.to_csv(TABLES / "correlation_matrix.csv")
    vif = vif_table(df[feats])
    _table(vif.assign(variable=vif.variable.map(FEATURE_LABELS)), "vif", ".2f")

    X = zscore(df[feats])
    y = df.log_income.rename("log_income")
    ols = ols_robust(y, X)
    (TABLES / "ols_summary.txt").write_text(ols.summary().as_text(), encoding="utf-8")

    # Parsimonious spec drops the cadastral value if it is collinear with education.
    models = spatial_regressions(y, X, w_b)
    coefs = spatial_coef_table(models, feats)
    # Report OLS with HC3 robust SEs (spreg's OLS uses classical SEs).
    hc3 = pd.DataFrame({"coef": ols.params, "se": ols.bse, "p_value": ols.pvalues})
    is_ols = coefs.model == "ols"
    coefs.loc[is_ols, ["coef", "se", "p_value"]] = hc3.values
    sens = ols_robust(y, X.drop(columns="log_cadastral_m2"))
    sens_tab = pd.DataFrame(
        {
            "term": [FEATURE_LABELS.get(t, t) for t in sens.params.index],
            "coef": sens.params.values,
            "se_hc3": sens.bse.values,
            "p_value": sens.pvalues.values,
        }
    )
    _table(sens_tab, "ols_sensitivity_no_cadastral", ".4f")
    RESULTS["ols_sensitivity_r2"] = float(sens.rsquared)
    coefs["term"] = coefs.term.map(lambda t: FEATURE_LABELS.get(t, t))
    _table(coefs, "spatial_regression_coefficients", ".4f")
    fit = spatial_fit_stats(models)
    _table(fit, "spatial_regression_fit", ".3f")
    diag = lm_diagnostics(models["ols"])
    _table(diag, "spatial_lm_diagnostics", ".4f")
    RESULTS["ols_r2"] = float(ols.rsquared)
    RESULTS["ols_adj_r2"] = float(ols.rsquared_adj)
    RESULTS["max_vif"] = float(vif.vif.max())
    RESULTS["lm_diagnostics"] = diag.to_dict(orient="records")
    RESULTS["spatial_fit"] = fit.to_dict(orient="records")

    # Coefficient plot: % change in income per +1 SD of each covariate.
    fig, ax = plt.subplots(figsize=(9, 4.6))
    sub = coefs[coefs.term.isin(FEATURE_LABELS.values())]
    order = list(FEATURE_LABELS.values())[::-1]
    offsets = {"ols": 0.17, "error": 0.0, "lag": -0.17}
    names = {"ols": "OLS", "error": "Spatial error", "lag": "Spatial lag"}
    for m, c in zip(["ols", "error", "lag"], viz.CATEGORICAL):
        d = sub[sub.model == m].set_index("term").loc[order]
        pos = np.arange(len(order)) + offsets[m]
        eff = 100 * (np.exp(d.coef) - 1)
        lo = 100 * (np.exp(d.coef - 1.96 * d.se) - 1)
        hi = 100 * (np.exp(d.coef + 1.96 * d.se) - 1)
        ax.errorbar(
            eff,
            pos,
            xerr=[eff - lo, hi - eff],
            fmt="o",
            color=c,
            ms=7,
            elinewidth=2,
            capsize=0,
            label=names[m],
        )
    ax.axvline(0, color=viz.TEXT_2, lw=0.8)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in income per person for +1 SD (%), 95% CI")
    ax.set_title(f"Correlates of neighbourhood income ({LAST_YEAR}, n = 73)")
    ax.legend(loc="lower right")
    viz.caption(
        fig,
        "Standardised predictors; dependent variable log income per person. "
        "Spatial models by maximum likelihood with Queen contiguity weights. "
        "Associations, not causal effects.",
    )
    viz.save(fig, "08_correlates")
    return df


# =========================================================================== 5. typology
TYPOLOGY_FEATURES = {
    "log_income": "Income (log)",
    "gini_within_sections": "Within-area Gini",
    f"share_tertiary_{LAST_YEAR}": "Tertiary education",
    "share_non_eu": "Non-EU nationals",
    "log_cadastral_m2": "Cadastral value/m2",
    "log_tourist_intensity": "Tourist flats",
    "log_density": "Density",
    "log_income_growth": "Income growth",
    "delta_share_foreign_pp": "Change foreign share",
}


def neighbourhood_typology(barris, df):
    feats = list(TYPOLOGY_FEATURES)
    X = StandardScaler().fit_transform(df[feats])
    scores = select_k(X, range(2, 8))
    _table(scores, "kmeans_model_selection", ".3f")
    # Pre-registered rule: k = 2 merely splits rich from poor, so among k >= 3
    # with bootstrap stability (mean ARI) >= 0.6 and no cluster smaller than 3
    # barris (a singleton is an outlier, not a "type"), take the best silhouette.
    eligible = scores[
        (scores.k >= 3) & (scores.stability_ari >= 0.6) & (scores.min_cluster_size >= 3)
    ]
    k = int(eligible.loc[eligible.silhouette.idxmax(), "k"])
    labels, info = typology(df, feats, k)

    # Relabel clusters by mean income (A = richest) for readable, stable names.
    order = df.assign(c=labels).groupby("c").log_income.mean().sort_values(ascending=False)
    mapping = {c: chr(65 + i) for i, c in enumerate(order.index)}
    df["cluster"] = labels.map(mapping)
    cent = info["centroids_z"].rename(index=mapping).sort_index()
    cent.columns = [TYPOLOGY_FEATURES[c] for c in cent.columns]
    cent.reset_index(names="cluster").to_csv(TABLES / "typology_centroids_z.csv", index=False)
    members = df.groupby("cluster").barri.apply(lambda s: ", ".join(sorted(s))).reset_index()
    members["n"] = df.groupby("cluster").size().values
    _table(members, "typology_members")
    RESULTS["typology"] = {
        "k": k,
        "silhouette": float(info["silhouette"]),
        "ari_vs_ward": float(info["ari_vs_ward"]),
        "sizes": df.cluster.value_counts().sort_index().to_dict(),
    }

    fig = plt.figure(figsize=(15, 6.4))
    ax_map = fig.add_axes([0.0, 0.05, 0.45, 0.85])
    ax_hm = fig.add_axes([0.53, 0.18, 0.45, 0.66])
    g = barris.merge(df[["barri_code", "cluster"]], on="barri_code")
    colors = dict(zip(sorted(df.cluster.unique()), viz.CATEGORICAL))
    g.plot(color=g.cluster.map(colors), ax=ax_map, edgecolor=viz.SURFACE, lw=1)
    for _, r in g.iterrows():
        p = r.geometry.representative_point()
        ax_map.text(
            p.x,
            p.y,
            r.cluster,
            ha="center",
            va="center",
            fontsize=7,
            color=viz.SURFACE,
            fontweight="bold",
        )
    viz.map_axes(ax_map)
    ax_map.legend(
        handles=[Patch(color=c, label=f"Type {k}") for k, c in colors.items()],
        loc="lower right",
        fontsize=9,
    )
    im = ax_hm.imshow(cent.values, cmap=viz.DIVERGING, vmin=-2, vmax=2, aspect="auto")
    ax_hm.set_xticks(range(cent.shape[1]), cent.columns, rotation=35, ha="right")
    ax_hm.set_yticks(range(cent.shape[0]), [f"Type {c}" for c in cent.index])
    ax_hm.grid(False)
    for i in range(cent.shape[0]):
        for j in range(cent.shape[1]):
            v = cent.values[i, j]
            ax_hm.text(
                j,
                i,
                f"{v:+.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color=viz.SURFACE if abs(v) > 1.2 else viz.TEXT,
            )
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.03, pad=0.02)
    cb.set_label("Centroid (z-score)")
    cb.outline.set_visible(False)
    ax_hm.set_title("Cluster profiles (standardised)", fontsize=11)
    fig.suptitle(
        f"A data-driven typology of Barcelona's 73 neighbourhoods (k-means, k = {k})",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    viz.save(fig, "09_typology")
    return df


# =========================================================================== 6. tourism
def tourism(barris, df, huts):
    shares = df.sort_values("tourist_places_per_1000")
    gini_hut = weighted_gini(df.tourist_places_per_1000, df.population)
    top10 = df.nlargest(10, "tourist_places").tourist_places.sum() / df.tourist_places.sum()
    RESULTS["tourist_flats_total"] = int(df.tourist_flats.sum())
    RESULTS["tourist_places_total"] = int(df.tourist_places.sum())
    RESULTS["tourist_top10_share"] = float(top10)
    RESULTS["tourist_gini"] = float(gini_hut)
    RESULTS["corr_tourism_income_spearman"] = float(
        df[["tourist_places_per_1000", "income_pc"]].corr("spearman").iloc[0, 1]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={"width_ratios": [1.3, 1]})
    g = barris.merge(df[["barri_code", "tourist_places_per_1000"]], on="barri_code")
    g.plot(
        "tourist_places_per_1000",
        ax=axes[0],
        cmap=viz.SEQUENTIAL,
        edgecolor=viz.SURFACE,
        lw=0.8,
        scheme="FisherJenks",
        k=6,
        legend=True,
        legend_kwds={
            "title": "Places per 1,000 residents",
            "loc": "upper left",
            "bbox_to_anchor": (0.0, 0.32),
            "fontsize": 8,
            "fmt": "{:.0f}",
            "title_fontsize": 9,
        },
    )
    huts.plot(ax=axes[0], markersize=0.4, color=viz.CATEGORICAL[1], alpha=0.35)
    viz.map_axes(axes[0])
    axes[0].set_title(
        "Licensed tourist-flat places (HUT) per 1,000 residents; dots = flats", fontsize=10
    )

    cum_pop = np.concatenate([[0], shares.population.cumsum() / shares.population.sum()])
    cum_t = np.concatenate([[0], shares.tourist_places.cumsum() / shares.tourist_places.sum()])
    axes[1].plot(cum_pop, cum_t, color=viz.CATEGORICAL[1], label="Tourist-flat places")
    axes[1].plot([0, 1], [0, 1], color=viz.MUTED, lw=1, ls="--", label="Perfect equality")
    axes[1].fill_between(cum_pop, cum_t, cum_pop, color=viz.CATEGORICAL[1], alpha=0.12)
    axes[1].set_xlabel("Cumulative share of residents (barris sorted by intensity)")
    axes[1].set_ylabel("Cumulative share of tourist-flat places")
    axes[1].set_title(f"Concentration curve (Gini = {gini_hut:.2f})")
    axes[1].legend(loc="upper left")
    for ax in axes[1:]:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle(
        "Tourist housing is extremely concentrated",
        x=0.02,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    viz.caption(
        fig,
        "Source: Ajuntament de Barcelona, register of tourist-use dwellings "
        "(current snapshot). Points spatially joined to barri polygons.",
    )
    viz.save(fig, "10_tourism")


# =========================================================================== main
def run():
    viz.set_style()
    barris, secs, sections, panel, cross, huts = load()
    fig_income_maps(barris, secs, sections, cross)
    fig_income_trajectories(panel)
    fig_inequality(sections)
    series_audit(sections)
    convergence(sections, panel, cross)
    covid_shock(sections)
    w_b = spatial_analysis(barris, secs, sections, panel)
    df = correlates(barris, cross, w_b)
    df = neighbourhood_typology(barris, df)
    tourism(barris, df, huts)
    df.to_csv(DATA_PROCESSED / "barri_analysis_table.csv", index=False)

    RESULTS["city_population_last"] = int(panel[panel.year == LAST_YEAR].population.sum())
    RESULTS["city_income_first"] = float(
        np.average(
            sections[sections.year == FIRST_YEAR].income_pc,
            weights=sections[sections.year == FIRST_YEAR].population,
        )
    )
    RESULTS["city_income_last"] = float(
        np.average(
            sections[sections.year == LAST_YEAR].income_pc,
            weights=sections[sections.year == LAST_YEAR].population,
        )
    )
    out = ROOT / "reports" / "results.json"
    save_results(RESULTS, out)
    log.info("Analysis complete: %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
