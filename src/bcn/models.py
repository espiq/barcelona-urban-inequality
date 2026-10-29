"""Explanatory models and neighbourhood typology.

* Cross-sectional correlates of neighbourhood income (OLS -> spatial
  diagnostics -> spatial lag / spatial error models).
* Unsupervised typology of the 73 barris (k-means with silhouette-based
  model selection and bootstrap stability).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from spreg import OLS as SpOLS
from spreg import ML_Error, ML_Lag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import RANDOM_STATE


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    Xc = sm.add_constant(X)
    return pd.DataFrame(
        {
            "variable": X.columns,
            "vif": [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])],
        }
    )


def ols_robust(y: pd.Series, X: pd.DataFrame):
    """OLS with HC3 heteroskedasticity-robust standard errors."""
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")


def spatial_regressions(y: pd.Series, X: pd.DataFrame, w) -> dict:
    """OLS with Lagrange-Multiplier diagnostics, plus ML spatial lag and error
    models (Anselin, 1988). Returns fitted spreg objects."""
    yv = y.to_numpy().reshape(-1, 1)
    Xv = X.to_numpy()
    names = list(X.columns)
    ols = SpOLS(yv, Xv, w=w, spat_diag=True, moran=True, name_y=y.name, name_x=names)
    lag = ML_Lag(yv, Xv, w=w, name_y=y.name, name_x=names)
    err = ML_Error(yv, Xv, w=w, name_y=y.name, name_x=names)
    return {"ols": ols, "lag": lag, "error": err}


def spatial_coef_table(models: dict, names: list[str]) -> pd.DataFrame:
    """Side-by-side coefficient table for the three spreg models."""
    rows = []
    for key, m in models.items():
        betas = np.asarray(m.betas).ravel()
        if key == "ols":
            se = np.sqrt(np.diag(m.vm))
            labels = ["const", *names]
        else:
            se = np.asarray(m.std_err).ravel()
            labels = ["const", *names, "rho" if key == "lag" else "lambda"]
        z = betas / se
        p = 2 * (1 - norm.cdf(np.abs(z)))
        for lab, b, s, pv in zip(labels, betas, se, p):
            rows.append({"model": key, "term": lab, "coef": b, "se": s, "p_value": pv})
    return pd.DataFrame(rows)


def spatial_fit_stats(models: dict) -> pd.DataFrame:
    ols, lag, err = models["ols"], models["lag"], models["error"]
    return pd.DataFrame(
        [
            {"model": "OLS", "log_likelihood": ols.logll, "aic": ols.aic, "r2": ols.r2},
            {
                "model": "Spatial lag (ML)",
                "log_likelihood": lag.logll,
                "aic": lag.aic,
                "r2": lag.pr2,
            },
            {
                "model": "Spatial error (ML)",
                "log_likelihood": err.logll,
                "aic": err.aic,
                "r2": err.pr2,
            },
        ]
    )


def lm_diagnostics(ols) -> pd.DataFrame:
    tests = {
        "Moran's I (residuals)": (ols.moran_res[0], ols.moran_res[2]),
        "LM lag": ols.lm_lag,
        "Robust LM lag": ols.rlm_lag,
        "LM error": ols.lm_error,
        "Robust LM error": ols.rlm_error,
    }
    return pd.DataFrame([{"test": k, "statistic": v[0], "p_value": v[1]} for k, v in tests.items()])


# --------------------------------------------------------------------------- typology
def select_k(X: np.ndarray, k_range=range(2, 9), n_boot: int = 200) -> pd.DataFrame:
    """Score candidate k by silhouette and bootstrap stability (mean ARI between
    the full-sample partition and partitions fitted on bootstrap resamples)."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for k in k_range:
        base = KMeans(k, n_init=100, random_state=RANDOM_STATE).fit(X)
        aris = []
        for _ in range(n_boot):
            idx = rng.choice(len(X), len(X), replace=True)
            km = KMeans(k, n_init=10, random_state=int(rng.integers(1e9))).fit(X[idx])
            aris.append(adjusted_rand_score(base.labels_, km.predict(X)))
        rows.append(
            {
                "k": k,
                "inertia": base.inertia_,
                "silhouette": silhouette_score(X, base.labels_),
                "stability_ari": float(np.mean(aris)),
                "min_cluster_size": int(np.bincount(base.labels_).min()),
            }
        )
    return pd.DataFrame(rows)


def typology(df: pd.DataFrame, features: list[str], k: int) -> tuple[pd.Series, dict]:
    """Fit the final k-means and compare with Ward hierarchical clustering."""
    X = StandardScaler().fit_transform(df[features])
    km = KMeans(k, n_init=100, random_state=RANDOM_STATE).fit(X)
    ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
    info = {
        "silhouette": silhouette_score(X, km.labels_),
        "ari_vs_ward": adjusted_rand_score(km.labels_, ward.labels_),
        "centroids_z": pd.DataFrame(km.cluster_centers_, columns=features),
    }
    return pd.Series(km.labels_, index=df.index, name="cluster"), info
