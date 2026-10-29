"""Exploratory spatial data analysis: weights, global and local Moran's I."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, Queen, attach_islands

from .config import RANDOM_STATE

PERMUTATIONS = 9999
LISA_LABELS = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}


def queen_weights(gdf: gpd.GeoDataFrame, ids: str):
    """Row-standardised Queen contiguity. Any island (e.g. a polygon separated
    by a gap in the cartography) is linked to its nearest neighbour so that
    every unit has at least one neighbour."""
    w = Queen.from_dataframe(gdf, ids=gdf[ids].tolist(), use_index=False, silence_warnings=True)
    if w.islands:
        w = attach_islands(w, KNN.from_dataframe(gdf, k=1, ids=gdf[ids].tolist()))
    w.transform = "r"
    return w


def global_moran(values, w, seed: int = RANDOM_STATE) -> dict:
    # esda's global Moran draws permutations from NumPy's legacy global RNG.
    np.random.seed(seed)  # noqa: NPY002
    mi = Moran(np.asarray(values, dtype=float), w, permutations=PERMUTATIONS)
    return {"moran_i": mi.I, "expected_i": mi.EI, "z_sim": mi.z_sim, "p_sim": mi.p_sim}


def fdr_threshold(pvalues, alpha: float = 0.05) -> float:
    """Benjamini-Hochberg critical p-value (controls the false discovery rate
    across the many simultaneous local tests)."""
    p = np.sort(np.asarray(pvalues))
    n = len(p)
    passed = p <= alpha * np.arange(1, n + 1) / n
    return float(p[passed].max()) if passed.any() else 0.0


def local_moran(
    values, w, alpha: float = 0.05, fdr: bool = True, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """LISA cluster labels. With ``fdr=True`` significance is assessed against
    the Benjamini-Hochberg threshold instead of a naive per-unit alpha."""
    lisa = Moran_Local(np.asarray(values, dtype=float), w, permutations=PERMUTATIONS, seed=seed)
    threshold = fdr_threshold(lisa.p_sim, alpha) if fdr else alpha
    significant = lisa.p_sim <= threshold
    labels = np.where(significant, pd.Series(lisa.q).map(LISA_LABELS), "Not significant")
    return pd.DataFrame(
        {"local_i": lisa.Is, "p_sim": lisa.p_sim, "quadrant": lisa.q, "cluster": labels},
        index=w.id_order,
    )
