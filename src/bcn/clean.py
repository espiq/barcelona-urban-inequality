"""Turn raw Open Data BCN files into tidy, analysis-ready tables.

Outputs (``data/processed/``):

* ``sections.geojson``        - 1,068 census sections with geometry.
* ``barris.geojson``          - 73 neighbourhoods (barris) with geometry and area.
* ``section_panel.csv``       - section x year: residents, income per capita, Gini.
* ``barri_panel.csv``         - barri x year: population-weighted aggregates.
* ``barri_cross_section.csv`` - barri-level covariates for the latest year.

Design choices are documented inline and in ``docs/methodology.md``.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt

from .config import (
    CRS_METRIC,
    DATA_PROCESSED,
    DATA_RAW,
    EDUCATION,
    FIRST_YEAR,
    LAST_YEAR,
    NATIONALITY,
    YEARS,
    age_group_code,
)

log = logging.getLogger(__name__)

# Statistical disclosure control: the register publishes counts < 5 as "..".
# We impute the expected value of a uniform draw on {1,...,4}.
SUPPRESSED_IMPUTATION = 2.5


def _read(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def _count(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("..", SUPPRESSED_IMPUTATION), errors="coerce")


def section_key(district: pd.Series, section: pd.Series) -> pd.Series:
    """Unique city-wide census section id: DDSSS (district * 1000 + section)."""
    return district.astype(int) * 1000 + section.astype(int)


# --------------------------------------------------------------------------- geometry
def load_geometries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    raw = DATA_RAW / "boundaries"

    barris = _read(raw / "BarcelonaCiutat_Barris.csv")
    barris = gpd.GeoDataFrame(
        {
            "barri_code": barris["codi_barri"].astype(int),
            "barri": barris["nom_barri"],
            "district_code": barris["codi_districte"].astype(int),
            "district": barris["nom_districte"],
        },
        geometry=barris["geometria_etrs89"].map(wkt.loads),
        crs=CRS_METRIC,
    )
    barris["area_km2"] = barris.area / 1e6

    secs = _read(raw / "BarcelonaCiutat_SeccionsCensals.csv")
    secs = gpd.GeoDataFrame(
        {
            "section_id": section_key(secs["codi_districte"], secs["codi_seccio_censal"]),
            "barri_code": secs["codi_barri"].astype(int),
            "district_code": secs["codi_districte"].astype(int),
        },
        geometry=secs["geometria_etrs89"].map(wkt.loads),
        crs=CRS_METRIC,
    )
    secs["area_km2"] = secs.area / 1e6
    return barris.sort_values("barri_code"), secs.sort_values("section_id")


# --------------------------------------------------------------------------- section panel
def load_section_panel() -> pd.DataFrame:
    """Section x year table with residents by nationality, income and Gini."""
    frames = []
    for year in YEARS:
        inc = _read(DATA_RAW / "income" / f"{year}_income.csv")
        inc["section_id"] = section_key(inc["Codi_Districte"], inc["Seccio_Censal"])
        inc = inc.rename(
            columns={
                "Import_Euros": "income_pc",
                "Codi_Barri": "barri_code",
                "Codi_Districte": "district_code",
            }
        )

        gini = _read(DATA_RAW / "gini" / f"{year}_gini.csv")
        gini["section_id"] = section_key(gini["Codi_Districte"], gini["Seccio_Censal"])

        pop = _read(DATA_RAW / "population" / f"{year}_population.csv")
        pop["residents"] = _count(pop["Valor"])
        pop["nat"] = pop["NACIONALITAT_G"].map(NATIONALITY)
        pop = (
            pop.pivot_table(index="Seccio_Censal", columns="nat", values="residents", aggfunc="sum")
            .reindex(columns=list(NATIONALITY.values()), fill_value=0)
            .fillna(0)
            .add_prefix("pop_")
        )
        pop["population"] = pop.sum(axis=1)
        pop.index.name = "section_id"

        df = (
            inc[["section_id", "district_code", "barri_code", "income_pc"]]
            .merge(gini[["section_id", "Index_Gini"]], on="section_id", how="left")
            .merge(pop.reset_index(), on="section_id", how="left", validate="1:1")
            .rename(columns={"Index_Gini": "gini"})
        )
        unmatched = df["population"].isna().sum()
        if unmatched:
            log.warning("%s: %d sections without population match", year, unmatched)
        df["year"] = year
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    return panel[
        [
            "year",
            "section_id",
            "district_code",
            "barri_code",
            "population",
            "pop_spain",
            "pop_eu",
            "pop_non_eu",
            "pop_unknown",
            "income_pc",
            "gini",
        ]
    ]


# --------------------------------------------------------------------------- barri panel
def _wmean(df: pd.DataFrame, value: str, weight: str = "population") -> float:
    w = df[weight]
    return float(np.average(df[value], weights=w)) if w.sum() > 0 else np.nan


def build_barri_panel(sections: pd.DataFrame, barris: gpd.GeoDataFrame) -> pd.DataFrame:
    """Population-weighted barri aggregates.

    Disposable income is published *per person*, so the barri mean is the
    population-weighted mean of section means. The section Gini cannot be
    aggregated exactly without micro-data; we report its population-weighted
    mean as an indicator of *within-section* inequality, and separately the
    between-section coefficient of variation inside each barri.
    """
    g = sections.groupby(["barri_code", "year"])
    out = pd.DataFrame(
        {
            "population": g["population"].sum(),
            "pop_spain": g["pop_spain"].sum(),
            "pop_eu": g["pop_eu"].sum(),
            "pop_non_eu": g["pop_non_eu"].sum(),
            "income_pc": g.apply(_wmean, "income_pc", include_groups=False),
            "gini_within_sections": g.apply(_wmean, "gini", include_groups=False),
            "income_cv_between_sections": g["income_pc"].agg(lambda s: s.std(ddof=0) / s.mean()),
            "n_sections": g.size(),
        }
    ).reset_index()
    out["share_foreign"] = (out["pop_eu"] + out["pop_non_eu"]) / out["population"]
    out["share_non_eu"] = out["pop_non_eu"] / out["population"]

    meta = barris[["barri_code", "barri", "district_code", "district", "area_km2"]]
    out = out.merge(pd.DataFrame(meta), on="barri_code", validate="m:1")
    out["density_per_km2"] = out["population"] / out["area_km2"]
    return out.sort_values(["barri_code", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------- covariates
def load_education(year: int) -> pd.DataFrame:
    """Share of residents aged 25+ with tertiary education, by barri.

    Restricting to 25+ (five-year age-group code >= 5) avoids counting
    students who have not yet had the chance to graduate.
    Unknown level is excluded from the denominator.
    """
    ed = _read(DATA_RAW / "education" / f"{year}_education.csv")
    ed["n"] = _count(ed["Valor"])
    ed["level"] = ed["NIV_EDUCA_esta"].map(EDUCATION)
    ed = ed[(ed["EDAT_Q"] >= age_group_code(25)) & (ed["level"] != "unknown")]
    tab = ed.pivot_table(index="Codi_Barri", columns="level", values="n", aggfunc="sum")
    total = tab.sum(axis=1)
    return pd.DataFrame(
        {
            "barri_code": tab.index.astype(int),
            f"share_tertiary_{year}": (tab["tertiary"] / total).values,
            f"share_low_education_{year}": ((tab["no_studies"] + tab["primary"]) / total).values,
        }
    )


def load_cadastre(year: int) -> pd.DataFrame:
    """Barri cadastral value per m2 of residential floor space.

    Raw values are section totals and unit values; floor area is recovered as
    total / unit, so the barri unit value is a floor-area-weighted mean.
    """
    cad = _read(DATA_RAW / "cadastre" / f"{year}_cadastre.csv")
    desc = cad["Desc_valors"].astype(str)
    cad["metric"] = np.select(
        [
            desc.str.startswith("Valor_cadastral_total"),
            desc.str.startswith("Valor_cadastral_unitari"),
        ],
        ["total", "unit"],
        default="other",
    )
    cad = cad[cad["metric"] != "other"]
    wide = cad.pivot_table(
        index=["Codi_barri", "Seccio_censal"], columns="metric", values="Valors", aggfunc="sum"
    )
    wide["floor_m2"] = wide["total"] / wide["unit"]
    agg = wide.groupby(level="Codi_barri")[["total", "floor_m2"]].sum()
    return pd.DataFrame(
        {
            "barri_code": agg.index.astype(int),
            "cadastral_value_m2": (agg["total"] / agg["floor_m2"]).values,
            "residential_floor_km2": (agg["floor_m2"] / 1e6).values,
        }
    )


def load_tourist_flats(barris: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """Licensed tourist flats (HUT). Points are re-assigned to barris spatially
    rather than trusting the registry's barri field, which has gaps."""
    hut = _read(DATA_RAW / "tourist_flats" / "hut_comunicacio_opendata.csv")
    hut = hut.dropna(subset=["LONGITUD_X", "LATITUD_Y"])
    pts = gpd.GeoDataFrame(
        {"places": pd.to_numeric(hut["NUMERO_PLACES"], errors="coerce")},
        geometry=gpd.points_from_xy(hut["LONGITUD_X"], hut["LATITUD_Y"]),
        crs="EPSG:4326",
    ).to_crs(CRS_METRIC)
    pts = gpd.sjoin(pts, barris[["barri_code", "geometry"]], how="inner", predicate="within")
    agg = pts.groupby("barri_code").agg(
        tourist_flats=("places", "size"), tourist_places=("places", "sum")
    )
    agg = agg.reindex(barris["barri_code"], fill_value=0).reset_index()
    return agg, pts.drop(columns="index_right")


def build_cross_section(
    panel: pd.DataFrame, barris: gpd.GeoDataFrame
) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    last = panel[panel["year"] == LAST_YEAR].copy()
    first = panel[panel["year"] == FIRST_YEAR].set_index("barri_code")

    last["income_pc_first"] = last["barri_code"].map(first["income_pc"])
    last["population_first"] = last["barri_code"].map(first["population"])
    last["share_foreign_first"] = last["barri_code"].map(first["share_foreign"])
    last["income_growth_pct"] = 100 * (last["income_pc"] / last["income_pc_first"] - 1)
    last["log_income_growth"] = np.log(last["income_pc"] / last["income_pc_first"])
    last["pop_growth_pct"] = 100 * (last["population"] / last["population_first"] - 1)
    last["delta_share_foreign_pp"] = 100 * (last["share_foreign"] - last["share_foreign_first"])

    for year in (FIRST_YEAR, LAST_YEAR):
        last = last.merge(load_education(year), on="barri_code", how="left")
    last["delta_tertiary_pp"] = 100 * (
        last[f"share_tertiary_{LAST_YEAR}"] - last[f"share_tertiary_{FIRST_YEAR}"]
    )

    last = last.merge(load_cadastre(LAST_YEAR), on="barri_code", how="left")
    hut, hut_points = load_tourist_flats(barris)
    last = last.merge(hut, on="barri_code", how="left")
    last["tourist_places_per_1000"] = 1000 * last["tourist_places"] / last["population"]
    return last.reset_index(drop=True), hut_points


# --------------------------------------------------------------------------- run
def run() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    barris, secs = load_geometries()
    sections = load_section_panel()
    panel = build_barri_panel(sections, barris)
    cross, hut_points = build_cross_section(panel, barris)

    _validate(sections, panel, cross)

    barris.to_file(DATA_PROCESSED / "barris.geojson", driver="GeoJSON")
    secs.to_file(DATA_PROCESSED / "sections.geojson", driver="GeoJSON")
    hut_points.to_file(DATA_PROCESSED / "tourist_flats.geojson", driver="GeoJSON")
    sections.to_csv(DATA_PROCESSED / "section_panel.csv", index=False)
    panel.to_csv(DATA_PROCESSED / "barri_panel.csv", index=False)
    cross.to_csv(DATA_PROCESSED / "barri_cross_section.csv", index=False)
    log.info("Processed data written to %s", DATA_PROCESSED)


def _validate(sections: pd.DataFrame, panel: pd.DataFrame, cross: pd.DataFrame) -> None:
    """Fail loudly if the data violate basic invariants."""
    assert sections.groupby("year").size().eq(1068).all(), "expected 1,068 sections per year"
    assert sections["income_pc"].gt(0).all(), "non-positive income found"
    assert sections["population"].notna().all(), "sections without population"
    assert panel.groupby("year")["barri_code"].nunique().eq(73).all(), "expected 73 barris"
    assert cross["cadastral_value_m2"].notna().all(), "missing cadastral values"
    city_pop = panel.groupby("year")["population"].sum()
    assert city_pop.between(1.5e6, 1.8e6).all(), f"implausible city population: {city_pop}"
    for share in ("share_foreign", f"share_tertiary_{LAST_YEAR}"):
        assert cross[share].between(0, 1).all(), f"{share} outside [0, 1]"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
