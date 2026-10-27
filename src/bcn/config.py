"""Project-wide configuration: paths, dataset identifiers and study period.

Part I (economic inequality) uses Open Data BCN; Part II (education) adds
student-level and school-level data from the Generalitat de Catalunya open
data portal, modelled in SQL with DuckDB.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

CKAN_API = "https://opendata-ajuntament.barcelona.cat/data/api/3/action"

# Study period: limited by the household disposable income series (2015-2023).
YEARS = list(range(2015, 2024))
FIRST_YEAR, LAST_YEAR = YEARS[0], YEARS[-1]

# Open Data BCN package identifiers (CKAN "name" field).
DATASETS = {
    "boundaries": "20170706-districtes-barris",
    "income": "renda-disponible-llars-bcn",
    "gini": "atles-renda-index-gini",
    "population": "pad_mdbas_nacionalitat-g_sexe",
    "education": "pad_mdb_niv-educa-esta_edat-q_sexe",
    "cadastre": "est-cadastre-habitatges-valor-cadastral",
    "tourist_flats": "habitatges-us-turistic",
}

# Coded categories (Oficina Municipal de Dades, Padró Municipal d'Habitants).
NATIONALITY = {1: "spain", 2: "eu", 3: "non_eu", 4: "unknown"}
EDUCATION = {
    1: "no_studies",
    2: "primary",
    3: "lower_secondary",
    4: "upper_secondary",
    5: "tertiary",
    6: "unknown",
}


# Five-year age groups ("EDAT_Q") are coded so that code k covers ages
# 5k to 5k + 4: code 4 = 20-24, code 5 = 25-29, code 6 = 30-34, ...
# (verified against attainment profiles; code 3 holds the 16-19 year-olds).
def age_group_code(age: int) -> int:
    """EDAT_Q code of the five-year group containing ``age``."""
    return age // 5


CRS_METRIC = "EPSG:25831"  # ETRS89 / UTM zone 31N, native CRS of the city's cartography

RANDOM_STATE = 42

# =========================================================================== Part II
SOCRATA_API = "https://analisi.transparenciacatalunya.cat"

# Generalitat de Catalunya open data (Socrata ids). ``where`` restricts the
# download server-side; student assessments are kept for the whole of
# Catalonia so Barcelona can be benchmarked against the rest of the region.
EDU_DATASETS = {
    "eso4_results": {"id": "59vm-wwq7", "where": None},
    "primary6_results": {"id": "rk5x-gny6", "where": None},
    "school_enrolment": {"id": "xvme-26kg", "where": "nom_municipi = 'Barcelona'"},
    "school_admissions": {"id": "99md-r3rq", "where": "nom_municipi = 'Barcelona'"},
}

# Register-based educational attainment by barri: the full published series.
PADRO_EDU_YEARS = list(range(1997, 2027))

SQL_DIR = ROOT / "sql"
WAREHOUSE = ROOT / "data" / "warehouse" / "education.duckdb"
EDU_PROCESSED = DATA_PROCESSED / "education"
EDU_FIGURES = FIGURES / "education"
EDU_TABLES = TABLES / "education"
