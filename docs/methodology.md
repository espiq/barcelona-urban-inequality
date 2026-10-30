# Methodology, data dictionary and limitations

The study has two parts that share one codebase, one geography (73 barris) and one income
measure:

- **Part I — economic inequality** ([report](../reports/01_economic_inequality.md)), in
  Python and pandas.
- **Part II — education** ([report](../reports/02_education.md)), modelled in SQL on
  DuckDB, with models in Python.

## Pipeline

```
Open Data BCN (CKAN API)              Dades Obertes de Catalunya (Socrata API)
        │  bcn.download  - cached, retried, paginated, SHA-256 manifest per file
        ▼                                         ▼
data/raw/<dataset>/*.csv   (git-ignored; ~400 MB in total)
        │                                         │
        │ Part I: bcn.clean (pandas)              │ Part II: bcn.education.warehouse
        ▼                                         ▼
data/processed/*.csv|geojson ──── ref.* ────▶ data/warehouse/education.duckdb (git-ignored)
        │                                   raw → staging → marts  (sql/)   + 9 SQL data tests
        │                                         │ exported marts
        ▼                                         ▼
 bcn.analysis                          data/processed/education/*.csv (committed)
        │                                         │ bcn.education.analysis (sql/analysis + models)
        ▼                                         ▼
reports/figures, tables, results.json   reports/figures/education, tables/education,
                                        results_education.json
```

`python -m bcn` runs everything. `python -m bcn edu-report` reproduces Part II from the
committed marts without downloading the raw data. Only the student-level models need the
full warehouse.

## Geographic keys

| Unit | Count | Key used here | Notes |
|---|---:|---|---|
| District | 10 | `district_code` (1–10) | |
| Barri (neighbourhood) | 73 | `barri_code` (1–73) | Stable since 2009; the register republishes 1997+ on the 73 barris |
| Census section | 1,068 | `section_id` = district × 1000 + section | The income files number sections *within* a district (1..n). The register uses the DSSS form. Both reconcile 1:1 in every year. |
| School | ~1,000 | `school_code` (Generalitat `codi_centre`) | Placed in a barri by point-in-polygon (nearest barri within 500 m) |

## Code lists (municipal register)

- **Nationality group:** 1 Spain · 2 other EU · 3 rest of world · 4 unknown.
- **Education level** (`NIV_EDUCA_esta`): 1 no studies · 2 primary · 3 lower secondary
  (ESO, graduat escolar, FP I) · 4 upper secondary / intermediate vocational · 5 tertiary
  (university, higher vocational) · 6 unknown.
- **Five-year age group** (`EDAT_Q`): code *k* covers ages 5*k* to 5*k*+4 (4 = 20–24,
  5 = 25–29, 6 = 30–34, …; code 3 holds the 16–19 year-olds). This was verified against
  attainment profiles and population totals, and is encoded once in
  `config.age_group_code`. *An earlier version of Part I treated code 6 as 25–29. This was
  corrected; the conclusions were unchanged.*

---

## Part I — processed tables

### `section_panel.csv` (section × year, 2015–2023, 9,612 rows)

| Column | Description |
|---|---|
| `population` | Registered residents (1 January) |
| `pop_spain`, `pop_eu`, `pop_non_eu`, `pop_unknown` | Residents by nationality group |
| `income_pc` | Household disposable income per person (€, nominal) |
| `gini` | Gini index of income within the section (INE atlas, 0–100) |

### `barri_panel.csv` (barri × year)

Population-weighted aggregates of the above, plus `share_foreign`, `share_non_eu`,
`density_per_km2`, `gini_within_sections` (population-weighted mean of section Ginis), and
`income_cv_between_sections`.

### `barri_cross_section.csv` / `barri_analysis_table.csv` (73 rows, 2023)

| Column | Definition |
|---|---|
| `share_tertiary_YYYY` | Residents aged 25+ with tertiary education ÷ residents aged 25+ with known level |
| `share_low_education_YYYY` | No studies + primary only, same denominator |
| `cadastral_value_m2` | Σ cadastral value ÷ Σ residential floor area (floor area recovered as total ÷ unit value per section) |
| `tourist_flats`, `tourist_places` | Licensed HUT flats / bed places, by spatial join |
| `tourist_places_per_1000` | Places per 1,000 residents |
| `income_growth_pct`, `log_income_growth` | 2015 → 2023 |
| `cluster` | Typology label (A/B/C) |

---

## Part II — the SQL warehouse

### Layers

| Schema | Built by | Content |
|---|---|---|
| `raw` | Python (`read_csv`, all VARCHAR) | Source files verbatim, so every cast is explicit in SQL |
| `ref` | Python | `barris`, `barri_income` (from Part I), `school_location` (GeoPandas spatial join) |
| `staging` | `sql/staging/*.sql` | One typed, harmonised model per source, plus macros (`to_num`, `register_count`, `school_sector`, `student_gender`) |
| `marts` | `sql/marts/*.sql` | Analysis-ready models (below) |
| — | `sql/analysis/q*.sql` | Analytical queries; each one becomes a table in `reports/tables/education/` |
| — | `sql/tests/assert_*.sql` | Data tests: every query must return zero rows or the build fails |

### Marts

| Mart | Grain | Key definitions |
|---|---|---|
| `fct_assessments` | student × test (2.0M) | Subject scores z-standardised within (test, year) over all of Catalonia; `z_core` = mean of ≥3 subject z-scores; `is_low_performer` = bottom 20% of the cohort |
| `mart_assessment_summary` | test × year × area × sector × gender | Means of z-scores, share of low performers, share overage |
| `mart_student_trajectories` | student (6th-of-primary cohorts 2016+) | First primary test linked to the first later ESO test by anonymised id; outcome = `on_time` (4 years later), `delayed` (5–6), `not_observed`; quintiles at 12 and 16 |
| `mart_trajectory_summary` | cohort × area × sector × gender × quintile × outcome | Compact export of the above (complete follow-up only) |
| `mart_barri_attainment` | barri × year (1997–2026) | `share_low_20_24`, `share_low_25_34` (at most lower secondary), `share_tertiary_25_34`, `share_tertiary_25_64`; Part I income; income quintile per year and fixed at 2015 |
| `mart_barri_schooling` | barri × school year | Primary + ESO enrolment by funding: public / concerted (private, publicly funded) / independent |
| `mart_school_demand` | school × year × entry grade (P3, ESO1) | `fill_rate` = assigned ÷ places offered; `first_choice_rate`; 2015 income quintile of the school's barri |

### Definitions and decisions

1. **Score standardisation.** Each year's test is different, so raw scores are not
   comparable over time. All comparisons use z-scores within (test, year, subject),
   computed over the whole of Catalonia.
2. **Overage** = born before *year* − 12 (primary) or *year* − 16 (ESO). In Spain this
   overwhelmingly reflects grade repetition.
3. **Early school leaving proxy** = residents aged 20–24 whose registered attainment is at
   most lower secondary. This is *not* the official Eurostat indicator (18–24, not in
   education or training, from the Labour Force Survey). It is used only for comparisons
   across barris and over time within one consistent series.
4. **Consistent period.** The register's attainment data were refreshed with the 2001 and
   2011 censuses, which creates level breaks in 2002 and 2012. Trend statements use
   2012–2026 only.
5. **Fixed income quintiles.** Long-run trends group barris by their *2015* income
   quintile, so a barri changing quintile cannot create a spurious trend.
6. **"Not observed" is an upper bound for leaving the system.** It includes students who
   moved out of Catalonia or into special education.
7. **Value added.** `z_core_16 ~ private + z_core_12 + female + overage + cohort FE`,
   estimated separately for Barcelona and the rest of Catalonia, with HC1 errors. The
   private coefficient is an upper bound for a school effect, because unobserved family
   resources and peers remain.

### Data-quality rules (each enforced by a SQL data test)

| Rule | Test |
|---|---|
| Raw scores in [0, 100] | `assert_scores_in_range` |
| ≤ 0.5% of records with an unmapped sector or gender per test-year | `assert_assessment_dimensions_known` |
| Zero-inflated subject-cohorts (> 1% exact zeros: 2016 ESO maths, 11%) have zeros treated as missing | `assert_no_zero_inflated_cohorts` |
| z-scores have mean 0 and SD 1 per cohort | `assert_z_scores_standardised` |
| One trajectory per student; a quintile at 16 exists if and only if an ESO test was observed | `assert_unique_trajectories`, `assert_trajectory_quintiles_consistent` |
| 73 barris per year, shares in [0, 1] | `assert_barri_attainment_complete` |
| < 0.5% of compulsory enrolment in schools that cannot be geolocated | `assert_schools_located` |
| Funding shares sum to 1 | `assert_enrolment_shares_sum_to_one` |
| School years with < 90% of the previous year's records are excluded (partial publication) | logic in `mart_barri_schooling` |

---

## Shared methodological decisions

1. **Population weighting.** Every city-wide statistic weights areas by residents.
2. **Suppressed counts.** Register cells < 5 are published as "..". We impute 2.5, the
   expected value of U{1..4}. The bias is at most ±2.5 per suppressed cell.
3. **Log income** in all spatial statistics and regressions.
4. **Multiple testing.** LISA significance uses the Benjamini–Hochberg FDR at 5%.
5. **Pre-stated model-selection rules** (k-means *k*, robustness windows, consistent
   period), fixed before looking at the results.
6. **Robust inference.** HC3/HC1 for cross-sections; standard errors clustered by barri
   for section-level and panel regressions.
7. **Determinism.** Fixed random seeds for permutation inference, k-means and bootstraps.

## Testing

- `clean._validate` (Part I) and the SQL data tests (Part II) stop the pipeline on
  violated invariants.
- `pytest` (30 tests):
  - Statistical functions with analytically known answers: Gini, Theil exactness,
    concentration index, recovery of a simulated β, Benjamini–Hochberg.
  - Data contracts on the committed processed tables and marts.
  - `test_sql_models.py`, which runs the real staging and mart SQL on a synthetic
    warehouse. It covers category harmonisation, overage, standardisation, the
    zero-inflation rule, trajectory linkage (including a regression test for the
    `least()`/NULL bug), suppression imputation and funding shares.

## Limitations

See §11 of Part I and §7 of Part II.
