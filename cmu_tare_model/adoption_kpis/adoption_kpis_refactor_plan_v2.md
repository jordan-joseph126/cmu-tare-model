# Adoption KPIs Module Refactor — Dedicated Metric Modules (v2)

> **Version notes.** v2 — all metrics confirmed state-level. County-level
> COP code archived to `_county_cop_ARCHIVED.py`. Simplified from v1.

## Your Role

Senior Python developer and research software engineer specializing in
energy systems modeling. You understand the TARE model's data pipeline,
the EUSS dataset structure, and the paper's analytical requirements
(Joseph et al. 2026, Energy Policy). You write clean, well-documented
research code following Google/NumPy docstring conventions with full type
hints.

## Project Context

The TARE model's `adoption_kpis` folder computes adoption economics
metrics for heat pump retrofits across the U.S. housing stock. The folder
currently contains a monolithic `kpi_functions.py` module (~400 lines)
that bundles all metric computations, plus `visualize_geospatial_data.py`
for mapping, and two notebooks (preTARE and postTARE) that import from
these modules.

The refactoring goal is to split `kpi_functions.py` into dedicated
modules — one per metric — each with clear docstrings and type hints.
All metrics are state-level. Any existing county-level code is archived
for potential future use. The notebooks need to be simplified into slim
orchestration scripts that import and call, rather than define functions.

**Metric → Module Mapping (all state-level):**

| Metric | Module |
|--------|--------|
| Spark gap (elec/gas price ratio) | `spark_gap.py` |
| Effective annual COP + baseline AFUE | `thermal_cop.py` |
| Break-even COP | `thermal_cop.py` |

Bill savings ratio is handled separately in the postTARE notebook (future task).
Bill impact ratio (`compute_spark_gap_metrics`) is retired — archived in deprecated file only.

**Key formulas:**
- `spark_gap = elec_price_mmbtu / gas_price_mmbtu`
- `thermal_cop = Σ(Q_delivered_kbtu) / Σ(E_hp + E_backup + E_fans_pumps)` per state
- `breakeven_cop = spark_gap × baseline_afue` (COP threshold for bill-neutral electrification)

## Scope Constraint — CRITICAL

**In scope:**
- Splitting `kpi_functions.py` into `spark_gap.py`, `thermal_cop.py`,
  and `data_loading.py`
- Archiving county-level COP code to `_county_cop_ARCHIVED.py`
- Retiring bill impact ratio code (remains only in deprecated file)
- Cleaning up preTARE notebook to import from new modules
- Full docstrings and type hints per coding standards
- Updating `__init__.py` exports

**Out of scope:**
- Bill savings ratio module (`bill_savings.py`) — deferred until updated
  code is provided; will be a separate session
- PostTARE notebook cleanup — deferred to bill savings session
- County-level analysis (archived, not developed further)
- Demand change metrics (stay in postTARE notebook)
- `visualize_geospatial_data.py` — leave as-is
- ResStock 2025.1 migration
- Peak load / grid impact code (`grid_impact/` folder)
- TARE model internals outside `adoption_kpis/`

## Key Design Principle (Non-Negotiable)

**All metrics are state-level.** The paper uses state-level metrics
because they are more robust and match available EIA price data
resolution. County-level COP code that already exists should be archived
to `_county_cop_ARCHIVED.py` with a header comment explaining it is
preserved for potential future work but is not part of the current
analysis pipeline.

## What Was Done Before

### Session 1 (April 8, 2026)
- Extracted 6 shared functions from duplicated preTARE/postTARE notebooks
  into `kpi_functions.py`
- Created `visualize_geospatial_data.py` with `prepare_state_geodataframe`
  and `create_choropleth_map`
- Deleted `get_state_fuel_prices_from_lookup()` (redundant)
- Created `README_adoption_kpis.md` with methodology docs

### Session 2 (April 20-22, 2026)
- Added county-level COP aggregation and break-even COP to `kpi_functions.py`
- Added county-level choropleth support to `visualize_geospatial_data.py`

### This session
- Split `kpi_functions.py` into dedicated per-metric modules
- Archive county-level code
- Retire bill impact ratio (replaced by bill savings ratio in future session)
- Clean up preTARE notebook only (postTARE deferred)
- Ensure consistent coding standards throughout

## Attached Files

- `cmu_tare_model/adoption_kpis/kpi_functions.py` — the monolithic module to split
- `cmu_tare_model/adoption_kpis/visualize_geospatial_data.py` — leave as-is
- `cmu_tare_model/adoption_kpis/__init__.py` — update exports
- `cmu_tare_model/constants.py` — shared constants (HEATING_LOAD_COL, etc.)
- `calculate_preTARE_am_kpis_sparkGap_COP.ipynb` — notebook to clean up

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| `kpi_functions.py` (monolithic) | ✅ Working | Has all functions, needs splitting |
| `visualize_geospatial_data.py` | ✅ Working | Leave as-is |
| State-level spark gap | ✅ Working | Extract to `spark_gap.py` |
| State-level COP | ✅ Working | Extract to `thermal_cop.py` |
| State-level break-even COP | ✅ Working | Extract to `thermal_cop.py` |
| Bill impact ratio | ⛔ Retired | Do not extract; leave in deprecated file |
| Bill savings ratio | ⏳ Deferred | Future session with updated code |
| County-level COP code | ✅ Working | Archive to `_county_cop_ARCHIVED.py` |
| `data_loading.py` | ⬜ Create | Extract shared EUSS loading + constants |
| preTARE notebook cleanup | ⬜ Pending | Depends on module split |
| postTARE notebook cleanup | ⏳ Deferred | Future session with bill savings |

## Required First Action

Open `kpi_functions.py` and inventory every function and constant.
Categorize each into its target module. Identify any county-level code
to archive. Then read both notebooks to confirm which functions each
one uses. Only then begin splitting.

## Tasks

### Task 1 — Create `data_loading.py` (shared EUSS data loading)

**Goal:** Extract data loading functions and shared constants into a
dedicated module that all metric modules import from.

**What moves here:**
- `mp_to_upgrade()` — MP number to EUSS upgrade string conversion
- `load_euss_baseline()` — load and filter baseline CSV
- `load_euss_upgrade()` — load and filter upgrade CSV
- Module-level constants: `BTU_PER_KWH`, `KBTU_PER_KWH`, `KWH_PER_MMBTU`,
  `DWELLING_UNIT_WEIGHT`, `STATE_NAMES`, `EUSS_DATA_DIR`,
  `HEATING_FUEL_COLS`, `HEATING_LOAD_COL`, `GAS_FUEL_COL`,
  `FUEL_PRICE_MAP`, `SHAPEFILE_PATH`, `FUEL_PRICES_PATH`

**Validation gate:** Import in a fresh kernel and call
`load_euss_baseline()` — returns a filtered DataFrame with expected columns.

### Task 2 — Create `spark_gap.py` (state-level price ratios)

**Goal:** Extract spark gap computation into a standalone module.

**What moves here:**
- `calculate_price_ratios()` → rename to `calculate_spark_gap()`

**Signature:**
```python
def calculate_spark_gap(
    filepath: str,
    year: int = 2022,
) -> pd.DataFrame:
```

**Returns:** DataFrame with columns: `state`, `state_name`, `elec_price_kwh`,
`gas_price_kwh`, `elec_price_mmbtu`, `gas_price_mmbtu`, `spark_gap`.

**Reference values:** FL ≈ 1.61, AK ≈ 6.35, national mean ≈ 3.15.

### Task 3 — Create `thermal_cop.py` (state-level COP and break-even COP)

**Goal:** Extract COP computation. State-level only.

**What moves here:**
- `compute_thermal_cop_by_state()` → keep name or rename to `compute_thermal_cop()`
- `compute_breakeven_cop()` — already exists

**Signature:**
```python
def compute_thermal_cop(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    fuel_filter: str = 'Natural Gas',
    verbose: bool = False,
) -> pd.DataFrame:
```

**Reference values:** Std COP ≈ 2.02, high-eff COP ≈ 3.34, AFUE ≈ 0.76.

### Task 4 — Archive county code, retire bill impact ratio, deprecate `kpi_functions.py`

1. Identify any county-level COP/aggregation functions in `kpi_functions.py`
2. Move them to `_county_cop_ARCHIVED.py` with header comment:
   ```python
   # ARCHIVED — County-level COP aggregation code.
   # Preserved for potential future use. Not part of current
   # analysis pipeline (paper uses state-level metrics).
   # See thermal_cop.py for active state-level implementation.
   ```
3. Identify `compute_spark_gap_metrics()` and any `bill_impact_ratio` code.
   Do NOT move to any new module — this metric is retired. It remains only
   in the deprecated file as historical reference
4. Rename remaining `kpi_functions.py` → `_kpi_functions_DEPRECATED.py`
5. Update `__init__.py` to import from new modules (spark_gap, thermal_cop,
   data_loading only — no bill_savings)
6. Verify all package-level imports resolve

### Task 5 — Clean up preTARE notebook

Slim orchestration notebook with zero `def` statements.

**Target structure (≤ 10 cells):**
- Cell 1: Imports
- Cell 2: Configuration (MP selection, batch mode)
- Cell 3: Load EUSS data
- Cell 4: Compute spark gap
- Cell 5: Compute thermal COP
- Cell 6: Compute break-even COP
- Cell 7: Generate maps (spark gap, COP, break-even COP)
- Cell 8: Display results

**Delete from notebook:**
- All `bill_impact_ratio` computation cells, map calls, and display references
- All `compute_spark_gap_metrics()` calls
- All `compute_actual_bill_savings_by_state()` calls
- Old Step 4b (actual bill savings)
- Debug/scratch cells
- Large summary markdown blocks

## Reference Values (golden)

| Metric | Value | Tolerance |
|--------|-------|-----------|
| FL spark gap | 1.61 | ±0.05 |
| AK spark gap | 6.35 | ±0.05 |
| National mean spark gap | 3.15 | ±0.05 |
| Std ASHP COP (national, MP3) | 2.02 | ±0.02 |
| High-eff COP (national, MP4) | 3.34 | ±0.02 |
| Baseline AFUE (national) | 0.76 | ±0.02 |

## Code Standards

1. Google/NumPy docstrings on every public function
2. Full type hints including `from typing import Dict, Optional, Tuple`
3. Input validation with `KeyError` / `ValueError` for missing columns
4. Named constants at module level — no magic numbers in function bodies
5. Strategic comments explaining WHY, not WHAT
6. `verbose: bool = False` parameter on all computation functions
7. f-string formatting for all print statements

## Known Anti-Patterns (do NOT suggest)

| Anti-pattern | Why it's wrong |
|-------------|----------------|
| Adding `geo_level` parameter to state-only functions | All metrics are state-level; county code is archived |
| Omitting fan/pump energy from COP denominator | Root cause of prior inflated COP values (6.0+) |
| Using assumed AFUE constant instead of data-derived | Causes double-counting bug |
| Deleting `kpi_functions.py` before notebook verified | Rename to `_DEPRECATED` first |
| Defining functions in notebooks | All computation logic in `.py` modules |
| Creating a `utils.py` catchall | Use semantically named modules |
| Moving `bill_impact_ratio` code into any new module | Metric is retired; leave only in deprecated file |
| Creating `bill_savings.py` | Deferred to future session with updated bill savings ratio code |
| Touching the postTARE notebook | Out of scope; bill savings work is a separate session |

## Session Summary Template

After completing all tasks, produce a summary covering:
1. Files created (with line counts)
2. Files modified
3. Files deprecated/renamed/archived
4. Any reference values that drifted (and by how much)
5. Confirm all package-level imports resolve
6. Confirm `bill_impact_ratio` grep across active files returns zero matches
7. Deferred work: bill savings ratio module and postTARE notebook cleanup
