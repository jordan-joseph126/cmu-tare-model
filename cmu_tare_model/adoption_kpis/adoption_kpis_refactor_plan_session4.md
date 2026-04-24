# Adoption KPIs Refactor — Session 4: Demand, Bill Savings, postTARE Notebook

> **Version notes.** Session 4 plan. Creates `demand.py` and `bill_savings.py`
> modules, corrects the bill savings ratio code to use TARE lifetime fuel costs,
> and cleans up the postTARE notebook.

## Your Role

Senior Python developer and research software engineer specializing in
energy systems modeling. You understand the TARE model's data pipeline,
the EUSS dataset structure, and the paper's analytical requirements
(Joseph et al. 2026, Energy Policy). You write clean, well-documented
research code following Google/NumPy docstring conventions with full type
hints.

## Project Context

Session 3 successfully split the monolithic `kpi_functions.py` into
`data_loading.py`, `spark_gap.py`, and `thermal_cop.py`, and cleaned
up the preTARE notebook. The `_kpi_functions_DEPRECATED.py` file still
contains two demand functions (`compute_scenario_demand`,
`aggregate_demand_by_state`) and retired bill impact ratio code.

This session extracts the demand functions into `demand.py`, creates a
new `bill_savings.py` module with corrected bill savings ratio code that
uses actual TARE lifetime fuel cost data (not the old analytical
bill_impact_ratio), and cleans up the postTARE notebook.

**Current folder structure (after Session 3):**
```
cmu_tare_model/adoption_kpis/
├── __init__.py                      # exports spark_gap, thermal_cop, data_loading
├── data_loading.py                  # ✅ EUSS loading + constants
├── spark_gap.py                     # ✅ state-level price ratios
├── thermal_cop.py                   # ✅ state-level COP + break-even
├── visualize_geospatial_data.py     # ✅ unchanged
├── _kpi_functions_DEPRECATED.py     # still contains demand functions
├── _county_cop_ARCHIVED.py          # archived county code
└── README_adoption_kpis.md          # methodology docs
```

**Target folder structure (after this session):**
```
cmu_tare_model/adoption_kpis/
├── __init__.py                      # updated — adds demand + bill_savings exports
├── data_loading.py                  # unchanged
├── spark_gap.py                     # unchanged
├── thermal_cop.py                   # unchanged
├── demand.py                        # NEW — scenario demand + state aggregation
├── bill_savings.py                  # NEW — bill savings ratio from TARE data
├── visualize_geospatial_data.py     # unchanged
├── _kpi_functions_DEPRECATED.py     # can be deleted after verification
├── _county_cop_ARCHIVED.py          # unchanged
└── README_adoption_kpis.md          # methodology docs
```

## Scope Constraint — CRITICAL

**In scope:**
- Creating `demand.py` (extract from `_kpi_functions_DEPRECATED.py`)
- Creating `bill_savings.py` (new code — NOT the old `bill_impact_ratio`)
- Cleaning up the postTARE notebook
- Updating `__init__.py` to export new modules
- Deleting `_kpi_functions_DEPRECATED.py` after verification

**Out of scope:**
- preTARE notebook (already clean from Session 3)
- `spark_gap.py`, `thermal_cop.py`, `data_loading.py` (already done)
- County-level analysis
- ResStock 2025.1 migration
- NPV analysis (strip from postTARE notebook, do not build)

## Key Design Decision: Bill Savings Ratio vs Bill Impact Ratio

The old `bill_impact_ratio` was an analytical approximation:
`bill_impact_ratio = spark_gap × (baseline_afue / thermal_cop)`

The new `bill_savings_ratio` uses actual TARE model outputs — per-building
lifetime fuel costs that already incorporate building-specific consumption,
state-level fuel prices, and system efficiency:

`bill_savings_ratio = retrofit_lifetime_fuel_cost / baseline_lifetime_fuel_cost`

**Data source:** TARE model exports accessed via
`DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']`

**Available columns per MP:**
- `baseline_heating_lifetime_fuel_cost` — baseline furnace fuel cost over system lifetime
- `iraRef_mp{N}_heating_lifetime_fuel_cost` — heat pump electricity cost (with IRA reference case)
- `preIRA_mp{N}_heating_lifetime_fuel_cost` — heat pump electricity cost (no IRA)
- `iraRef_mp{N}_heating_lifetime_savings_fuel_cost` — savings (baseline - retrofit)
- `preIRA_mp{N}_heating_lifetime_savings_fuel_cost` — savings (no IRA)

**Interpretation:** ratio < 1 = HP saves money; ratio > 1 = HP costs more.

**Aggregation:** state-level median (not mean — median is robust to outlier
buildings with extreme consumption patterns).

## What Was Done Before

### Session 1 (April 8, 2026)
- Extracted shared functions into monolithic `kpi_functions.py`
- Created `visualize_geospatial_data.py`

### Session 2 (April 20-22, 2026)
- Added county-level COP (now archived)

### Session 3 (April 24, 2026)
- Split `kpi_functions.py` into `data_loading.py`, `spark_gap.py`, `thermal_cop.py`
- Archived county code, deprecated `kpi_functions.py`
- Cleaned up preTARE notebook
- Retired bill impact ratio metric

### This session (Session 4)
- Extract demand functions to `demand.py`
- Create `bill_savings.py` with corrected TARE-based ratio
- Clean up postTARE notebook
- Delete `_kpi_functions_DEPRECATED.py`

## Attached Files

- `cmu_tare_model/adoption_kpis/_kpi_functions_DEPRECATED.py` — source for demand functions
- `calculate_postTARE_am_kpis_demand_bill_savings_NPV.ipynb` — notebook to clean up
- `cmu_tare_model/adoption_kpis/__init__.py` — update exports

## Tasks

### Task 1 — Create `demand.py`

**Goal:** Extract demand change functions from `_kpi_functions_DEPRECATED.py`.

**What moves here (preserve existing logic exactly):**
- `compute_scenario_demand()` — per-building heating demand change
- `aggregate_demand_by_state()` — state-level aggregation in GWh

**Dependencies to import from `data_loading`:**
- `HEATING_FUEL_COLS`, `HP_BACKUP_ELEC_COL`, `HP_FANS_PUMPS_COL`, `KBTU_PER_KWH`

### Task 2 — Create `bill_savings.py`

**Goal:** New module for bill savings ratio using TARE lifetime fuel cost data.

**Functions to create:**

```python
def compute_bill_savings_ratio(
    df_tare: pd.DataFrame,
    mp: int,
    policy_scenario: str = 'iraRef',
    fuel_filter: str = 'Natural Gas',
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute per-building bill savings ratio from TARE lifetime fuel costs.

    ratio = retrofit_lifetime_fuel_cost / baseline_lifetime_fuel_cost

    Ratio < 1 = HP saves money; ratio > 1 = HP costs more.

    Unlike the retired analytical bill_impact_ratio (spark_gap × AFUE / COP),
    this uses actual per-building lifetime fuel costs from the TARE model that
    incorporate building-specific consumption, state fuel prices, and system
    efficiency.

    Args:
        df_tare: TARE model output DataFrame (from
            DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']).
        mp: Measure package number (e.g., 3 or 4).
        policy_scenario: 'iraRef' for IRA reference case or 'preIRA'
            for no-IRA scenario.
        fuel_filter: Baseline heating fuel to filter by (e.g., 'Natural Gas').
            Set to None to include all fuels.
        verbose: Print diagnostic info.

    Returns:
        DataFrame with columns: bldg_id (index), in.state, in.heating_fuel,
        weight, baseline_lifetime_fuel_cost, retrofit_lifetime_fuel_cost,
        bill_savings_ratio.

    Raises:
        KeyError: If required fuel cost columns are missing from df_tare.
        ValueError: If policy_scenario is not 'iraRef' or 'preIRA'.
    """


def aggregate_bill_savings_by_state(
    df_ratio: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    """Aggregate per-building bill savings ratios to state-level summary.

    Uses MEDIAN as the primary statistic (robust to outlier buildings).

    Args:
        df_ratio: Per-building DataFrame from compute_bill_savings_ratio().
        verbose: Print diagnostic info.

    Returns:
        DataFrame with columns: state, home_count,
        median_bill_savings_ratio, mean_bill_savings_ratio,
        total_baseline_cost, total_retrofit_cost, weighted_ratio.
    """
```

**Implementation details:**

1. Column name construction:
   - `baseline_col = 'baseline_heating_lifetime_fuel_cost'`
   - `retrofit_col = f'{policy_scenario}_mp{mp}_heating_lifetime_fuel_cost'`
2. Input validation: check both columns exist in `df_tare`
3. Filter to `fuel_filter` via `in.heating_fuel` column
4. Guard against zero/negative baseline cost: set ratio to NaN
5. State aggregation uses `in.state` column, weighted by `weight`
6. `weighted_ratio = total_retrofit_cost / total_baseline_cost` as cross-check
7. Round display columns: ratios to 3 decimals, costs to 2 decimals

### Task 3 — Update `__init__.py`

Add new module exports:
```python
from .demand import compute_scenario_demand, aggregate_demand_by_state
from .bill_savings import compute_bill_savings_ratio, aggregate_bill_savings_by_state
```

### Task 4 — Clean up postTARE notebook

**Target structure (≤ 15 cells):**
- Cell 1: Imports (from new modules — NOT from `kpi_functions`)
- Cell 2: Configuration (MP selection, batch mode)
- Cell 3: Load TARE model data (`DATAFRAMES_BY_MP`)
- Cell 4: Load EUSS data
- Cell 5: Bill savings ratio (loop over selected MPs)
- Cell 6: Bill savings ratio display (top/bottom 5 states per MP)
- Cell 7: Demand change — per-building
- Cell 8: Demand change — state aggregation + display
- Cell 9: Geospatial visualization (maps)
- Cell 10: Display results summary

**Import block (Cell 1) — corrected:**
```python
from cmu_tare_model.adoption_kpis import (
    load_euss_baseline, load_euss_upgrade, mp_to_upgrade,
    calculate_spark_gap,
    compute_scenario_demand, aggregate_demand_by_state,
    compute_bill_savings_ratio, aggregate_bill_savings_by_state,
)
from cmu_tare_model.adoption_kpis.visualize_geospatial_data import (
    prepare_state_geodataframe, create_choropleth_map,
)
from cmu_tare_model.adoption_kpis.data_loading import (
    FUEL_PRICES_PATH, SHAPEFILE_PATH,
)
from cmu_tare_model.utils.load_exported_results_to_df import load_measure_package_data
from cmu_tare_model.constants import VALID_MENU_MPS
```

**Bill savings ratio cell (Cell 5) — corrected, replaces placeholder:**
```python
bill_savings_results = {}
for mp in selected_mps:
    df_tare = DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']
    df_ratio = compute_bill_savings_ratio(
        df_tare, mp=mp, policy_scenario='iraRef',
        fuel_filter='Natural Gas', verbose=True,
    )
    df_ratio_state = aggregate_bill_savings_by_state(df_ratio, verbose=True)
    bill_savings_results[mp] = df_ratio_state
```

**Cleanup — delete from notebook:**
- All `compute_spark_gap_metrics()` references
- All `bill_impact_ratio` references
- All NPV computation cells and columns
- All `compute_thermal_cop_by_state()` calls (use `compute_thermal_cop` if needed)
- All `calculate_price_ratios()` calls (use `calculate_spark_gap`)
- Old placeholder bill savings code (commented-out blocks)
- Import of `kpi_functions` (dead module)
- References to `primary_mp` (use loop over `selected_mps` instead)
- References to `df_upgrade_primary` (use `upgrade_data[mp]` dict)
- County-level TODO comments
- Debug/scratch cells

### Task 5 — Delete `_kpi_functions_DEPRECATED.py`

After the postTARE notebook passes verification, delete the deprecated file.
All functions have been extracted to their dedicated modules.

## Reference Values (golden)

| Metric | Value | Tolerance |
|--------|-------|-----------|
| FL spark gap | 1.61 | ±0.05 |
| AK spark gap | 6.35 | ±0.05 |
| National mean spark gap | 3.15 | ±0.05 |
| Std ASHP COP (national, MP3) | 2.02 | ±0.02 |
| High-eff COP (national, MP4) | 3.34 | ±0.02 |
| Baseline AFUE (national) | 0.76 | ±0.02 |

Bill savings ratio reference values will be established by this session
(no prior golden values — the old code was placeholder).

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
| Using `bill_impact_ratio` formula | Retired metric; use `bill_savings_ratio` from TARE fuel costs |
| Computing bill savings from raw EUSS energy × prices | TARE already computed lifetime fuel costs; use those directly |
| Using mean instead of median for state aggregation | Median is robust to outlier buildings |
| Importing from `kpi_functions` | Module is deprecated; import from new dedicated modules |
| Using `primary_mp` single-MP pattern | Loop over `selected_mps` for all MP-dependent computations |
| Keeping NPV code in the notebook | Out of scope; strip all NPV references |
| Adding county-level TODO comments | County analysis is archived; remove TODO comments |

## Session Summary Template

After completing all tasks, produce a summary covering:
1. Files created (with line counts)
2. Files modified
3. Files deleted
4. New bill savings ratio reference values (to become golden for future sessions)
5. Confirm all package-level imports resolve
6. Confirm `kpi_functions` string appears nowhere in active imports
7. Confirm `bill_impact_ratio` string appears nowhere in active files
