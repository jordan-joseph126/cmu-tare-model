# REMDB v4 Capital Cost Calculation System Documentation


This document describes the **current implementation** of the REMDB v4 capital cost calculation system as of January 2026. The system calculates equipment replacement and upgrade costs using REMDB v4's deterministic regression methodology.

### What the System Does

The system calculates two types of installed costs for residential equipment retrofits:

1. **Replacement Costs**: Cost to replace existing equipment with like-for-like technology (counterfactual baseline)
2. **Upgrade Costs**: Cost to retrofit to improved technology (e.g., gas furnace → heat pump)

### Current Implementation Scope

| End-Use | Status | Notes |
|---------|--------|-------|
| **Heating** | Implemented | Furnaces, baseboard, heat pumps |
| **Cooling** | Implemented | Central AC, room AC, heat pumps |
| Water Heating | Structured | Code present but commented out |
| Clothes Drying | Structured | Code present but commented out |
| Cooking | Structured | Code present but commented out |

### Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `validation_framework.py` | 504 | Core 5-step validation framework |
| `remdb_v4_installed_cost_utils.py` | 506 | Metric extraction and REMDB parameter mapping |
| `calculation_utils.py` | 385 | Specialized calculation utilities |
| `calculate_equipment_replacement_costs.py` | 248 | Replacement cost calculations |
| `calculate_equipment_installation_costs.py` | 244 | Upgrade cost calculations |

### Cost Formula (REMDB v4)

```
Material_Price = (pm1 × pm1_coef) + (pm2 × pm2_coef) + intercept
Installed_Cost = (Material_Price × multiplier) + adder
```

All costs are in **2023 dollars** (no CPI adjustment required).

---

## Changes from REMDB v3 to REMDB v4

This section summarizes the key methodological and architectural changes from the previous implementation.

### Methodology Changes

| Aspect | REMDB v3 (Previous) | REMDB v4 (Current) |
|--------|---------------------|---------------------|
| **Cost Calculation** | Probabilistic sampling from normal distributions | Deterministic regression formula |
| **Cost Parameters** | `unitCost`, `cost_per_kBtuh`, `otherCost` | `pm1_coef`, `pm2_coef`, `intercept`, `multiplier`, `adder` |
| **Percentiles** | progressive/reference/conservative (10th/50th/90th) sampled stochastically | low/mid/high coefficients applied deterministically |
| **Dollar Year** | 2013$ adjusted via CPI ratios | Already in 2023$ (no adjustment needed) |
| **Data Format** | Excel workbook with multiple sheets per end-use | Single CSV file with all equipment types |
| **Lookup Method** | Dictionary with `(technology, efficiency)` tuple keys | DataFrame lookup by `row_id` |

### Cost Formula Comparison

**REMDB v3 (Probabilistic):**
```python
# Sample from normal distribution
mean = reference_value
std = (conservative - progressive) / 2.563  # z-score for 10th-90th percentile
sampled_cost = np.random.normal(mean, std)

# End-use specific formulas
heating_cost = unitCost + otherCost + (heating_load_kBtuh × cost_per_kBtuh) + installation_premium
water_heating_cost = unitCost + (tank_size_gallons × cost_per_gallon)
```

**REMDB v4 (Deterministic Regression):**
```python
# Single formula for all end-uses
material_price = (pm1 × pm1_coef) + (pm2 × pm2_coef) + intercept
installed_cost = (material_price × multiplier) + adder
```

### Data Structure Changes

**REMDB v3 Data File:** `tare_retrofit_costs_cpi.xlsx`
- Sheet 1: CPI adjustment factors
- Sheet 2: Heating costs
- Sheet 3: Water heating costs
- Sheet 4: Clothes drying costs
- Sheet 5: Cooking costs
- Sheet 6: Enclosure costs

**REMDB v4 Data File:** `remdb_v4_tare_retrofit_costs.csv`
- Single table with 32 rows (all equipment types)
- 26 columns including coefficients, bounds, and multipliers
- Indexed by `row_id` (e.g., `furnaces_gas_furnace`, `air_source_heat_pump_centrally_ducted`)

### Key Improvements in v4

1. **Reproducibility**: Deterministic calculations produce identical results each run (no random sampling)

2. **Simplified Code**: Single regression formula replaces multiple end-use-specific cost equations

3. **Data-Driven Units**: Unit conversions read from REMDB columns (`pm1_unit`, `pm2_metric`) rather than hardcoded

4. **Bounds Handling**: REMDB provides explicit `pm1_lower_bound`/`pm1_upper_bound` for filling missing values

5. **No CPI Adjustment**: Costs already in 2023 dollars eliminates inflation adjustment step

### What Stayed the Same

- **5-step validation framework**: Same pattern for data quality control
- **Replacement vs Upgrade distinction**: Counterfactual baseline costs still calculated separately
- **End-use categories**: Same equipment categories (heating, cooling, water heating, clothes drying, cooking)
- **Percentile options**: Still support low/mid/high cost estimates (now deterministic instead of stochastic)

---

## Architecture Overview

### File Organization

```
cmu_tare_model/
├── utils/
│   ├── validation_framework.py          (504 lines)
│   ├── calculation_utils.py             (385 lines)
│   └── remdb_v4_installed_cost_utils.py (506 lines)
│
├── costs/
│   ├── calculate_equipment_replacement_costs.py (248 lines)
│   └── calculate_equipment_installation_costs.py (244 lines)
│
└── data/retrofit_costs/
    └── remdb_v4_tare_retrofit_costs.csv (32 rows, 26 columns)
```

### Function Organization

The implementation distinguishes between **public API functions** and **internal helper functions** using Python's underscore convention:

**Main Functions:**
- `load_remdb_v4_data()` - Load REMDB v4 database
- `add_remdb_replacement_metrics()` - Prepare replacement metrics from baseline equipment
- `add_remdb_upgrade_metrics()` - Prepare upgrade metrics from upgrade equipment specs
- `calculate_replacement_installed_cost()` - Calculate replacement costs
- `calculate_upgrade_installed_cost()` - Calculate upgrade costs

**Helper Functions (Internal, underscore-prefixed):**
- `_assign_replacement_row_id()` - Map baseline equipment to REMDB row_id
- `_assign_upgrade_row_id()` - Map upgrade equipment to REMDB row_id
- `_map_remdb_parameters()` - Map REMDB coefficients to DataFrame columns
- `_convert_pm1()` - Convert capacity to REMDB units (data-driven)
- `_convert_pm2()` - Convert efficiency to REMDB units (data-driven)
- `_fill_missing_from_bounds()` - Fill missing values from REMDB bounds

### Two-Step Workflow

Both replacement and upgrade costs follow a consistent two-step pattern:

```python
# STEP 1: Prepare metrics (row_id assignment, unit conversion, bounds filling)
df = add_remdb_replacement_metrics(df, remdb_v4_costs, 'heating', 'mid')

# STEP 2: Calculate costs (regression formula with validation)
df = calculate_replacement_installed_cost(df, menu_mp=0, end_use='heating', percentile='mid')
```

---

## Module Details

### remdb_v4_installed_cost_utils.py (506 lines)

Prepares equipment metrics for REMDB v4 cost calculations using data-driven unit conversions.

#### Data Loading

```python
def load_remdb_v4_data(
    data_dir: Optional[str] = None,
    filename: str = "remdb_v4_tare_retrofit_costs.csv"
) -> pd.DataFrame:
    """Load REMDB v4 retrofit cost data, indexed by row_id."""
```

#### Main Functions

**`add_remdb_replacement_metrics(df, remdb_v4_costs, end_use, percentile='mid')`**

Prepares REPLACEMENT metrics from BASELINE equipment. Performs four steps:
1. Assigns `row_id` based on equipment type
2. Maps REMDB coefficients and unit specifications
3. Extracts metrics with correct unit conversions
4. Fills missing values from REMDB bounds

**Output Columns Created:**
| Column Pattern | Example | Description |
|----------------|---------|-------------|
| `row_id_{end_use}_replacement` | `row_id_heating_replacement` | REMDB row identifier |
| `euss_{end_use}_replacement_pm1` | `euss_heating_replacement_pm1` | Performance metric 1 (converted units) |
| `euss_{end_use}_replacement_pm2` | `euss_heating_replacement_pm2` | Performance metric 2 (converted units) |
| `{end_use}_replacement_pm1_coef_{percentile}` | `heating_replacement_pm1_coef_mid` | PM1 coefficient |
| `{end_use}_replacement_pm2_coef_{percentile}` | `heating_replacement_pm2_coef_mid` | PM2 coefficient |
| `{end_use}_replacement_intercept_{percentile}` | `heating_replacement_intercept_mid` | Intercept value |
| `{end_use}_replacement_multiplier_retrofit` | `heating_replacement_multiplier_retrofit` | Installation multiplier |
| `{end_use}_replacement_adder_retrofit` | `heating_replacement_adder_retrofit` | Installation adder |

> **Note:** The source code's docstring incorrectly lists column names without the `euss_` prefix and uses `_replace` instead of `_replacement` in some places. The table above reflects the **actual** column names created by the code.

**`add_remdb_upgrade_metrics(df, remdb_v4_costs, end_use, percentile='mid')`**

Same structure as replacement metrics but:
- Uses `_assign_upgrade_row_id()` instead of `_assign_replacement_row_id()`
- Reads from upgrade equipment specification columns
- Output column pattern: `euss_{end_use}_upgrade_pm1`, etc.

#### Helper Functions

**`_assign_replacement_row_id(df, end_use)`**

Maps baseline equipment to REMDB row_id using `np.select()`:

**Heating row_id mapping:**
| Baseline Equipment | row_id |
|-------------------|--------|
| Propane fuel | `furnaces_gas_furnace` |
| Fuel Oil | `furnaces_gas_furnace` |
| Natural Gas | `furnaces_gas_furnace` |
| Electric (non-ASHP) | `electric_baseboard_default` |
| ASHP with ducts | `air_source_heat_pump_centrally_ducted` |
| ASHP without ducts | `air_source_heat_pump_non_ducted_multi_zone` |

**Cooling row_id mapping:**
| Baseline Equipment | row_id |
|-------------------|--------|
| Room AC | `air_conditioner_room_ac_window_or_through_wall` |
| Central AC | `air_conditioner_centrally_ducted` |
| Heat Pump with ducts | `air_source_heat_pump_centrally_ducted` |
| Heat Pump without ducts | `air_source_heat_pump_non_ducted_multi_zone` |

**`_assign_upgrade_row_id(df, end_use)`**

Maps upgrade equipment (all heat pumps) based on duct status:
- Ducted homes → `air_source_heat_pump_centrally_ducted`
- Non-ducted homes → `air_source_heat_pump_non_ducted_multi_zone`

**`_convert_pm1(df, capacity_col, pm1_unit_col)` → pd.Series**

Data-driven capacity conversion based on `pm1_unit` column from REMDB:

| pm1_unit Value | Conversion | Equipment Types |
|----------------|------------|-----------------|
| `"Tons"` | kBtu/h ÷ 12 | Heat pumps, Central ACs |
| `"BTU/hr"` | kBtu/h × 1000 | Furnaces, Boilers, Baseboard |

**`_convert_pm2(df, efficiency_col, pm2_metric_col)` → pd.Series**

Data-driven efficiency conversion based on `pm2_metric` column from REMDB:

| pm2_metric Value | Source Format | Conversion | Output Range |
|------------------|---------------|------------|--------------|
| `"SEER1"` | "SEER 15" | Extract numeric | 13-30 |
| `"AFUE"` | "80% AFUE" | Extract ÷ 100 | 0.60-0.97 |
| `"CEER"` | "EER 10.7" | Extract (no direct match) | 9-15 |
| Empty/NaN | N/A | Set to 0 | 0 |

> **Important:** The AFUE divide-by-100 conversion is critical. REMDB expects AFUE as a decimal (0.80), not a percentage (80).

**`_fill_missing_from_bounds(df, remdb_v4_costs, end_use, replacement_or_upgrade, pm1_col, pm2_col)`**

Fills missing pm1/pm2 values using midpoint of REMDB bounds:
```python
filled_value = (pm1_lower_bound + pm1_upper_bound) / 2.0
```

---

### calculate_equipment_replacement_costs.py (248 lines)

Calculates REPLACEMENT installed costs using REMDB v4 regression.

#### Function Signature

```python
def calculate_replacement_installed_cost(
    df: pd.DataFrame,
    menu_mp: int,
    end_use: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """
    Calculate REPLACEMENT installed costs using REMDB v4 regression formula.
    
    PREREQUISITE: Call add_remdb_replacement_metrics() first.
    
    Args:
        df: DataFrame with prepared metrics.
        menu_mp: Measure package number (typically 0 for baseline).
        end_use: Equipment category ('heating' or 'cooling').
        percentile: Cost percentile ('low', 'mid', 'high').
        
    Returns:
        DataFrame with new column: mp{menu_mp}_{end_use}_replacement_installed_cost_{percentile}
    """
```

#### Required Input Columns

Must exist before calling (created by `add_remdb_replacement_metrics()`):
- `euss_{end_use}_replacement_pm1`
- `euss_{end_use}_replacement_pm2`
- `{end_use}_replacement_pm1_coef_{percentile}`
- `{end_use}_replacement_pm2_coef_{percentile}`
- `{end_use}_replacement_intercept_{percentile}`
- `{end_use}_replacement_multiplier_retrofit`
- `{end_use}_replacement_adder_retrofit`

#### Output Column

Pattern: `mp{menu_mp}_{end_use}_replacement_installed_cost_{percentile}`

Example: `mp0_heating_replacement_installed_cost_mid`

---

### calculate_equipment_installation_costs.py (244 lines)

Calculates UPGRADE installed costs. Structurally identical to replacement costs.

#### Function Signature

```python
def calculate_upgrade_installed_cost(
    df: pd.DataFrame,
    menu_mp: int,
    end_use: str,
    percentile: str = 'mid'
) -> pd.DataFrame:
    """
    Calculate UPGRADE installed costs using REMDB v4 regression formula.
    
    PREREQUISITE: Call add_remdb_upgrade_metrics() first.
    """
```

#### Output Column

Pattern: `mp{menu_mp}_{end_use}_upgrade_installed_cost_{percentile}`

Example: `mp7_heating_upgrade_installed_cost_mid`

---

### validation_framework.py (504 lines)

Implements the 5-step validation framework ensuring data quality.

#### Five-Step Framework

1. **Mask Initialization**: Determine which homes have valid data
2. **Series Initialization**: Initialize with zeros for valid homes, NaN for others
3. **Valid-Only Calculation**: Calculate only for valid homes
4. **Valid-Only Updates**: Update only valid homes
5. **Final Masking**: Apply consistent masking to all result columns

#### Key Functions

**`initialize_validation_tracking(df, category, menu_mp, verbose=True)`**

Returns tuple: `(df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask)`

**`get_valid_fuel_types(category)`**

Returns valid fuel types per category:

| Category | Valid Fuel Types |
|----------|------------------|
| heating | Electricity, Natural Gas, Propane, Fuel Oil |
| cooling | Electricity |
| waterHeating | Electricity, Natural Gas, Propane, Fuel Oil |
| clothesDrying | Electricity, Natural Gas, Propane |
| cooking | Natural Gas, Propane |

**`get_valid_calculation_mask(df, category, menu_mp, verbose=True)`**

Combines data validation with retrofit status:
- Baseline (menu_mp=0): Uses only data validation
- Measure packages: Combines data validation AND retrofit status

**`create_retrofit_only_series(df, retrofit_mask, ...)`**

Initializes Series with zeros for retrofitted homes, NaN for others.

**`apply_final_masking(df, all_columns_to_mask, verbose=True)`**

Ensures all tracked columns respect validation masks.

---

### calculation_utils.py (385 lines)

Specialized utilities for calculations.

#### Key Functions

- `get_all_possible_fuel_columns(category)` - Returns consumption column names
- `get_post_retrofit_columns(category, menu_mp)` - Returns post-retrofit column names
- `identify_valid_homes(df)` - Creates data quality flags (`include_{category}`, etc.)
- `mask_invalid_data(df, menu_mp=None)` - Sets invalid consumption to NaN
- `validate_common_parameters(menu_mp, policy_scenario, discounting_method)` - Input validation

---

## Data-Driven Unit Conversions

A key architectural feature is that unit conversions are **driven by REMDB column values**, not hardcoded logic.

### How It Works

1. `_map_remdb_parameters()` copies `pm1_unit` and `pm2_metric` from REMDB to the DataFrame
2. `_convert_pm1()` reads the `pm1_unit` column to determine conversion
3. `_convert_pm2()` reads the `pm2_metric` column to determine conversion

### Benefits

- **Self-documenting**: REMDB defines expected units
- **Automatically adapts**: New equipment types work without code changes
- **Single source of truth**: No hardcoded assumptions scattered in code

---

## Validation Framework Integration

Both cost calculation functions follow the identical 5-step pattern:

```python
def calculate_xxx_installed_cost(df, menu_mp, end_use, percentile='mid'):
    # ===== STEP 1: Initialize validation tracking =====
    df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = \
        initialize_validation_tracking(df, end_use, menu_mp=menu_mp, verbose=True)
    
    # ===== STEP 2: Initialize result series =====
    result_series = create_retrofit_only_series(df_copy, valid_mask)

    # ===== STEP 3 & 4: Calculate for valid homes only =====
    pm1 = df_copy[f'euss_{end_use}_xxx_pm1']
    pm2 = df_copy[f'euss_{end_use}_xxx_pm2']
    # ... get coefficients ...
    
    material_price = (pm1 * pm1_coef) + (pm2 * pm2_coef) + intercept
    installed_cost = (material_price * multiplier) + adder
    
    result_series.loc[valid_mask] = installed_cost.loc[valid_mask].round(2)
    
    # Track and apply columns
    df_new_columns = pd.DataFrame({cost_col: result_series})
    df_copy, all_columns_to_mask = apply_new_columns_to_dataframe(
        df_copy, df_new_columns, end_use, category_columns_to_mask, all_columns_to_mask)
    
    # ===== STEP 5: Final masking =====
    df_copy = apply_final_masking(df_copy, all_columns_to_mask, verbose=True)
    
    return df_copy
```

---

## Complete Usage Example

```python
from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
    load_remdb_v4_data,
    add_remdb_replacement_metrics,
    add_remdb_upgrade_metrics
)
from cmu_tare_model.costs.calculate_equipment_replacement_costs import (
    calculate_replacement_installed_cost
)
from cmu_tare_model.costs.calculate_equipment_installation_costs import (
    calculate_upgrade_installed_cost
)
from cmu_tare_model.utils.calculation_utils import identify_valid_homes

# Load REMDB v4 data
remdb_v4_costs = load_remdb_v4_data()

# Identify valid homes (creates include_heating, include_cooling, etc.)
df = identify_valid_homes(df)

# ========================================
# REPLACEMENT COSTS (Counterfactual)
# ========================================
for end_use in ['heating', 'cooling']:
    # Step 1: Prepare metrics from baseline equipment
    df = add_remdb_replacement_metrics(df, remdb_v4_costs, end_use, 'mid')
    
    # Step 2: Calculate costs
    df = calculate_replacement_installed_cost(df, menu_mp=0, end_use=end_use, percentile='mid')
    # Output: mp0_{end_use}_replacement_installed_cost_mid

# ========================================
# UPGRADE COSTS (Retrofit Scenario)
# ========================================
menu_mp = 7  # Standard heat pump package

for end_use in ['heating', 'cooling']:
    # Step 1: Prepare metrics from upgrade equipment specs
    df = add_remdb_upgrade_metrics(df, remdb_v4_costs, end_use, 'mid')
    
    # Step 2: Calculate costs
    df = calculate_upgrade_installed_cost(df, menu_mp=menu_mp, end_use=end_use, percentile='mid')
    # Output: mp7_{end_use}_upgrade_installed_cost_mid
```

---

## Verification

### Check Unit Conversions

```python
# Verify pm1 (capacity) conversions
pm1 = df['euss_heating_replacement_pm1']
pm1_unit = df['heating_replacement_pm1_unit']

# Heat pumps should be in Tons (1.5-5.0 typical range)
hp_mask = pm1_unit == 'Tons'
print(f"Heat pump capacity (Tons): min={pm1[hp_mask].min():.1f}, max={pm1[hp_mask].max():.1f}")

# Furnaces should be in BTU/hr (30,000-150,000 typical range)
furnace_mask = pm1_unit.str.lower() == 'btu/hr'
print(f"Furnace capacity (BTU/hr): min={pm1[furnace_mask].min():,.0f}, max={pm1[furnace_mask].max():,.0f}")
```

### Check Efficiency Conversions

```python
# Verify AFUE is divided by 100 (should be 0.60-0.97, NOT 60-97)
pm2 = df['euss_heating_replacement_pm2']
pm2_metric = df['heating_replacement_pm2_metric']

afue_mask = pm2_metric == 'AFUE'
afue_values = pm2[afue_mask].dropna()
print(f"AFUE range: {afue_values.min():.2f} to {afue_values.max():.2f}")
assert afue_values.max() < 1.5, "AFUE should be < 1.5 (decimal form)"
```

### Check Cost Reasonableness

```python
cost_col = 'mp0_heating_replacement_installed_cost_mid'
print(f"Valid costs: {df[cost_col].notna().sum():,}")
print(f"Mean: ${df[cost_col].mean():,.0f}")
print(f"Range: ${df[cost_col].min():,.0f} - ${df[cost_col].max():,.0f}")
```

---

## Known Limitations

1. **Only heating and cooling implemented**: Water heating, clothes drying, and cooking code is structured but commented out pending testing.

2. **CEER/EER mismatch**: Room AC efficiency uses CEER in REMDB but EUSS provides EER. No direct conversion exists, so these values may be NaN → filled from bounds.

3. **Single-zone mini-split logic commented out**: All non-ducted homes currently get multi-zone pricing. Future enhancement could use square footage threshold.

4. **Source docstring errors**: The docstrings in `add_remdb_replacement_metrics()` list incorrect column names. The actual column names are documented in this file.

---

## REMDB v4 Database Reference

### Database Structure

**File:** `remdb_v4_tare_retrofit_costs.csv`  
**Index:** `row_id` column

### Key Columns

| Column | Description |
|--------|-------------|
| `row_id` | Equipment identifier (e.g., `furnaces_gas_furnace`) |
| `tare_category` | Category: heating, cooling, waterHeating, clothesDrying, cooking, enclosure |
| `pm1_metric`, `pm1_unit` | Performance metric 1 name and expected unit |
| `pm2_metric`, `pm2_unit` | Performance metric 2 name and expected unit |
| `pm1_coef_low/mid/high` | PM1 regression coefficients |
| `pm2_coef_low/mid/high` | PM2 regression coefficients |
| `intercept_low/mid/high` | Intercept values |
| `multiplier_retrofit` | Installation cost multiplier |
| `adder_retrofit` | Installation cost adder |
| `pm1_lower_bound`, `pm1_upper_bound` | Valid PM1 range |
| `pm2_lower_bound`, `pm2_upper_bound` | Valid PM2 range |
| `lifetime_years` | Equipment lifetime |

### Heating/Cooling Row IDs

| row_id | Category | Description |
|--------|----------|-------------|
| `furnaces_gas_furnace` | heating | Gas/propane/oil furnaces |
| `electric_baseboard_default` | heating | Electric resistance |
| `air_source_heat_pump_centrally_ducted` | heating/cooling | Ducted ASHP |
| `air_source_heat_pump_non_ducted_multi_zone` | heating/cooling | Ductless mini-split |
| `air_conditioner_centrally_ducted` | cooling | Central AC |
| `air_conditioner_room_ac_window_or_through_wall` | cooling | Window/room AC |

---

## Column Naming Convention Summary

| Type | Pattern | Example |
|------|---------|---------|
| Row ID | `row_id_{end_use}_{replacement\|upgrade}` | `row_id_heating_replacement` |
| PM values | `euss_{end_use}_{replacement\|upgrade}_pm1` | `euss_heating_replacement_pm1` |
| Coefficients | `{end_use}_{replacement\|upgrade}_pm1_coef_{percentile}` | `heating_replacement_pm1_coef_mid` |
| Final cost | `mp{menu_mp}_{end_use}_{replacement\|upgrade}_installed_cost_{percentile}` | `mp0_heating_replacement_installed_cost_mid` |

---

**Last Updated:** December 15, 2025
