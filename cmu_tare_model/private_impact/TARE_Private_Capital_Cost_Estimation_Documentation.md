# TARE Private Capital Cost Estimation Documentation

**TARE Version:** 2.1  
**Last Updated:** November 2025  
**Purpose:** Technical documentation for retrofit capital cost calculations and NPV analysis    
**Author:** Jordan M. Joseph, PhD (produced with assistance from Claude)

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Cost Data Spreadsheet](#cost-data-spreadsheet)
3. [Calculation Workflow](#calculation-workflow)
4. [Code Implementation](#code-implementation)
5. [Validation Framework](#validation-framework)
6. [Flow Diagram and Private Impact Model Components](#flow-diagram-and-private-impact-model-components)

---

## System Overview

The TARE capital cost system calculates the upfront costs and lifetime economic impacts of residential building retrofit measures. The system:

- **Loads cost data** from a structured Excel spreadsheet
- **Calculates installation costs** for retrofit equipment upgrades
- **Calculates replacement costs** for baseline equipment (counterfactual scenario)
- **Applies IRA rebates** when applicable based on household income
- **Computes net present value (NPV)** of retrofit investments considering lifetime fuel savings

### End-Use Categories Covered
- **Heating**: Heat pumps, furnaces (electric, gas, oil, propane)
- **Water Heating**: Heat pump water heaters, standard electric/gas/oil/propane tanks
- **Clothes Drying**: Heat pump dryers, standard electric/gas/propane dryers
- **Cooking**: Induction ranges, standard electric/gas/propane ranges
- **Enclosure Upgrades**: Attic insulation, air sealing, duct sealing, wall insulation, foundation insulation

### Percentile Structure for Cost Uncertainty

**Percentile Structure:** Each cost component has three values:
- `progressive` (TARE 10th percentile): Uses the REMDB 50th percentile
- `reference` (TARE 50th percentile): Uses the REMDB 90th percentile
- `conservative` (TARE 90th percentile): Reference + (Reference - Progressive)

---

## Cost Data Spreadsheet

**File:** `tare_retrofit_costs_cpi.xlsx`

### Sheet 1: CPI (Consumer Price Index)
**Purpose:** Adjusts historical cost data to 2023 dollars

| Column | Description |
|--------|-------------|
| `Year` | Years 2010-2023 |
| `Annual` | Annual CPI values |

**Example:** 2013 CPI ratio = 1.308 (2013 dollars × 1.308 = 2023 dollars)

---

### Sheet 2: Heating Costs
**Contains:** Space heating equipment capital costs

**Key Columns:**
- `action_measure`: "Replace" (existing) or "Install" (retrofit)
- `technology`: Technology type (e.g., "Electric ASHP", "Natural Gas Furnace")
- `efficiency`: Efficiency rating (e.g., "SEER 18, 9.3 HSPF", "95 AFUE")
- `unitCost_[percentile]`: Base equipment cost
- `cost_per_kBtuh_[percentile]`: Variable cost per kBtuh of heating capacity
- `otherCost_[percentile]`: Additional costs (e.g., ductwork for mini-splits)
- `lifetime`: Equipment lifetime in years

**Technologies Covered:**
- Electric ASHP (Air Source Heat Pump)
- Electric MSHP (Mini-Split Heat Pump)
- Electric Furnace
- Natural Gas Furnace
- Fuel Oil Furnace
- Propane Furnace

**Data Source:** Primarily NREL REMDB (2013 data, adjusted to 2023 dollars)

---

### Sheet 3: Water Heating Costs
**Contains:** Water heater capital costs

**Key Columns:**
- `technology`: Technology and capacity (e.g., "Electric Heat Pump Water Heater, 50 gal")
- `efficiency`: UEF (Unified Energy Factor) value
- `unitCost_[percentile]`: Base equipment cost
- `cost_per_gallon_[percentile]`: Variable cost per gallon of tank capacity
- `lifetime`: Equipment lifetime in years

**Technologies Covered:**
- Electric Heat Pump Water Heaters (50, 66, 80 gallon)
- Electric Standard Water Heaters
- Natural Gas Water Heaters
- Fuel Oil Water Heaters
- Propane Water Heaters

**Data Sources:** NREL REMDB (2013), Navigant Consulting (2018)

---

### Sheet 4: Clothes Drying Costs
**Contains:** Clothes dryer capital costs

**Key Columns:**
- `technology`: Dryer type (e.g., "Electric HP Clothes Dryer")
- `efficiency`: CEF (Combined Energy Factor) value
- `unitCost_[percentile]`: Equipment cost (no variable components)
- `lifetime`: Equipment lifetime in years

**Technologies Covered:**
- Electric Heat Pump Dryers
- Electric Standard Dryers
- Natural Gas Dryers
- Propane Dryers

**Data Sources:** NREL REMDB (2013), Redwood Energy (2021)

---

### Sheet 5: Cooking Costs
**Contains:** Cooking range capital costs

**Key Columns:**
- `technology`: Range type (e.g., "Electric Induction Range")
- `efficiency`: Efficiency factor
- `unitCost_[percentile]`: Equipment cost (no variable components)
- `lifetime`: Equipment lifetime in years

**Technologies Covered:**
- Electric Induction Ranges
- Electric Resistance Ranges
- Natural Gas Ranges
- Propane Ranges

**Note:** Induction ranges while more efficient (+10%), are much more expensive standard electric ranges

**Data Sources:** NREL REMDB (2013), Redwood Energy (2021)

---

### Sheet 6: Enclosure Upgrade Costs
**Contains:** Building envelope improvement costs

**Key Columns:**
- `technology`: Upgrade type (e.g., "Attic Floor Insulation: R-49")
- `existing_characteristic`: Current condition (e.g., "R-13")
- `retrofit_characteristic`: Target condition (e.g., "R-49")
- `normalized_cost_[percentile]`: Cost per square foot
- `cpi_ratio`: Inflation adjustment factor

**Technologies Covered:**
- Attic Floor Insulation (R-30, R-49, R-60)
- Air Leakage Reduction (30% reduction)
- Duct Sealing (10% leakage target)
- Drill-and-Fill Wall Insulation (R-13)
- Foundation Wall Insulation (R-10)
- Rim Joist Insulation (R-10)
- Crawlspace Sealing
- Finished Attic/Cathedral Ceiling Insulation (R-30)

**Cost Structure:** `normalized_cost` × `home_area` = total cost

**Data Source:** NREL REMDB (2013)

---

## Calculation Workflow

### Phase 1: Cost Data Loading
**Process:**
1. Load all 6 sheets from Excel file
2. Apply CPI adjustment: `cost × cpi_ratio × cost_multiplier`
3. Convert to dictionaries with `(technology, efficiency)` tuple keys

**Dictionary Structure:**
```python
{
    ('Electric ASHP', 'SEER 18, 9.3 HSPF'): {
        'unitCost_progressive': 4500.0,
        'unitCost_reference': 5000.0,
        'unitCost_conservative': 5500.0,
        'cost_per_kBtuh_progressive': 30.0,
        'cost_per_kBtuh_reference': 35.0,
        'cost_per_kBtuh_conservative': 40.0,
        # ... other cost components
    }
}
```

---

### Phase 2: Installation Costs
**Script:** `calculate_equipment_installation_costs.py`  
**Function:** `calculate_installation_cost(df, cost_dict, menu_mp, end_use)`

**Purpose:** Calculate the capital cost of installing retrofit equipment

**Process:**
1. **Match homes to technologies** based on upgrade specifications in DataFrame
2. **Sample costs probabilistically** from normal distributions using progressive/reference/conservative estimates
3. **Calculate total cost** using end-use-specific formulas

**Cost Formulas by End-Use:**

| End-Use | Formula |
|---------|---------|
| **Heating** | `unitCost + otherCost + (heating_load_kBtuh × cost_per_kBtuh) + installation_premium` |
| **Water Heating** | `unitCost + (tank_size_gallons × cost_per_gallon)` |
| **Clothes Drying** | `unitCost` only |
| **Cooking** | `unitCost` only |

**Heating Installation Premium:**
- No existing AC, has furnace/baseboard: +$400
- No existing AC, has boiler: +$1,500
- Has existing AC: $0

**Probabilistic Sampling:**
The system treats progressive/reference/conservative as the 10th/50th/90th percentiles of a normal distribution:
- Mean = reference value (50th percentile)
- Standard deviation = `(conservative - progressive) / (z₀.₉₀ - z₀.₁₀)`
- Each home gets a random draw from this distribution

---

### Phase 3: Replacement Costs
**Script:** `calculate_equipment_replacement_costs.py`  
**Function:** `calculate_replacement_cost(df, cost_dict, menu_mp, end_use)`

**Purpose:** Calculate the cost to replace existing equipment with like-for-like technology (counterfactual scenario)

**Process:**
1. **Match homes to baseline technologies** based on existing equipment specifications
2. **Sample costs probabilistically** using the same method as installation costs
3. **Calculate total cost** using same formulas as installation costs

**Key Insight:** Replacement costs represent the avoided cost that would have been incurred when equipment fails naturally. This is subtracted to calculate the incremental "net capital cost" of the retrofit.

---

### Phase 4: Enclosure Upgrade Costs
**Script:** `calculate_enclosure_upgrade_costs.py`  
**Function:** `calculate_enclosure_retrofit_upgradeCosts(df, menu_mp, cost_dict, retrofit_col, params_col)`

**Purpose:** Calculate the cost of building envelope improvements (only for Menu Packages 9 and 10)

**Process:**
1. **Match homes to upgrade paths** based on existing → target conditions
2. **Sample normalized costs** probabilistically ($/sq ft)
3. **Calculate total cost:** `normalized_cost × area`
4. **Sum all enclosure upgrades** for total weatherization cost

**Example Upgrade Paths:**
- Attic: R-13 → R-49
- Air Sealing: 30% reduction
- Ducts: Existing leakage → 10% leakage

---

### Phase 5: IRA Rebates
**Script:** `determine_rebate_eligibility_and_amount.py`  
**Functions:** `calculate_percent_AMI()`, `calculate_rebateIRA()`

**Purpose:** Calculate federal IRA (Inflation Reduction Act) rebates based on household income

**Rebate Tiers:**

| Household Income | Rebate Amount | Statutory Limits |
|-----------------|---------------|-----------------|
| ≤80% AMI | 100% of cost | Up to category maximum |
| 80-150% AMI | 50% of cost | Up to category maximum |
| >150% AMI | $0 | N/A |

**Category Maximums (per IRA legislation):**
- Heat pumps (HVAC): $8,000
- Heat pump water heaters: $1,750
- Heat pump clothes dryers: $840
- Electric cooking ranges: $840
- Weatherization: $1,600

**Note:** Rebates only apply in IRA policy scenarios, not in baseline, "Pre-IRA", or "No IRA" scenarios.

---

### Phase 6: Private NPV Calculation
**Script:** `calculate_lifetime_private_impact.py`  
**Function:** `calculate_private_npv()`

**Purpose:** Calculate the net present value of retrofit investments from the homeowner's perspective

**Capital Cost Calculation:**

```python
# With IRA rebates:
Total Capital Cost = Installation Cost - IRA Rebates + Weatherization Cost*
Net Capital Cost = Total Capital Cost - Replacement Cost

# Without IRA rebates:
Total Capital Cost = Installation Cost + Weatherization Cost*
Net Capital Cost = Total Capital Cost - Replacement Cost

# *Weatherization only for heating (MP9/MP10)
```

**Fuel Savings Calculation:**
For each year in equipment lifetime:
```python
annual_savings = (baseline_fuel_cost - retrofit_fuel_cost) × discount_factor
total_discounted_savings = sum(annual_savings for all years)
```

**NPV Calculations (Two Scenarios):**

| Scenario | Formula | Interpretation |
|----------|---------|----------------|
| **Less Willing-to-Pay** | `Total Fuel Savings - Total Capital Cost` | Early replacement of equipment (total capital cost) |
| **More Willing-to-Pay** | `Total Fuel Savings - Net Capital Cost` | Equipment replaced at end of life (incremental cost only) |

**Key Insight:**
- **Less WTP:** Assumes homeowner replaces equipment *before* it fails (bears full installation cost)
- **More WTP:** Assumes equipment replaced *at failure* (only pays incremental upgrade premium, since replacement was inevitable)

**Discounting:**
- Base year: 2024
- Method: Private fixed discount rate (currently assumed to be 7%)
- Discount factor calculated for each year: `1 / (1 + r)^t`

---

## Code Implementation

### Installation Costs (`calculate_equipment_installation_costs.py`)

**Main Functions:**

```python
calculate_installation_cost(df, cost_dict, menu_mp, end_use)
    """Calculate installation costs for retrofit equipment.
    
    Args:
        df: DataFrame with home characteristics
        cost_dict: Dictionary of (tech, efficiency) -> cost components
        menu_mp: Measure package ID (7, 8, 9, 10)
        end_use: Category ('heating', 'waterHeating', etc.)
    
    Returns:
        DataFrame with new cost column: mp{X}_{end_use}_installationCost
    """
```

**Helper Functions:**
- `get_end_use_installation_parameters()`: Retrieves technology-efficiency pairs and cost components for each end-use
- `calculate_installation_cost_per_row()`: Applies end-use-specific cost formulas
- `obtain_heating_system_specs()`: Extracts heating system specifications (load, efficiency)
- `calculate_heating_installation_premium()`: Calculates premium for homes without AC or with boilers

---

### Replacement Costs (`calculate_equipment_replacement_costs.py`)

**Main Functions:**

```python
calculate_replacement_cost(df, cost_dict, menu_mp, end_use)
    """Calculate replacement costs for baseline equipment.
    
    Uses same probabilistic sampling and formulas as installation costs,
    but matches to baseline/existing technologies instead of upgrades.
    
    Returns:
        DataFrame with new cost column: mp{X}_{end_use}_replacementCost
    """
```

**Helper Functions:**
- `get_end_use_replacement_parameters()`: Retrieves baseline technology-efficiency pairs
- `calculate_replacement_cost_per_row()`: Applies cost formulas for baseline equipment

---

### Enclosure Upgrades (`calculate_enclosure_upgrade_costs.py`)

**Main Functions:**

```python
calculate_enclosure_retrofit_upgradeCosts(df, menu_mp, cost_dict, retrofit_col, params_col)
    """Calculate enclosure upgrade costs.
    
    Args:
        retrofit_col: Output column name (e.g., 'insulation_atticFloor_upgradeCost')
        params_col: Area column to multiply by (e.g., 'area_attic_floor_sqft')
    
    Returns:
        DataFrame with calculated retrofit cost column
    """
```

**Helper Functions:**
- `get_enclosure_parameters()`: Defines upgrade paths (existing → target conditions)

**Supported Upgrades:**
- `insulation_atticFloor_upgradeCost`
- `infiltration_reduction_upgradeCost`
- `duct_sealing_upgradeCost`
- `insulation_wall_upgradeCost`
- `insulation_foundation_wall_upgradeCost`
- `insulation_rim_joist_upgradeCost`
- `seal_crawlspace_upgradeCost`
- `insulation_roof_upgradeCost`

---

### Private NPV (`calculate_lifetime_private_impact.py`)

**Main Functions:**

```python
calculate_private_npv(df, df_fuel_costs, df_baseline_costs, input_mp, menu_mp, 
                      policy_scenario, discounting_method='private_fixed', 
                      base_year=2024)
    """Calculate private NPV for all equipment categories.
    
    Args:
        df: Main DataFrame with installation/replacement costs
        df_fuel_costs: Measure package fuel costs by year
        df_baseline_costs: Baseline fuel costs by year
        policy_scenario: 'No Inflation Reduction Act' or 'AEO2023 Reference Case'
    
    Returns:
        DataFrame with NPV columns for each category:
        - {prefix}_{category}_total_capitalCost
        - {prefix}_{category}_net_capitalCost
        - {prefix}_{category}_private_npv_lessWTP
        - {prefix}_{category}_private_npv_moreWTP
    """
```

**Helper Functions:**
- `calculate_capital_costs()`: Calculates total and net capital costs with/without IRA rebates
- `calculate_and_update_npv()`: Computes discounted fuel savings and NPV values

---

## Validation Framework

All calculation functions implement a consistent validation framework to ensure data quality and proper handling of invalid/missing data.

### Five-Step Validation Process

**1. Mask Initialization**
Identifies valid homes using `include_{category}` flags (e.g., `include_heating`, `include_waterHeating`)

```python
valid_mask = df['include_heating'] == True
```

**2. Series Initialization**
Creates result series with zeros for valid homes, NaN for invalid homes

```python
result_series = pd.Series(np.nan, index=df.index)
result_series.loc[valid_mask] = 0.0
```

**3. Valid-Only Calculation**
Performs calculations only where:
- Home is flagged as eligible (`valid_mask`)
- Technology can be matched (not 'unknown')
- Required data exists (e.g., fuel type, efficiency)

**4. Valid-Only Updates**
Updates result series only for homes meeting all criteria

```python
result_series.loc[valid_calculation_indices] = calculated_values
```

**5. Final Masking**
Ensures all output columns have NaN for excluded homes

```python
df_result = apply_final_masking(df, all_columns_to_mask, verbose=True)
```

### Validation Flags

**Inclusion Flags** (overall eligibility):
- `include_heating`
- `include_waterHeating`
- `include_clothesDrying`
- `include_cooking`

**Technology Matching** (set during calculations):
- `valid_tech_{category}`: Technology can be matched to cost data
- `valid_fuel_{category}`: Fuel type is compatible

---

## Flow Diagram and Private Impact Model Components

### 1. Probabilistic Cost Modeling
**Purpose:** Capture realistic cost variation across homes

The progressive/reference/conservative structure represents:
- **10th percentile (progressive):** Lower cost scenario (favorable conditions, bulk pricing, simple install)
- **50th percentile (reference):** Typical cost scenario (average conditions)
- **90th percentile (conservative):** Higher cost scenario (challenging install, premium equipment)

**Implementation:** Normal distribution with:
- μ = reference value
- σ = (conservative - progressive) / 2.563 (z-score range for 10th-90th percentile)

**Benefit:** Produces realistic variation in aggregate results while maintaining deterministic individual calculations (when run with same random seed).

---

### 2. CPI Adjustment
**Purpose:** Convert historical cost data to current dollars

**Formula:** `adjusted_cost = historical_cost × (CPI_current / CPI_historical)`

**Example:**
- 2013 equipment cost: $5,000
- 2013 CPI: 233.0
- 2023 CPI: 304.7
- CPI ratio: 304.7 / 233.0 = 1.308
- 2023 equivalent: $5,000 × 1.308 = $6,540

**Note:** Current REMDB data is already in 2023 dollars, so CPI adjustment may not be necessary for base data (but useful for mixed-vintage sources).

---

### 3. Variable Cost Components
**Purpose:** Capture cost heterogeneity across home sizes and characteristics

**Examples:**
- **Heating:** Larger homes need larger/more expensive heat pumps (cost_per_kBtuh)
- **Water Heating:** Larger households need bigger tanks (cost_per_gallon)
- **Enclosure:** Larger areas cost more to insulate (normalized_cost × area)

**Benefit:** More accurate costs than flat per-unit pricing.

---

### 4. Separate Rebate Logic
**Purpose:** Keep cost data independent of policy scenarios

**Design:** Rebates are calculated separately and subtracted from capital costs in NPV phase, not embedded in installation cost data.

**Benefit:**
- Cost data remains policy-neutral
- Easy to model different rebate scenarios
- Clear accounting of gross vs. net costs

---

### 5. Net vs. Total Capital Cost
**Purpose:** Distinguish between early replacement and end-of-life replacement

**Total Capital Cost:**
```
Installation Cost (- IRA Rebates if applicable) + Weatherization
```
Used for "Less Willing-to-Pay" NPV calculation (early replacement scenario)

**Net Capital Cost:**
```
Total Capital Cost - Replacement Cost
```
Used for "More Willing-to-Pay" NPV calculation (end-of-life replacement scenario)

**Economic Interpretation:**
- **Total Cost:** Full upfront investment required
- **Net Cost:** Incremental cost premium vs. like-for-like replacement
- **Replacement Cost:** Sunk cost that would be incurred anyway at equipment failure

---

### 6. Data Validation
**Purpose:** Prevent errors from propagating through calculations

**Strategy:**
- Flag ineligible homes early (inclusion criteria)
- Validate technology matching before cost calculation
- Use NaN (not zero) for invalid/missing data
- Apply consistent masking throughout pipeline

**Benefit:**
- Clear distinction between "zero cost" and "not applicable"
- Easier debugging and data quality checks
- Prevents contamination of aggregate statistics

---

## Data Flow Diagram

```
Excel Spreadsheet (tare_retrofit_costs_cpi.xlsx)
              ↓
    Load & Apply CPI Adjustment
              ↓
Convert to (technology, efficiency) dictionaries
              ↓
┌─────────────┴──────────────┐
│                            │
↓                            ↓
Installation Costs    Replacement Costs
(retrofit equipment)  (baseline equipment)
│                            │
└─────────────┬──────────────┘
              ↓
     IRA Rebates (if applicable)
              ↓
     Calculate Capital Costs:
     - Total Capital = Installation - Rebates (+ weatherization)
     - Net Capital = Total Capital - Replacement
              ↓
     Calculate Private NPV:
     - NPV (less WTP) = Fuel Savings - Total Capital
     - NPV (more WTP) = Fuel Savings - Net Capital
```

---

## Update History

**March 24, 2025:** Removed RSMeans CCI adjustments  
**April 9, 2025:** Improved documentation  
**April 21, 2025:** Refactored cost utility functions (consolidated into `calculation_utils`)  
**April 29, 2025:** Added data validation checks and error handling  
**November 2025:** Comprehensive documentation update

---

## Notes for REMDB v4 Update

When updating to REMDB v4, pay attention to:

1. **Cost structure changes:** Verify that progressive/reference/conservative percentile approach is still valid
2. **Technology coverage:** Check for new technologies or deprecated equipment types
3. **Cost equations:** Confirm variable cost components (per kBtuh, per gallon) are still appropriate
4. **Efficiency metrics:** Ensure UEF, CEF, AFUE, SEER, HSPF definitions haven't changed
5. **Data year:** Update CPI adjustments if REMDB v4 uses a different base year
6. **Validation:** Compare outputs between v3 and v4 to identify significant cost changes

---

**End of Documentation**
