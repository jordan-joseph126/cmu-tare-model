# CMU TARE Model: Migration Guide to NREL REMDB 2023
**Updating Capital Cost Calculations to Use Regression-Based Pricing**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Understanding the Changes](#understanding-the-changes)
3. [Step-by-Step Migration Process](#step-by-step-migration-process)
4. [Detailed Implementation Instructions](#detailed-implementation-instructions)
5. [Testing and Validation](#testing-and-validation)
6. [Migration Timeline](#migration-timeline)
7. [Troubleshooting Guide](#troubleshooting-guide)

---

## Executive Summary

### What is Changing?

The CMU TARE Model currently uses **fixed cost estimates** from the 2013 NREL REMDB database. We are updating to the **2023 NREL REMDB**, which uses **regression-based pricing** that calculates costs dynamically based on equipment performance characteristics.

### Why This Matters

The new system provides:
- **More accurate costs** based on actual equipment specifications
- **Current 2023 pricing** without needing inflation adjustments
- **Better uncertainty quantification** with probabilistic cost distributions
- **Separation of material and labor costs** for better transparency

### Key Impact

Instead of looking up a single fixed cost for "Electric Heat Pump", the model will now **calculate** the cost based on:
- Equipment capacity (e.g., 3 tons)
- Efficiency rating (e.g., SEER 18)
- Installation scenario (new construction vs retrofit)
- Statistical distribution (10th, 50th, 90th percentile)

---

## Understanding the Changes

### Old System: Fixed Cost Lookup

**How it worked:**
1. Look up equipment type (e.g., "Electric ASHP, SEER 18")
2. Get three fixed costs: Progressive, Reference, Conservative
3. Randomly sample from normal distribution
4. Apply CPI adjustment from 2013 to 2023 dollars

**Example:**
```
Electric ASHP, SEER 18, 9.3 HSPF:
- Base cost: $4,800
- Cost per kBtuh: $55
- Other costs: $0
- Total for 3-ton unit (36 kBtuh): $4,800 + (36 × $55) = $6,780
- Adjusted to 2023$: $6,780 × 1.308 = $8,868
```

### New System: Regression-Based Calculation

**How it works:**
1. Extract equipment performance metrics (capacity, efficiency)
2. Calculate material cost using regression equation
3. Add installation cost (either multiplier or fixed adder)
4. Already in 2023 dollars (no adjustment needed)

**Example:**
```
Electric ASHP with 3 tons capacity and SEER 15:

Material Cost = (1,065.28 × 3 tons) + (594.74 × 15 SEER) - 2,291
              = $9,826

Installation = Material Cost × 1.5 (retrofit multiplier)
             = $9,826 × 1.5 = $14,739

Labor Cost = $14,739 - $9,826 = $4,913

Total Installed Cost = $14,739
```

### Side-by-Side Comparison

| Feature | Old System | New System |
|---------|-----------|------------|
| **Cost Basis** | Fixed values | Regression equations |
| **Inputs Required** | Technology name only | Performance metrics (SEER, capacity, etc.) |
| **Cost Components** | Combined | Separated (material + labor) |
| **Currency Year** | 2013$ → adjusted | 2023$ (current) |
| **Customization** | Limited to 3 cost levels | Continuous based on actual specs |
| **Installation** | Included in base cost | Separate multipliers/adders |
| **Uncertainty** | Normal distribution sampling | Quantile regression (10th/50th/90th) |

---

## Step-by-Step Migration Process

### Overview of 7 Phases

```
Phase 1: Update Spreadsheet Structure (Data Entry)
    ↓
Phase 2: Create Data Loading Functions (Code)
    ↓
Phase 3: Update Cost Calculation Logic (Code)
    ↓
Phase 4: Modify Workflow Notebooks (Code)
    ↓
Phase 5: Add Validation Checks (Code)
    ↓
Phase 6: Test Everything (Verification)
    ↓
Phase 7: Deploy and Document (Implementation)
```

---

## Phase 1: Update the Excel Spreadsheet

### Current Spreadsheet: `tare_retrofit_costs_cpi.xlsx`

**Current sheets:**
1. CPI (inflation adjustments)
2. heating_costs
3. waterHeating_costs
4. clothesDrying_costs
5. cooking_costs
6. enclosure_upgrade_costs

### New Spreadsheet: `tare_retrofit_costs_remdb_2023.xlsx`

**New sheets (CPI removed, structure changed):**
1. heating_costs (redesigned)
2. waterHeating_costs (redesigned)
3. clothesDrying_costs (redesigned)
4. cooking_costs (redesigned)
5. enclosure_upgrade_costs (redesigned)

---

### Redesigning the Heating Costs Sheet

#### Old Structure (18 columns):
```
action_measure | technology | fuel | efficiency | data_year | cpi_ratio |
cost_multiplier | unitCost_progressive | unitCost_reference |
unitCost_conservative | cost_per_kBtuh_progressive |
cost_per_kBtuh_reference | cost_per_kBtuh_conservative |
otherCost_progressive | otherCost_reference | otherCost_conservative |
lifetime | source | notes
```

#### New Structure (25 columns):
```
Component | Class | Scenario | Output_Units |
Metric1_Name | Metric1_Unit | Metric1_LowerBound | Metric1_UpperBound |
Coefficient1_Low | Coefficient1_Mid | Coefficient1_High |
Metric2_Name | Metric2_Unit | Metric2_LowerBound | Metric2_UpperBound |
Coefficient2_Low | Coefficient2_Mid | Coefficient2_High |
Intercept_Low | Intercept_Mid | Intercept_High |
Installation_Type | Installation_NewConstruction | Installation_Retrofit |
Lifetime | Cost_Variation | Data_Sources | Qualitative_Rank | Notes
```

---

### Example: Air Source Heat Pump Entry

**What you need to enter:**

| Column | Value | Explanation |
|--------|-------|-------------|
| Component | Heat Pump | Broad category |
| Class | Air Source Heat Pump | Specific type |
| Scenario | Retrofit | New construction or retrofit |
| Output_Units | 2023$ | Final cost units |
| | | |
| **Performance Metric 1 (Capacity)** | | |
| Metric1_Name | Capacity | What drives cost |
| Metric1_Unit | tons | Units of measurement |
| Metric1_LowerBound | 1.5 | Minimum valid value |
| Metric1_UpperBound | 5.0 | Maximum valid value |
| Coefficient1_Low | 639.17 | 10th percentile coefficient |
| Coefficient1_Mid | 1065.28 | 50th percentile coefficient |
| Coefficient1_High | 1491.39 | 90th percentile coefficient |
| | | |
| **Performance Metric 2 (Efficiency)** | | |
| Metric2_Name | SEER1 | What drives cost |
| Metric2_Unit | unitless | Units of measurement |
| Metric2_LowerBound | 13 | Minimum valid value |
| Metric2_UpperBound | 21 | Maximum valid value |
| Coefficient2_Low | 365.84 | 10th percentile coefficient |
| Coefficient2_Mid | 594.74 | 50th percentile coefficient |
| Coefficient2_High | 832.64 | 90th percentile coefficient |
| | | |
| **Regression Constants** | | |
| Intercept_Low | -1374.60 | 10th percentile intercept |
| Intercept_Mid | -2291.00 | 50th percentile intercept |
| Intercept_High | -3207.40 | 90th percentile intercept |
| | | |
| **Installation Costs** | | |
| Installation_Type | Multiplier | Multiplier or Adder |
| Installation_NewConstruction | 1.3 | New construction multiplier |
| Installation_Retrofit | 1.5 | Retrofit multiplier |
| | | |
| **Additional Information** | | |
| Lifetime | 15 | Expected equipment life (years) |
| Cost_Variation | See REMDB | Factors that affect cost |
| Data_Sources | 1, 2, 3 | Reference numbers |
| Qualitative_Rank | High R2, High SS | Data quality rating |
| Notes | See NREL REMDB 2023 | Any special notes |

---

### Example: Attic Insulation Entry

**For enclosure upgrades (uses adders, not multipliers):**

| Column | Value | Explanation |
|--------|-------|-------------|
| Component | Insulation | Broad category |
| Class | Unfinished Attic Ceiling Batt | Specific type |
| Output_Units | 2023$/sqft | Cost per square foot |
| | | |
| Metric1_Name | R_Value | Target insulation value |
| Metric1_Unit | unitless | R-value has no units |
| Metric1_LowerBound | 0 | Minimum R-value |
| Metric1_UpperBound | 60 | Maximum R-value |
| Coefficient1_Low | 0.03 | 10th percentile |
| Coefficient1_Mid | 0.04 | 50th percentile |
| Coefficient1_High | 0.05 | 90th percentile |
| | | |
| Metric2_Name | (leave blank) | No second metric |
| | | |
| Intercept_Low | 0.16 | 10th percentile |
| Intercept_Mid | 0.26 | 50th percentile |
| Intercept_High | 1.56 | 90th percentile |
| | | |
| Installation_Type | Adder | Fixed labor cost added |
| Installation_NewConstruction | 0.83 | $ per sqft for new |
| Installation_Retrofit | 1.00 | $ per sqft for retrofit |
| | | |
| Lifetime | 50 | Insulation lasts 50 years |

**Calculation example:**
```
For R-49 insulation retrofit:
Material: (0.04 × 49) + 0.26 = $2.22/sqft
Labor: $1.00/sqft (fixed adder)
Total: $3.22/sqft

For 1,500 sqft attic:
Total cost = $3.22/sqft × 1,500 sqft = $4,830
```

---

### Where to Get the Data

The regression coefficients, intercepts, and bounds come from the **NREL REMDB 2023 Machine Readable File**. You'll need to:

1. Download the official REMDB 2023 database from NREL
2. Locate the machine-readable CSV file
3. Find the appropriate component and class
4. Copy the coefficient values into your Excel spreadsheet

**Components you need data for:**

**Heating:**
- Air Source Heat Pump
- Mini-Split Heat Pump
- Mini-Split Heat Pump - Ducted
- Natural Gas Furnace (for replacement costs)
- Electric Furnace (for replacement costs)
- Fuel Oil Furnace (for replacement costs)
- Propane Furnace (for replacement costs)

**Water Heating:**
- Heat Pump Water Heater, 50 gal
- Heat Pump Water Heater, 66 gal
- Heat Pump Water Heater, 80 gal
- Electric Water Heater (for replacement costs)
- Gas Water Heater (for replacement costs)

**Clothes Drying:**
- Electric Heat Pump Clothes Dryer
- Electric Standard Clothes Dryer (for replacement costs)
- Gas Clothes Dryer (for replacement costs)

**Cooking:**
- Electric Induction Range
- Electric Resistance Range (for replacement costs)
- Gas Range (for replacement costs)

**Enclosure:**
- Attic Floor Insulation (various R-values)
- Air Sealing
- Duct Sealing
- Wall Insulation
- Foundation Wall Insulation
- Rim Joist Insulation
- Crawlspace Sealing

---

## Phase 2: Create Data Loading Functions

### What This Does

Instead of reading the Excel file and manually creating dictionaries, we need new functions that:
1. Read the regression structure
2. Store coefficients, intercepts, and bounds
3. Make them easily accessible for calculations

### File to Create: `cmu_tare_model/utils/load_remdb_data.py`

This file will contain three main functions:

#### Function 1: `load_remdb_regression_data()`
**Purpose:** Load all regression data from Excel into Python

**What it returns:**
A dictionary that looks like this:
```python
{
    ('Heat Pump', 'Air Source Heat Pump'): {
        'metric1': {
            'name': 'Capacity',
            'unit': 'tons',
            'bounds': (1.5, 5.0),
            'coefficients': {'low': 639.17, 'mid': 1065.28, 'high': 1491.39}
        },
        'metric2': {
            'name': 'SEER1',
            'unit': 'unitless',
            'bounds': (13, 21),
            'coefficients': {'low': 365.84, 'mid': 594.74, 'high': 832.64}
        },
        'intercepts': {'low': -1374.60, 'mid': -2291.00, 'high': -3207.40},
        'installation': {
            'type': 'Multiplier',
            'new_construction': 1.3,
            'retrofit': 1.5
        },
        'lifetime': 15
    }
}
```

#### Function 2: `calculate_cost_from_regression()`
**Purpose:** Calculate the cost for one home using the regression equation

**Inputs:**
- Cost data (from function 1)
- Performance metric values (e.g., 3 tons, SEER 15)
- Scenario (retrofit or new construction)
- Percentile (low/mid/high - which estimate to use)

**Output:**
Three numbers: material cost, installation cost, total cost

**Example:**
```python
material, installation, total = calculate_cost_from_regression(
    cost_data=ashp_data,
    metric1_value=3.0,      # 3 tons
    metric2_value=15.0,     # SEER 15
    scenario='retrofit',
    percentile='mid'
)
# Returns: ($9,826, $4,913, $14,739)
```

#### Function 3: `calculate_cost_probabilistic()`
**Purpose:** Randomly sample cost using normal distribution (for uncertainty)

**Why this matters:**
Not all 3-ton, SEER 15 heat pumps cost exactly the same. This function creates realistic variation by sampling from a distribution where:
- The middle (mean) is at the 50th percentile cost
- The spread captures the range between 10th and 90th percentiles
- No negative costs are allowed

**Example:**
```python
# Get 1000 different cost samples for the same equipment
costs = calculate_cost_probabilistic(
    cost_data=ashp_data,
    metric1_value=3.0,
    metric2_value=15.0,
    scenario='retrofit',
    n_samples=1000
)
# Returns array of 1000 values, average around $14,739
# Range approximately $9,000 to $20,000
```

---

## Phase 3: Update Cost Calculation Logic

### What Needs to Change

The current functions that calculate costs need to be updated to:
1. Extract performance metrics from home data
2. Match homes to the right equipment class
3. Use regression instead of lookup
4. Handle cases where metrics are missing or out of bounds

### Three Main Files to Update

#### File 1: `calculate_equipment_installation_costs.py`

**Old function:** `calculate_installation_cost()`
**New function:** `calculate_installation_cost_remdb()`

**Key changes:**

**BEFORE (simplified):**
```python
def calculate_installation_cost(df, cost_dict, menu_mp, end_use):
    # Look up technology and efficiency from upgrade column
    tech = df['upgrade_heating_type']
    eff = df['upgrade_heating_efficiency']

    # Get fixed cost from dictionary
    for tech, eff in combinations:
        cost = cost_dict[(tech, eff)]['unitCost']
        cost += capacity * cost_dict[(tech, eff)]['cost_per_kBtuh']

    # Apply CPI adjustment
    cost = cost * cpi_ratio

    return df with cost column
```

**AFTER (simplified):**
```python
def calculate_installation_cost_remdb(df, cost_dict, menu_mp, end_use, scenario='retrofit'):
    # Extract performance metrics
    capacity_tons = df['total_heating_load_kBtuh'] / 12.0
    seer_rating = extract_seer_from_string(df['upgrade_heating_efficiency'])

    # Map homes to component/class
    component = 'Heat Pump'
    class_type = 'Air Source Heat Pump' if has_ducts else 'Mini-Split Heat Pump'

    # Get regression data
    cost_data = cost_dict[(component, class_type)]

    # Calculate cost using regression
    for each home:
        cost = calculate_cost_probabilistic(
            cost_data,
            metric1_value=capacity_tons,
            metric2_value=seer_rating,
            scenario=scenario
        )

    # No CPI adjustment needed (already in 2023$)

    return df with cost columns (material, labor, total)
```

#### File 2: `calculate_equipment_replacement_costs.py`

**Similar changes needed, but using BASELINE equipment metrics instead of upgrade metrics**

**Key difference:**
```python
# For installation costs, use UPGRADE equipment:
seer = df['upgrade_hvac_heating_efficiency']  # e.g., "SEER 18"

# For replacement costs, use BASELINE equipment:
seer = df['baseline_SEER']  # e.g., 13
```

#### File 3: `calculate_enclosure_upgrade_costs.py`

**Changes needed for insulation and weatherization:**

**Key difference from equipment:**
- Output units are $/sqft instead of total $
- Need to multiply by area at the end
- Usually only one performance metric (R-value)

**Example calculation:**
```python
# Get target R-value
r_value = extract_r_value(df['upgrade_insulation_atticFloor'])  # e.g., 49

# Get attic area
area = df['area_attic_floor_sqft']  # e.g., 1,500

# Calculate cost per sqft using regression
cost_per_sqft = calculate_cost_from_regression(
    cost_data=insulation_data,
    metric1_value=r_value,
    metric2_value=None,  # Only one metric
    scenario='retrofit',
    percentile='mid'
)  # Returns $3.22/sqft

# Multiply by area
total_cost = cost_per_sqft * area  # $3.22 × 1,500 = $4,830
```

---

### New Helper Functions Needed

#### Function: `extract_performance_metrics()`
**Purpose:** Pull the right numbers from the dataframe

**For heating:**
```python
def extract_performance_metrics(df, end_use='heating'):
    if end_use == 'heating':
        # Metric 1: Capacity in tons
        capacity = df['total_heating_load_kBtuh'] / 12.0

        # Metric 2: SEER rating
        # Extract from string like "SEER 18, 9.3 HSPF"
        seer = df['upgrade_hvac_heating_efficiency'].str.extract(r'SEER\s+([\d.]+)')

        return capacity, seer
```

**For water heating:**
```python
    if end_use == 'waterHeating':
        # Metric 1: UEF (energy factor)
        # Extract from string like "3.45 UEF"
        uef = df['upgrade_water_heater_efficiency'].str.extract(r'([\d.]+)\s+UEF')

        # Metric 2: Tank size
        gallons = df['size_water_heater_gal']

        return uef, gallons
```

**For enclosure:**
```python
    if end_use == 'attic_insulation':
        # Metric 1: Target R-value
        # Extract from string like "R-49"
        r_value = df['upgrade_insulation_atticFloor'].str.extract(r'R-(\d+)')

        return r_value, None  # No second metric
```

#### Function: `get_component_class_mapping()`
**Purpose:** Figure out which equipment class each home needs

**Example logic:**
```python
def get_component_class_mapping(df, end_use, menu_mp):
    mapping = {}

    for idx, row in df.iterrows():
        if end_use == 'heating':
            # Decision: ASHP vs MSHP vs MSHP-Ducted
            has_ducts = row['hvac_has_ducts'] == 'Yes'

            if menu_mp == 8:  # Basic package
                if has_ducts:
                    mapping[idx] = ('Heat Pump', 'Air Source Heat Pump')
                else:
                    mapping[idx] = ('Heat Pump', 'Mini-Split Heat Pump')

            elif menu_mp == 10:  # Advanced package
                if has_ducts:
                    mapping[idx] = ('Heat Pump', 'Mini-Split Heat Pump - Ducted')
                else:
                    mapping[idx] = ('Heat Pump', 'Mini-Split Heat Pump')

        elif end_use == 'waterHeating':
            # Decision based on tank size
            tank_size = row['size_water_heater_gal']

            if tank_size <= 55:
                mapping[idx] = ('Water Heater', 'HP Tank 50 gal')
            elif tank_size <= 73:
                mapping[idx] = ('Water Heater', 'HP Tank 66 gal')
            else:
                mapping[idx] = ('Water Heater', 'HP Tank 80 gal')

    return mapping
```

---

## Phase 4: Update Workflow Notebooks

### Files to Modify

The scenario notebooks that run the model need updated function calls:

- `model_scenarios/tare_basic_v2_1.ipynb`
- `model_scenarios/tare_moderate_v2_1.ipynb`
- `model_scenarios/tare_advanced_v2_1.ipynb`

### What to Change

#### Change 1: Loading Cost Data

**OLD CODE (around line 79a2a06d):**
```python
# Load Excel file
filename = "tare_retrofit_costs_cpi.xlsx"
file_path = os.path.join(PROJECT_ROOT, "data", "retrofit_costs", filename)

df_heating_retrofit_costs = pd.read_excel(file_path, sheet_name='heating_costs')

# Apply CPI adjustments
cost_columns = ['unitCost_progressive', 'unitCost_reference', ...]
for column in cost_columns:
    df_heating_retrofit_costs[column] = (
        df_heating_retrofit_costs[column] *
        df_heating_retrofit_costs['cpi_ratio'] *
        df_heating_retrofit_costs['cost_multiplier']
    )

# Create dictionary
dict_heating_equipment_cost = df_heating_retrofit_costs.set_index(
    ['technology', 'efficiency']).to_dict(orient='index')
```

**NEW CODE:**
```python
from cmu_tare_model.utils.load_remdb_data import load_remdb_regression_data

# Load new regression-based cost data
filename = "tare_retrofit_costs_remdb_2023.xlsx"
file_path = os.path.join(PROJECT_ROOT, "data", "retrofit_costs", filename)

print(f"Loading NREL REMDB 2023 data from: {file_path}")

# Load all equipment costs
dict_heating_equipment_cost = load_remdb_regression_data(
    file_path, 'heating_costs')

dict_waterHeating_equipment_cost = load_remdb_regression_data(
    file_path, 'waterHeating_costs')

dict_clothesDrying_equipment_cost = load_remdb_regression_data(
    file_path, 'clothesDrying_costs')

dict_cooking_equipment_cost = load_remdb_regression_data(
    file_path, 'cooking_costs')

dict_enclosure_retrofit_cost = load_remdb_regression_data(
    file_path, 'enclosure_upgrade_costs')

# No CPI adjustment needed - data already in 2023$
print("Cost data loaded successfully (2023$ basis)")
```

#### Change 2: Calling Cost Calculation Functions

**OLD CODE:**
```python
df_euss_am_mp8_home = calculate_installation_cost(
    df=df_euss_am_mp8_home,
    cost_dict=dict_heating_equipment_cost,
    menu_mp=menu_mp,
    end_use='heating'
)
```

**NEW CODE:**
```python
df_euss_am_mp8_home = calculate_installation_cost_remdb(
    df=df_euss_am_mp8_home,
    cost_dict=dict_heating_equipment_cost,
    menu_mp=menu_mp,
    end_use='heating',
    scenario='retrofit'  # NEW: specify retrofit vs new construction
)
```

**Repeat for all equipment types:**
- Water heating
- Clothes drying
- Cooking
- Enclosure upgrades

#### Change 3: Remove CPI-Related Code

**DELETE these sections:**
```python
# No longer needed
from cmu_tare_model.utils.inflation_adjustment import *

# No longer needed
cpi_ratio_2023_2013 = 1.3079752915774157
```

The new data is already in 2023 dollars.

---

## Phase 5: Add Validation Checks

### Why Validation Matters

The regression equations only work within certain bounds. For example:
- ASHP capacity: 1.5 to 5.0 tons
- SEER rating: 13 to 21

If a home needs a 6-ton heat pump, it's **outside the regression bounds** and we need to handle it specially.

### Validation Function to Add

**File:** `cmu_tare_model/utils/validate_remdb_metrics.py`

```python
def validate_metric_bounds(df, end_use, cost_dict):
    """
    Check if homes have performance metrics within valid regression bounds.
    Flag homes that are outside bounds for review.
    """

    # Extract metrics
    metrics = extract_performance_metrics(df, end_use)

    # Map homes to component/class
    mapping = get_component_class_mapping(df, end_use, menu_mp=8)

    # Check each home
    validation_report = []

    for idx in df.index:
        component, class_type = mapping[idx]
        cost_data = cost_dict[(component, class_type)]

        # Check metric 1
        metric1_val = metrics.loc[idx, 'metric1']
        if cost_data['metric1']:
            lower, upper = cost_data['metric1']['bounds']

            if not (lower <= metric1_val <= upper):
                validation_report.append({
                    'home_id': idx,
                    'metric': cost_data['metric1']['name'],
                    'value': metric1_val,
                    'valid_range': f"[{lower}, {upper}]",
                    'status': 'OUT OF BOUNDS'
                })

        # Check metric 2 (if exists)
        if cost_data['metric2']:
            metric2_val = metrics.loc[idx, 'metric2']
            lower, upper = cost_data['metric2']['bounds']

            if not (lower <= metric2_val <= upper):
                validation_report.append({
                    'home_id': idx,
                    'metric': cost_data['metric2']['name'],
                    'value': metric2_val,
                    'valid_range': f"[{lower}, {upper}]",
                    'status': 'OUT OF BOUNDS'
                })

    return pd.DataFrame(validation_report)
```

### How to Use

**Add this check before running cost calculations:**
```python
# Validate heating equipment metrics
validation_heating = validate_metric_bounds(
    df_euss_am_mp8_home,
    'heating',
    dict_heating_equipment_cost
)

if len(validation_heating) > 0:
    print(f"\nWARNING: {len(validation_heating)} homes outside regression bounds:")
    print(validation_heating)
    print("\nThese homes will be excluded from cost calculations.")
else:
    print("✓ All homes within valid metric bounds for heating")
```

### What to Do with Out-of-Bounds Homes

**Option 1 (Conservative):** Exclude them from the analysis
```python
# Mark homes as invalid if outside bounds
df['include_heating'] = df['include_heating'] & df['metrics_in_bounds']
```

**Option 2 (Extrapolation):** Allow the calculation but log a warning
```python
# Continue calculation but flag for review
df['cost_extrapolated'] = True  # Flag for review
# Proceed with cost calculation
```

**Recommendation:** Start with Option 1 (exclude), then review the excluded homes to see if the bounds need to be expanded or if special handling is needed.

---

## Phase 6: Testing and Validation

### Step 1: Unit Tests

Create test file: `tests/test_remdb_regression.py`

**Test 1: Verify NREL Example - Air Source Heat Pump**
```python
def test_ashp_matches_nrel_documentation():
    """
    Test that our calculation matches the NREL documentation example.

    Example from NREL docs:
    - 3 tons capacity
    - 15 SEER
    - Retrofit installation
    - Expected: $9,826 material, $14,739 total (mid percentile)
    """
    # Set up test data
    cost_data = {
        'metric1': {
            'coefficients': {'low': 639.17, 'mid': 1065.28, 'high': 1491.39}
        },
        'metric2': {
            'coefficients': {'low': 365.84, 'mid': 594.74, 'high': 832.64}
        },
        'intercepts': {'low': -1374.60, 'mid': -2291.00, 'high': -3207.40},
        'installation': {'type': 'Multiplier', 'retrofit': 1.5}
    }

    # Calculate
    material, installation, total = calculate_cost_from_regression(
        cost_data, 3.0, 15.0, 'retrofit', 'mid')

    # Verify
    assert abs(material - 9825.94) < 0.10, f"Material cost wrong: {material}"
    assert abs(total - 14738.91) < 0.10, f"Total cost wrong: {total}"

    print("✓ ASHP calculation matches NREL documentation")
```

**Test 2: Verify NREL Example - Attic Insulation**
```python
def test_insulation_matches_nrel_documentation():
    """
    Test attic insulation calculation.

    Example from NREL docs:
    - R-15 insulation
    - Retrofit installation
    - Expected: $0.86/sqft material, $1.86/sqft total (mid percentile)
    """
    cost_data = {
        'metric1': {
            'coefficients': {'low': 0.03, 'mid': 0.04, 'high': 0.05}
        },
        'metric2': None,
        'intercepts': {'low': 0.16, 'mid': 0.26, 'high': 1.56},
        'installation': {'type': 'Adder', 'retrofit': 1.00}
    }

    material, installation, total = calculate_cost_from_regression(
        cost_data, 15.0, None, 'retrofit', 'mid')

    assert abs(material - 0.86) < 0.01, f"Material cost wrong: {material}"
    assert abs(total - 1.86) < 0.01, f"Total cost wrong: {total}"

    print("✓ Insulation calculation matches NREL documentation")
```

### Step 2: Compare Old vs New System

**Create comparison script:** `scripts/compare_cost_systems.py`

```python
def compare_old_vs_new_costs():
    """
    Run both old and new cost calculations on same homes.
    Compare results to understand the differences.
    """

    # Load sample of homes
    df_sample = df_euss_am_mp8_home.sample(1000)

    # Calculate with OLD system
    df_old = calculate_installation_cost(
        df_sample.copy(), dict_heating_old, 8, 'heating')

    # Calculate with NEW system
    df_new = calculate_installation_cost_remdb(
        df_sample.copy(), dict_heating_new, 8, 'heating', 'retrofit')

    # Compare
    comparison = pd.DataFrame({
        'old_cost': df_old['mp8_heating_installationCost'],
        'new_cost': df_new['mp8_heating_installationCost'],
        'difference': df_new['mp8_heating_installationCost'] -
                      df_old['mp8_heating_installationCost'],
        'pct_diff': 100 * (df_new['mp8_heating_installationCost'] -
                          df_old['mp8_heating_installationCost']) /
                          df_old['mp8_heating_installationCost']
    })

    # Summary statistics
    print("\n" + "="*80)
    print("COST COMPARISON: Old System vs New System (Heating)")
    print("="*80)
    print(f"Sample size: {len(comparison)} homes")
    print(f"\nOld system (2013$ adjusted):")
    print(f"  Mean cost: ${comparison['old_cost'].mean():,.2f}")
    print(f"  Median cost: ${comparison['old_cost'].median():,.2f}")
    print(f"\nNew system (2023$ regression):")
    print(f"  Mean cost: ${comparison['new_cost'].mean():,.2f}")
    print(f"  Median cost: ${comparison['new_cost'].median():,.2f}")
    print(f"\nDifference:")
    print(f"  Mean difference: ${comparison['difference'].mean():,.2f}")
    print(f"  Median difference: ${comparison['difference'].median():,.2f}")
    print(f"  Mean % difference: {comparison['pct_diff'].mean():.1f}%")
    print("="*80)

    # Histogram of differences
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(comparison['pct_diff'], bins=50)
    plt.xlabel('Percent Difference (%)')
    plt.ylabel('Number of Homes')
    plt.title('Distribution of Cost Differences: New vs Old System')
    plt.axvline(0, color='red', linestyle='--', label='No difference')
    plt.legend()
    plt.savefig('cost_comparison_histogram.png')

    return comparison
```

**Expected outcome:**
- Some variation is normal (different data sources, time periods)
- Flag homes with >50% difference for investigation
- Document systematic patterns (e.g., "new costs 20% higher for large homes")

### Step 3: Validate Against Real-World Costs

If you have actual project cost data:

```python
def validate_against_real_costs():
    """
    Compare model estimates to actual installation costs.
    """

    # Load real project data
    real_costs = pd.read_csv('actual_installation_costs.csv')

    # Calculate model estimates for same projects
    for idx, project in real_costs.iterrows():
        model_cost = calculate_cost_from_regression(
            cost_data=ashp_data,
            metric1_value=project['capacity_tons'],
            metric2_value=project['seer_rating'],
            scenario='retrofit',
            percentile='mid'
        )

        real_costs.loc[idx, 'model_estimate'] = model_cost[2]  # total cost

    # Compare
    real_costs['error'] = real_costs['model_estimate'] - real_costs['actual_cost']
    real_costs['pct_error'] = 100 * real_costs['error'] / real_costs['actual_cost']

    print(f"\nValidation against {len(real_costs)} real projects:")
    print(f"Mean absolute error: ${abs(real_costs['error']).mean():,.2f}")
    print(f"Mean % error: {real_costs['pct_error'].mean():.1f}%")
    print(f"Model within ±20% for {(abs(real_costs['pct_error']) < 20).sum()} homes")
```

---

## Phase 7: Migration Timeline

### Recommended: Phased Approach (6 weeks)

```
Week 1-2: PILOT TEST
├── Update heating costs sheet only
├── Create data loading functions
├── Implement for heating equipment only
├── Run parallel calculations (old + new)
├── Compare results on 1,000 sample homes
├── Investigate any large discrepancies
└── Refine and document

Week 3-4: EXPAND TO ALL EQUIPMENT
├── Update all cost sheets (water, clothes, cooking)
├── Implement for all equipment types
├── Run parallel calculations on full dataset
├── Document systematic differences
├── Adjust as needed
└── Stakeholder review

Week 5: ENCLOSURE COSTS
├── Update enclosure cost sheet
├── Implement regression for insulation, air sealing, etc.
├── Test carefully (more complex with area multipliers)
├── Validate against subset of homes
└── Document

Week 6: FULL DEPLOYMENT
├── Remove old cost functions
├── Update all scenario notebooks
├── Archive old spreadsheet
├── Update documentation
├── Train users
└── Release notes
```

### Alternative: Big Bang Approach (2 weeks)

If timeline is compressed:

```
Week 1: IMPLEMENTATION
├── Update all spreadsheets
├── Implement all code changes
├── Basic testing
└── Fix critical bugs

Week 2: VALIDATION & DEPLOY
├── Comprehensive testing
├── Compare results
├── Document changes
└── Deploy
```

**Risk:** Higher chance of unexpected issues, less time to investigate discrepancies.

---

## Troubleshooting Guide

### Issue 1: Homes Excluded Due to Out-of-Bounds Metrics

**Symptom:** Many homes marked as invalid, fewer homes in analysis

**Possible causes:**
1. Metric extraction error (wrong regex pattern)
2. Regression bounds too narrow
3. Homes truly have unusual equipment sizes

**Solution:**
```python
# Check what metrics are being extracted
print("Sample of extracted metrics:")
print(df[['bldg_id', 'total_heating_load_kBtuh', 'metric1_capacity',
          'upgrade_hvac_heating_efficiency', 'metric2_seer']].head(20))

# Check distribution
print(f"\nCapacity distribution:")
print(df['metric1_capacity'].describe())
print(f"\nSEER distribution:")
print(df['metric2_seer'].describe())

# Compare to regression bounds
print(f"\nRegression bounds:")
print(f"Capacity: {cost_data['metric1']['bounds']}")
print(f"SEER: {cost_data['metric2']['bounds']}")

# Identify which homes are problematic
out_of_bounds = df[~df['metrics_in_bounds']]
print(f"\n{len(out_of_bounds)} homes out of bounds:")
print(out_of_bounds[['bldg_id', 'metric1_capacity', 'metric2_seer']])
```

**Actions:**
- If extraction error: Fix regex patterns
- If bounds too narrow: Consider expanding bounds (with justification)
- If truly unusual: Document and accept exclusion

---

### Issue 2: Negative Costs Calculated

**Symptom:** Some homes have negative installation costs

**Possible causes:**
1. Incorrect intercept values (should be negative for some regressions)
2. Metrics outside valid range causing extrapolation
3. Missing data treated as zero

**Solution:**
```python
# Check for negative costs
negative_costs = df[df['mp8_heating_installationCost'] < 0]
print(f"{len(negative_costs)} homes with negative costs")

# Investigate specific cases
for idx in negative_costs.index[:5]:
    print(f"\nHome {idx}:")
    print(f"  Capacity: {df.loc[idx, 'metric1_capacity']}")
    print(f"  SEER: {df.loc[idx, 'metric2_seer']}")
    print(f"  Calculated cost: ${df.loc[idx, 'mp8_heating_installationCost']}")
```

**Actions:**
- Verify intercept values match REMDB
- Check for missing data (NaN) being used in calculations
- Ensure truncnorm is preventing negative samples

---

### Issue 3: Costs Much Higher/Lower Than Expected

**Symptom:** New costs systematically 50%+ different from old costs

**Possible causes:**
1. CPI adjustment still being applied (double-counting)
2. Wrong installation type (multiplier vs adder)
3. Coefficients entered incorrectly
4. Metrics in wrong units (e.g., kBtuh instead of tons)

**Solution:**
```python
# Manual calculation check
home_idx = df_sample.index[0]
capacity = df.loc[home_idx, 'total_heating_load_kBtuh'] / 12.0
seer = df.loc[home_idx, 'metric2_seer']

print(f"Manual calculation for home {home_idx}:")
print(f"Capacity: {capacity} tons")
print(f"SEER: {seer}")

# Calculate step-by-step
coef1 = cost_data['metric1']['coefficients']['mid']
coef2 = cost_data['metric2']['coefficients']['mid']
intercept = cost_data['intercepts']['mid']

material = (coef1 * capacity) + (coef2 * seer) + intercept
print(f"\nMaterial = ({coef1} × {capacity}) + ({coef2} × {seer}) + {intercept}")
print(f"         = ${material:.2f}")

multiplier = cost_data['installation']['retrofit']
total = material * multiplier
print(f"\nTotal = ${material:.2f} × {multiplier} = ${total:.2f}")

# Compare to function result
func_result = calculate_cost_from_regression(
    cost_data, capacity, seer, 'retrofit', 'mid')
print(f"\nFunction returned: ${func_result[2]:.2f}")
print(f"Match: {abs(total - func_result[2]) < 0.01}")
```

---

### Issue 4: Excel Loading Errors

**Symptom:** KeyError or column not found when loading spreadsheet

**Possible causes:**
1. Column names don't match exactly
2. Extra spaces in column headers
3. Sheet name misspelled

**Solution:**
```python
# Check what columns actually exist
df_test = pd.read_excel(file_path, sheet_name='heating_costs')
print("Columns found in spreadsheet:")
print(df_test.columns.tolist())

# Check for extra spaces
print("\nColumn names (with quotes to see spaces):")
for col in df_test.columns:
    print(f"  '{col}'")

# Load and clean
df_test.columns = df_test.columns.str.strip()  # Remove leading/trailing spaces
```

**Actions:**
- Ensure exact match: `Coefficient1_Low` not `coefficient1_low`
- No spaces: `Metric1_Name` not `Metric1 Name`
- Use underscore: `Output_Units` not `Output-Units`

---

### Issue 5: Missing Performance Metrics

**Symptom:** Many NaN values in extracted metrics

**Possible causes:**
1. Regex pattern doesn't match string format
2. Data not in expected columns
3. Equipment specs not populated in EUSS data

**Solution:**
```python
# Check raw data
print("Sample of efficiency strings:")
print(df['upgrade_hvac_heating_efficiency'].head(20))

# Test regex
test_string = "SEER 18, 9.3 HSPF"
seer = pd.Series([test_string]).str.extract(r'SEER\s+([\d.]+)')
print(f"\nTest extraction: '{test_string}' → SEER {seer[0][0]}")

# Check extraction success rate
df['seer_extracted'] = df['upgrade_hvac_heating_efficiency'].str.extract(r'SEER\s+([\d.]+)')
success_rate = df['seer_extracted'].notna().sum() / len(df) * 100
print(f"\nExtraction success rate: {success_rate:.1f}%")

if success_rate < 90:
    print("WARNING: Low extraction success rate")
    print("Sample of failed extractions:")
    failed = df[df['seer_extracted'].isna()]['upgrade_hvac_heating_efficiency'].head(10)
    print(failed)
```

---

## Key Differences Summary

### Material vs Labor Separation

**Old System:**
- Everything bundled into one "installation cost"
- Can't distinguish equipment from labor

**New System:**
- Material cost calculated separately
- Labor cost explicitly modeled (multiplier or adder)
- Better transparency and cost breakdowns

**Why it matters:**
- Material costs may scale with equipment performance
- Labor costs may be more fixed
- Enables sensitivity analysis on labor rates

---

### Multipliers vs Adders

**Multipliers (equipment):**
```
Total = Material × Multiplier
Labor = Total - Material

Example (ASHP):
Material = $9,826
Multiplier = 1.5 (retrofit)
Total = $9,826 × 1.5 = $14,739
Labor = $14,739 - $9,826 = $4,913
```

**Adders (insulation):**
```
Total = Material + Adder
Labor = Adder

Example (Attic insulation):
Material = $0.86/sqft
Adder = $1.00/sqft (retrofit)
Total = $0.86 + $1.00 = $1.86/sqft
Labor = $1.00/sqft
```

**When to use which:**
- Multiplier: When labor scales with equipment cost (more expensive equipment = more labor)
- Adder: When labor is relatively fixed (insulation labor same regardless of R-value)

---

### Replacement Cost Handling

**Important note:** Replacement costs also need regression treatment

**Current approach (still valid):**
The net capital cost = total capital cost - replacement cost

This represents the **incremental cost** of choosing the efficient option vs. replacing with standard equipment.

**What changes:**
Both costs now calculated via regression:

```python
# Installation cost (upgrade to ASHP)
upgrade_cost = regression(capacity=3 tons, seer=18)  # $14,739

# Replacement cost (baseline furnace)
replacement_cost = regression(capacity=3 tons, afue=95)  # $8,500

# Net cost (what homeowner pays extra for efficiency)
net_cost = upgrade_cost - replacement_cost  # $6,239
```

---

### Installation Premiums

**Special adjustments still apply:**

Some homes need extra work beyond the regression estimate:
- No existing AC (need to add ductwork)
- Boiler systems (more complex removal)
- Electrical upgrades (panel/circuit capacity)

**Current code handles this correctly:**
```python
# Calculate base cost from regression
base_cost = regression_calculation(...)

# Add premium for special cases
if no_existing_ac and has_furnace:
    base_cost += 400  # Premium for homes without AC

if has_boiler:
    base_cost += 1500  # Premium for boiler removal
```

**These premiums should be retained** - they capture real costs not in the regression.

---

## Appendix A: Quick Reference

### Regression Formula

**General form:**
```
Cost = (Coefficient1 × Metric1) + (Coefficient2 × Metric2) + Intercept
```

**For equipment with installation multiplier:**
```
Material = (Coefficient1 × Metric1) + (Coefficient2 × Metric2) + Intercept
Total = Material × Multiplier
Labor = Total - Material
```

**For insulation with installation adder:**
```
Material_per_sqft = (Coefficient1 × R_value) + Intercept
Total_per_sqft = Material_per_sqft + Adder
Total_cost = Total_per_sqft × Area
```

### Common Metrics by Equipment Type

| Equipment | Metric 1 | Metric 2 | Output Units |
|-----------|----------|----------|--------------|
| Air Source Heat Pump | Capacity (tons) | SEER | 2023$ |
| Mini-Split Heat Pump | Capacity (tons) | SEER | 2023$ |
| Heat Pump Water Heater | UEF | Tank size (gal) | 2023$ |
| Attic Insulation | R-value | None | 2023$/sqft |
| Air Sealing | % reduction | None | 2023$ |

### Data Sources

**Where to find REMDB data:**
- NREL REMDB website: https://remdb.nrel.gov/
- Machine-readable file (CSV format)
- Documentation and user guide
- Published research papers

**Key contacts:**
- NREL ResStock team
- REMDB support email

---

## Appendix B: Glossary

**CPI (Consumer Price Index):** Measure of inflation; used in old system to adjust historical costs

**Quantile Regression:** Statistical method that estimates different percentiles (10th, 50th, 90th) rather than just the mean

**Coefficient:** Multiplier in regression equation; shows how much cost changes per unit of performance metric

**Intercept:** Constant term in regression equation; base cost when all metrics are zero

**UEF (Unified Energy Factor):** Energy efficiency metric for water heaters (higher is better)

**SEER (Seasonal Energy Efficiency Ratio):** Cooling efficiency for heat pumps and AC (higher is better)

**HSPF (Heating Seasonal Performance Factor):** Heating efficiency for heat pumps (higher is better)

**AFUE (Annual Fuel Utilization Efficiency):** Efficiency for furnaces and boilers (higher is better)

**Multiplier:** Factor applied to material cost to get total installed cost (accounts for labor)

**Adder:** Fixed amount added to material cost (typically for labor)

**Bounds:** Valid range for performance metrics in regression (extrapolation beyond bounds may be inaccurate)

**Retrofit:** Upgrading existing equipment (vs new construction)

**NPV (Net Present Value):** Present value of future savings minus upfront costs

**WTP (Willingness to Pay):** How much a homeowner is willing to spend upfront

---

## Appendix C: Checklist

### Pre-Migration

- [ ] Downloaded NREL REMDB 2023 machine-readable file
- [ ] Reviewed documentation and examples
- [ ] Identified all equipment types needed
- [ ] Backed up current spreadsheet and code
- [ ] Documented current cost calculation process

### Spreadsheet Updates

- [ ] Created new Excel file: `tare_retrofit_costs_remdb_2023.xlsx`
- [ ] Updated heating_costs sheet
- [ ] Updated waterHeating_costs sheet
- [ ] Updated clothesDrying_costs sheet
- [ ] Updated cooking_costs sheet
- [ ] Updated enclosure_upgrade_costs sheet
- [ ] Verified all coefficients and intercepts
- [ ] Checked all column headers match code expectations
- [ ] Removed CPI sheet

### Code Updates

- [ ] Created `utils/load_remdb_data.py`
- [ ] Updated `calculate_equipment_installation_costs.py`
- [ ] Updated `calculate_equipment_replacement_costs.py`
- [ ] Updated `calculate_enclosure_upgrade_costs.py`
- [ ] Created `extract_performance_metrics()` function
- [ ] Created `get_component_class_mapping()` function
- [ ] Created `validate_remdb_metrics.py`
- [ ] Updated all scenario notebooks
- [ ] Removed CPI-related code

### Testing

- [ ] Created unit tests for regression calculations
- [ ] Tested NREL documentation examples
- [ ] Created cost comparison script (old vs new)
- [ ] Ran parallel calculations on sample data
- [ ] Investigated discrepancies
- [ ] Validated metric bounds
- [ ] Checked for negative costs
- [ ] Verified no NaN/missing values in results

### Documentation

- [ ] Updated CHANGELOG.md
- [ ] Documented new spreadsheet structure
- [ ] Created regression calculation examples
- [ ] Noted assumptions and limitations
- [ ] Listed homes excluded (if any)
- [ ] Updated user guide
- [ ] Created migration notes

### Deployment

- [ ] Archived old spreadsheet
- [ ] Deployed new code to production
- [ ] Updated all scenario scripts
- [ ] Notified stakeholders
- [ ] Provided training/walkthrough
- [ ] Created release notes

---

## Questions or Issues?

**For technical support:**
- Review troubleshooting section (page XX)
- Check code comments and docstrings
- Consult NREL REMDB documentation

**For data questions:**
- Refer to NREL REMDB user guide
- Contact NREL ResStock team
- Review published methodology papers

**For model-specific questions:**
- Contact model development team
- Review CMU TARE Model documentation
- Check project GitHub repository

---

*Document Version 1.0 - Created for CMU TARE Model Migration Project*
*Last Updated: [Current Date]*
