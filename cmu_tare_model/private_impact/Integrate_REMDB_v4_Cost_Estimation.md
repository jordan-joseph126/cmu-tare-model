TARE Model: Integrate REMDB v4 Capital Cost Estimation
Focus on core functionality first, document next steps second.

PROJECT CONTEXT
Current State: cmu-tare-model-v2-3 branch

Successfully refactored to use nested dictionaries and loops
Discount rate sensitivity implemented with loops over 4 rates
Main scenario file: tare_scenarios_v2_2.ipynb (unified, parameterized)
REMDB v4 Files: Already copied to cmu_tare_model/private_impact/remdb_v4_update/

These are the most recent REMDB v4 implementations
Need to integrate into v2-3 workflow following existing patterns
Goal: Integrate REMDB v4 capital cost estimation following the same loop pattern used for discount rates.

WHAT NEEDS TO HAPPEN

Priority 1: Review REMDB v4 Files in their folder: cmu_tare_model/private_impact/remdb_v4_update/

Actions:
- Review files in remdb_v4_update folder (stored in one place for convenience)
- Refactor so it is compatible with the current codebase and add to the appropriate locations (in the same directory as their remdb v3 counterparts)
    - For example Cost calculation functions → Keep in private_impact/ or move to utils/
- Simplify during integration:
    - Remove excessive intermediate columns
    - Reduce print verbosity (use compact ✓/✗ indicators)
    - Keep only essential functions

Priority 2: Define Cost Scenario Constants

Location: cmu_tare_model/constants.py

Add these constants:

```python
# ============================================================================
# CAPITAL COST SCENARIOS
# ============================================================================

# Cost calculation methods and percentiles to loop over
COST_SCENARIO_KEYS = [
    'remdb_v3',
    'remdb_v4_low',
    'remdb_v4_mid', 
    'remdb_v4_high'
]

```

Priority 3: Update Scenario Notebook Capital Costs Section

Location: "PRIVATE IMPACTS: CAPITAL COSTS" section of tare_scenarios_v2_2.ipynb

Replace current code with loop structure (an example is shown below):

```python
# ============================================================================
# PRIVATE IMPACTS: CAPITAL COSTS
# ============================================================================

from cmu_tare_model.constants import COST_SCENARIO_KEYS, IMPLEMENTED_ENDUSES, parse_cost_scenario

# Import REMDB v4 functions (adjust import paths as needed after file moves)
from cmu_tare_model.private_impact.remdb_v4_update.remdb_v4_installed_cost_utils import (
    load_remdb_v4_data,
    add_remdb_replacement_metrics,
    add_remdb_upgrade_metrics
)
from cmu_tare_model.private_impact.remdb_v4_update.calculate_equipment_replacement_costs import (
    calculate_replacement_installed_cost
)
from cmu_tare_model.private_impact.remdb_v4_update.calculate_equipment_installation_costs import (
    calculate_upgrade_installed_cost
)

# Load REMDB v4 data
print("\nLoading REMDB v4 cost database...")
remdb_v4_costs = load_remdb_v4_data()
print("✓ REMDB v4 data loaded\n")

# Initialize nested dictionary for capital costs
# Structure: CAPITAL_COSTS[end_use][cost_type][scenario_key] = DataFrame
CAPITAL_COSTS_MPX = {
    end_use: {
        'replacement': {},
        'upgrade': {}
    }
    for end_use in IMPLEMENTED_ENDUSES
}

print("="*80)
print(f"CALCULATING CAPITAL COSTS - MEASURE PACKAGE {menu_mp}")
print("="*80)
print(f"Cost scenarios: {COST_SCENARIO_KEYS}")
print(f"End-uses: {IMPLEMENTED_ENDUSES}\n")

# Loop over cost scenarios
for scenario_key in COST_SCENARIO_KEYS:
    method, percentile = parse_cost_scenario(scenario_key)
    
    print(f"\n{'─'*80}")
    print(f"Cost Scenario: {scenario_key.upper()}")
    print(f"  Method: {method} | Percentile: {percentile if percentile else 'N/A'}")
    print(f"{'─'*80}")
    
    # Make a copy of the base DataFrame for this cost scenario
    df_cost_scenario = df_mpX_home.copy()
    
    # Calculate costs for each end-use
    for end_use in IMPLEMENTED_ENDUSES:
        print(f"  {end_use.title()}: ", end="", flush=True)
        
        if method == 'remdb_v4':
            # === REPLACEMENT COSTS (Baseline Equipment) ===
            df_cost_scenario = add_remdb_replacement_metrics(
                df_cost_scenario, 
                remdb_v4_costs, 
                end_use, 
                percentile
            )
            df_cost_scenario = calculate_replacement_installed_cost(
                df_cost_scenario, 
                menu_mp=0, 
                end_use=end_use, 
                percentile=percentile
            )
            
            # === UPGRADE COSTS (Retrofit Equipment) ===
            df_cost_scenario = add_remdb_upgrade_metrics(
                df_cost_scenario, 
                remdb_v4_costs, 
                end_use, 
                percentile
            )
            df_cost_scenario = calculate_upgrade_installed_cost(
                df_cost_scenario, 
                menu_mp=menu_mp, 
                end_use=end_use, 
                percentile=percentile
            )
            
            print("✓", end=" ")
        
        elif method == 'remdb_v3':
            # TODO: Add remdb_v3 cost calculation method
            print("⊘ (not implemented)", end=" ")
    
    # Store complete DataFrame for this cost scenario
    # Each end-use gets its own copy (for now - can optimize later)
    for end_use in IMPLEMENTED_ENDUSES:
        CAPITAL_COSTS_MPX[end_use]['replacement'][scenario_key] = df_cost_scenario.copy()
        CAPITAL_COSTS_MPX[end_use]['upgrade'][scenario_key] = df_cost_scenario.copy()
    
    print()  # Newline after end-uses

print("\n" + "="*80)
print("CAPITAL COST CALCULATION COMPLETE")
print("="*80)
print(f"Scenarios calculated: {len(COST_SCENARIO_KEYS)}")
print(f"End-uses: {len(IMPLEMENTED_ENDUSES)}")
print(f"Total DataFrames stored: {len(COST_SCENARIO_KEYS) * len(IMPLEMENTED_ENDUSES) * 2}")
print("="*80 + "\n")
```

Priority 4: Add helper function or code to handle combined heating and cooling replacement costs.

```python
# =============================================================================
# COMBINE COSTS
# =============================================================================
heating_replacement_col = f'mp{menu_mp}_heating_replacement_installed_cost_{percentile}'
cooling_replacement_col = f'mp{menu_mp}_cooling_replacement_installed_cost_{percentile}'
combined_replacement_col = f'mp{menu_mp}_heating_and_cooling_replacement_installed_cost_{percentile}'
combined_net_capital_col = f'mp{menu_mp}_heating_and_cooling_net_capital_cost_{percentile}'

# Main DataFrame (this works)
df_euss_am_mp4_home[combined_replacement_col] = (
    df_euss_am_mp4_home[heating_replacement_col] + 
    df_euss_am_mp4_home[cooling_replacement_col]
)

df_euss_am_mp4_home[combined_net_capital_col] = (
    df_euss_am_mp4_home[f'mp{menu_mp}_heating_upgrade_installed_cost_{percentile}'] -
    df_euss_am_mp4_home[combined_replacement_col]
    )

# Combine detailed DataFrames
df_detailed_combined = pd.concat([
    df_detailed_heating_upgrade,
    df_detailed_heating_replacement,
    df_detailed_cooling_replacement
], axis=1)

# Remove duplicate columns (if any)
df_detailed_combined = df_detailed_combined.loc[:, ~df_detailed_combined.columns.duplicated()]

# Add combined cost column
df_detailed_combined[combined_replacement_col] = (
    df_detailed_combined[heating_replacement_col] + 
    df_detailed_combined[cooling_replacement_col]
)
```

Priority 5: Add Skip Logic for Non-Implemented End-Uses throughout the codebase (not included in EQUIPMENT_SPECS)

Priority 6: Document Next Steps

Create: /docs/REMDB_v4_Integration_Next_Steps.md

# REMDB v4 Integration - Remaining Work

## Phase 1: COMPLETED ✓
- Added REMDB v4 cost calculation functions
- Implemented cost scenario loops (3 scenarios: low/mid/high)
- Combined heating + cooling costs for heat pumps
- Skip logic for non-implemented end-uses

## Phase 2: TODO - Propagate Through Workflow


### Update Dataframe dictionary name and loop structure:

Update from DATAFRAMES_MPX_RCM_DISCOUNT_RATE to DATAFRAMES_MPX_SCENARIOS and add another loop layer for the capital cost assumptions. This will impact the function calls for:
- Rebate calculations
- Private NPV
- Public NPV
- Total NPV
- Adoption Potential
- Export model run results
- Load model run results

SUCCESS CRITERIA

✓ REMDB v4 files in proper locations
✓ Constants defined (COST_SCENARIO_KEYS, IMPLEMENTED_ENDUSES)
✓ Capital costs loop implemented in scenario notebook
✓ Results stored in nested dictionary: CAPITAL_COSTS_MPX[end_use][cost_type][scenario_key]
✓ Heating + cooling costs combined correctly
✓ Code runs without errors for heating and cooling

IMPORTANT NOTES

Dictionary Structure:

The dataframes in this dictionary should contain all relevant columns for someone to walk through the calculation left to right (e.g., bldg_id as index and columns: state, county, census division, heating load, equipment specs, columns used for cost calculations, replacement cost, upgrade cost, net capital cost).
```
CAPITAL_COSTS_MPX = {
    'heating': {
        'replacement': {
            'remdb_v3': df_heating_replacement_v3,
            'remdb_v4_low': df_heating_replacement_v4_low,
            'remdb_v4_mid': df_heating_replacement_v4_mid,
            'remdb_v4_high': df_heating_replacement_v4_high
        },
        'upgrade': {
            'remdb_v3': df_heating_upgrade_v3,
            'remdb_v4_low': df_heating_upgrade_v4_low,
            'remdb_v4_mid': df_heating_upgrade_v4_mid,
            'remdb_v4_high': df_heating_upgrade_v4_high
        }
    },
    'cooling': {
        'replacement': {...},
        'upgrade': {...}
    }
}
```

The installed upgrade cost, replacement cost, and net capital costs are the only columns that should be stored in the main dataframes. The private NPV function can be updated to to access and obtain these costs from the dictionaries (similar to what is shown below).

Access Pattern:
```
# Get heating upgrade costs for mid scenario
df = CAPITAL_COSTS_MPX['heating']['upgrade']['remdb_v4_mid']

# Get specific cost column
heating_cost = df['mp8_heating_upgrade_installed_cost']
```

QUESTIONS BEFORE STARTING

1. Which files are in remdb_v4_update/? (List them first)
2. Should REMDB functions stay in private_impact/ or move to utils/?
3. Are there any existing cost calculation functions to preserve?
4. What's the current structure of the CAPITAL COSTS section in the notebook?

Please start by reviewing the remdb_v4_update/ folder and proposing the file organization before implementing the loop.