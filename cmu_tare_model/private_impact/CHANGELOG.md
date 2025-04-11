# CHANGELOG

## April 9, 2025

### calculate_enclosure_upgrade_costs.py

#### Added
- Google-style docstrings to all functions:
  - `get_enclosure_parameters()`
  - `calculate_enclosure_retrofit_upgradeCosts()`
- Inline comments for normal distribution sampling calculations

### calculate_equipment_installation_costs.py

#### Added
- Google-style docstrings to all functions:
  - `obtain_heating_system_specs()`
  - `calculate_heating_installation_premium()`
  - `get_end_use_installation_parameters()`
  - `calculate_installation_cost_per_row()`
  - `calculate_installation_cost()`
- Type hints to improve code robustness and IDE support
- Inline comments explaining:
  - Normal distribution sampling logic
  - Cost component selection logic
  - Structure of the cost dictionary

### calculate_equipment_replacement_costs.py

#### Added
- Google-style docstrings to all functions, including:
  - Parameter types
  - Return types
  - Exception documentation
- Type hints to function signatures
- Inline comments explaining complex calculations covering:
  - Data manipulation logic
  - Normal distribution sampling

#### Maintained
- Original functionality to ensure consistent behavior

### calculate_lifetime_fuel_costs.py

#### Major Structural Changes
- Renamed and split functions:
  - Original: `calculate_annual_fuelCost()`
  - New: `calculate_lifetime_fuel_costs()` with helper function `calculate_annual_fuel_costs()`

#### Added
- Type hints for improved readability:
  ```python
  def calculate_lifetime_fuel_costs(
      df: pd.DataFrame,
      menu_mp: int,
      policy_scenario: str,
      df_baseline_costs: Optional[pd.DataFrame] = None
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  ```
- Comprehensive error handling:
  ```python
  try:
      # Validate required columns
      required_columns = ['state', 'census_division']
      missing_columns = [col for col in required_columns if col not in df_copy.columns]
      if missing_columns:
          raise KeyError(f"Required columns missing from input dataframe: {', '.join(missing_columns)}")
  except ValueError as e:
      raise ValueError(f"Invalid policy scenario: {policy_scenario}. {str(e)}")
  ```
- Return value for detailed annual results:
  ```python
  return df_main, df_detailed
  ```
- Helper function for annual calculations:
  ```python
  def calculate_annual_fuel_costs(
      df: pd.DataFrame,
      category: str,
      year_label: int,
      menu_mp: int,
      lookup_fuel_prices: Dict[str, Dict[str, Dict[str, Dict[int, float]]]],
      policy_scenario: str,
      scenario_prefix: str,
      is_elec_or_gas: Optional[pd.Series] = None
  ) -> Tuple[Dict[str, pd.Series], pd.Series]:
  ```

#### Modified
- Externalized constants:
  ```python
  from cmu_tare_model.constants import EQUIPMENT_SPECS, FUEL_MAPPING
  ```
- Encapsulated scenario parameter determination:
  ```python
  scenario_prefix, _, _, _, _, lookup_fuel_prices = define_scenario_params(
      menu_mp=menu_mp,
      policy_scenario=policy_scenario
  )
  ```
- DataFrame processing with explicit initialization:
  ```python
  df_copy = df.copy()
  df_detailed = pd.DataFrame(index=df_copy.index)
  ```
- Improved temporary column cleanup:
  ```python
  try:
      # Calculation logic
  finally:
      # Drop the temporary column after processing
      try:
          if '_temp_price' in df.columns:
              df.drop(columns=['_temp_price'], inplace=True)
      except:
          # Ignore errors in cleanup
          pass
  ```

#### Removed
- Sensitivity parameter (from `calculate_lifetime_fuel_costs_sensitivity.py`)
- Direct result column manipulation in favor of cleaner data handling

#### Performance Optimizations
- Modified fuel price lookup with more concise mechanism
- Added null safety:
  ```python
  # Get consumption data from the appropriate column (with null safety)
  consumption = df[consumption_col].fillna(0)
  ```

#### Data Verification
- Validated data consistency between versions:
  - `df_euss_am_baseline_home1` vs `df_euss_am_baseline_home2`: Exact match
  - `df_baseline_fuel_costs1` vs `df_baseline_fuel_costs2`: Exact match

## April 10, 2025

### calculate_lifetime_private_impact.py

Confirmed that updated program produces equivalent results

Starting equivalence testing between implementations...

========== TESTING CASE: menu_mp=8, input_mp=upgrade08 ==========
Creating test data for menu_mp=8...


Testing with policy scenario: No Inflation Reduction Act, menu_mp=8, input_mp=upgrade08
-- Scenario: No Inflation Reduction Act --
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'


Calculating costs for heating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for heating...
          lifetime: 15, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for waterHeating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for waterHeating...
          lifetime: 12, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for clothesDrying...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for clothesDrying...
          lifetime: 13, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for cooking...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for cooking...
          lifetime: 15, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act

-- Scenario: No Inflation Reduction Act -- 
              scenario_prefix: f'preIRA_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'


Determining lifetime private impacts for category: heating with lifetime: 15

Calculating costs for heating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for heating with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_


Determining lifetime private impacts for category: waterHeating with lifetime: 12

Calculating costs for waterHeating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for waterHeating with lifetime: 12 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_


Determining lifetime private impacts for category: clothesDrying with lifetime: 13

Calculating costs for clothesDrying...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for clothesDrying with lifetime: 13 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_


Determining lifetime private impacts for category: cooking with lifetime: 15

Calculating costs for cooking...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for cooking with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp8_


Comparing results for policy scenario: No Inflation Reduction Act

===== COMPARISON RESULTS =====
✓ Column 'bldg_id' matches between implementations
✓ Column 'mp8_clothesDrying_installationCost' matches between implementations
✓ Column 'mp8_clothesDrying_rebate_amount' matches between implementations
✓ Column 'mp8_clothesDrying_replacementCost' matches between implementations
✓ Column 'mp8_cooking_installationCost' matches between implementations
✓ Column 'mp8_cooking_rebate_amount' matches between implementations
✓ Column 'mp8_cooking_replacementCost' matches between implementations
✓ Column 'mp8_enclosure_upgradeCost' matches between implementations
✓ Column 'mp8_heating_installationCost' matches between implementations
✓ Column 'mp8_heating_installation_premium' matches between implementations
✓ Column 'mp8_heating_rebate_amount' matches between implementations
✓ Column 'mp8_heating_replacementCost' matches between implementations
✓ Column 'mp8_waterHeating_installationCost' matches between implementations
✓ Column 'mp8_waterHeating_rebate_amount' matches between implementations
✓ Column 'mp8_waterHeating_replacementCost' matches between implementations
✓ Column 'preIRA_mp8_clothesDrying_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp8_clothesDrying_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp8_clothesDrying_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp8_clothesDrying_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp8_cooking_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp8_cooking_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp8_cooking_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp8_cooking_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp8_heating_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp8_heating_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp8_heating_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp8_heating_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp8_waterHeating_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp8_waterHeating_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp8_waterHeating_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp8_waterHeating_total_capitalCost' matches between implementations
✓ Column 'weatherization_rebate_amount' matches between implementations

Summary: 32/32 columns match exactly between implementations
Equivalence rate: 100.00%

✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓


Testing with policy scenario: AEO2023 Reference Case, menu_mp=8, input_mp=upgrade08
-- Scenario: Inflation Reduction Act (IRA) Reference --
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'


Calculating costs for heating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for heating...
          lifetime: 15, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for waterHeating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for waterHeating...
          lifetime: 12, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for clothesDrying...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for clothesDrying...
          lifetime: 13, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for cooking...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for cooking...
          lifetime: 15, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case

-- Scenario: Inflation Reduction Act (IRA) Reference --
              scenario_prefix: 'iraRef_mp8_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'


Determining lifetime private impacts for category: heating with lifetime: 15

Calculating costs for heating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for heating with lifetime: 15 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp8_


Determining lifetime private impacts for category: waterHeating with lifetime: 12

Calculating costs for waterHeating...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for waterHeating with lifetime: 12 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp8_


Determining lifetime private impacts for category: clothesDrying with lifetime: 13

Calculating costs for clothesDrying...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for clothesDrying with lifetime: 13 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp8_


Determining lifetime private impacts for category: cooking with lifetime: 15

Calculating costs for cooking...
          input_mp: upgrade08, menu_mp: 8, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for cooking with lifetime: 15 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp8_


Comparing results for policy scenario: AEO2023 Reference Case

===== COMPARISON RESULTS =====
✓ Column 'bldg_id' matches between implementations
✓ Column 'iraRef_mp8_clothesDrying_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp8_clothesDrying_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp8_clothesDrying_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp8_clothesDrying_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp8_cooking_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp8_cooking_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp8_cooking_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp8_cooking_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp8_heating_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp8_heating_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp8_heating_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp8_heating_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp8_waterHeating_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp8_waterHeating_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp8_waterHeating_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp8_waterHeating_total_capitalCost' matches between implementations
✓ Column 'mp8_clothesDrying_installationCost' matches between implementations
✓ Column 'mp8_clothesDrying_rebate_amount' matches between implementations
✓ Column 'mp8_clothesDrying_replacementCost' matches between implementations
✓ Column 'mp8_cooking_installationCost' matches between implementations
✓ Column 'mp8_cooking_rebate_amount' matches between implementations
✓ Column 'mp8_cooking_replacementCost' matches between implementations
✓ Column 'mp8_enclosure_upgradeCost' matches between implementations
✓ Column 'mp8_heating_installationCost' matches between implementations
✓ Column 'mp8_heating_installation_premium' matches between implementations
✓ Column 'mp8_heating_rebate_amount' matches between implementations
✓ Column 'mp8_heating_replacementCost' matches between implementations
✓ Column 'mp8_waterHeating_installationCost' matches between implementations
✓ Column 'mp8_waterHeating_rebate_amount' matches between implementations
✓ Column 'mp8_waterHeating_replacementCost' matches between implementations
✓ Column 'weatherization_rebate_amount' matches between implementations

Summary: 32/32 columns match exactly between implementations
Equivalence rate: 100.00%

✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓


========== TESTING CASE: menu_mp=9, input_mp=upgrade09 ==========
Creating test data for menu_mp=9...


Testing with policy scenario: No Inflation Reduction Act, menu_mp=9, input_mp=upgrade09
-- Scenario: No Inflation Reduction Act --
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'


Calculating costs for heating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for heating...
          lifetime: 15, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for waterHeating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for waterHeating...
          lifetime: 12, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for clothesDrying...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for clothesDrying...
          lifetime: 13, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for cooking...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for cooking...
          lifetime: 15, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act

-- Scenario: No Inflation Reduction Act --
              scenario_prefix: f'preIRA_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'


Determining lifetime private impacts for category: heating with lifetime: 15

Calculating costs for heating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for heating with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp9_


Determining lifetime private impacts for category: waterHeating with lifetime: 12

Calculating costs for waterHeating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for waterHeating with lifetime: 12 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp9_


Determining lifetime private impacts for category: clothesDrying with lifetime: 13

Calculating costs for clothesDrying...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for clothesDrying with lifetime: 13 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp9_


Determining lifetime private impacts for category: cooking with lifetime: 15

Calculating costs for cooking...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for cooking with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp9_


Comparing results for policy scenario: No Inflation Reduction Act

===== COMPARISON RESULTS =====
✓ Column 'bldg_id' matches between implementations
✓ Column 'mp9_clothesDrying_installationCost' matches between implementations
✓ Column 'mp9_clothesDrying_rebate_amount' matches between implementations
✓ Column 'mp9_clothesDrying_replacementCost' matches between implementations
✓ Column 'mp9_cooking_installationCost' matches between implementations
✓ Column 'mp9_cooking_rebate_amount' matches between implementations
✓ Column 'mp9_cooking_replacementCost' matches between implementations
✓ Column 'mp9_enclosure_upgradeCost' matches between implementations
✓ Column 'mp9_heating_installationCost' matches between implementations
✓ Column 'mp9_heating_installation_premium' matches between implementations
✓ Column 'mp9_heating_rebate_amount' matches between implementations
✓ Column 'mp9_heating_replacementCost' matches between implementations
✓ Column 'mp9_waterHeating_installationCost' matches between implementations
✓ Column 'mp9_waterHeating_rebate_amount' matches between implementations
✓ Column 'mp9_waterHeating_replacementCost' matches between implementations
✓ Column 'preIRA_mp9_clothesDrying_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp9_clothesDrying_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp9_clothesDrying_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp9_clothesDrying_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp9_cooking_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp9_cooking_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp9_cooking_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp9_cooking_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp9_heating_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp9_heating_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp9_heating_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp9_heating_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp9_waterHeating_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp9_waterHeating_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp9_waterHeating_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp9_waterHeating_total_capitalCost' matches between implementations
✓ Column 'weatherization_rebate_amount' matches between implementations

Summary: 32/32 columns match exactly between implementations
Equivalence rate: 100.00%

✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓


Testing with policy scenario: AEO2023 Reference Case, menu_mp=9, input_mp=upgrade09
-- Scenario: Inflation Reduction Act (IRA) Reference --
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'


Calculating costs for heating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for heating...
          lifetime: 15, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for waterHeating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for waterHeating...
          lifetime: 12, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for clothesDrying...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for clothesDrying...
          lifetime: 13, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for cooking...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for cooking...
          lifetime: 15, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case

-- Scenario: Inflation Reduction Act (IRA) Reference --
              scenario_prefix: 'iraRef_mp9_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'


Determining lifetime private impacts for category: heating with lifetime: 15

Calculating costs for heating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for heating with lifetime: 15 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp9_


Determining lifetime private impacts for category: waterHeating with lifetime: 12

Calculating costs for waterHeating...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for waterHeating with lifetime: 12 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp9_


Determining lifetime private impacts for category: clothesDrying with lifetime: 13

Calculating costs for clothesDrying...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for clothesDrying with lifetime: 13 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp9_


Determining lifetime private impacts for category: cooking with lifetime: 15

Calculating costs for cooking...
          input_mp: upgrade09, menu_mp: 9, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for cooking with lifetime: 15 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp9_


Comparing results for policy scenario: AEO2023 Reference Case

===== COMPARISON RESULTS =====
✓ Column 'bldg_id' matches between implementations
✓ Column 'iraRef_mp9_clothesDrying_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp9_clothesDrying_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp9_clothesDrying_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp9_clothesDrying_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp9_cooking_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp9_cooking_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp9_cooking_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp9_cooking_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp9_heating_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp9_heating_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp9_heating_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp9_heating_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp9_waterHeating_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp9_waterHeating_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp9_waterHeating_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp9_waterHeating_total_capitalCost' matches between implementations
✓ Column 'mp9_clothesDrying_installationCost' matches between implementations
✓ Column 'mp9_clothesDrying_rebate_amount' matches between implementations
✓ Column 'mp9_clothesDrying_replacementCost' matches between implementations
✓ Column 'mp9_cooking_installationCost' matches between implementations
✓ Column 'mp9_cooking_rebate_amount' matches between implementations
✓ Column 'mp9_cooking_replacementCost' matches between implementations
✓ Column 'mp9_enclosure_upgradeCost' matches between implementations
✓ Column 'mp9_heating_installationCost' matches between implementations
✓ Column 'mp9_heating_installation_premium' matches between implementations
✓ Column 'mp9_heating_rebate_amount' matches between implementations
✓ Column 'mp9_heating_replacementCost' matches between implementations
✓ Column 'mp9_waterHeating_installationCost' matches between implementations
✓ Column 'mp9_waterHeating_rebate_amount' matches between implementations
✓ Column 'mp9_waterHeating_replacementCost' matches between implementations
✓ Column 'weatherization_rebate_amount' matches between implementations

Summary: 32/32 columns match exactly between implementations
Equivalence rate: 100.00%

✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓


========== TESTING CASE: menu_mp=10, input_mp=upgrade10 ==========
Creating test data for menu_mp=10...


Testing with policy scenario: No Inflation Reduction Act, menu_mp=10, input_mp=upgrade10
-- Scenario: No Inflation Reduction Act --
              scenario_prefix: f'preIRA_mp10_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'


Calculating costs for heating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for heating...
          lifetime: 15, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for waterHeating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for waterHeating...
          lifetime: 12, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for clothesDrying...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for clothesDrying...
          lifetime: 13, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act


Calculating costs for cooking...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act

Calculating Private NPV for cooking...
          lifetime: 15, interest_rate: 0.07, policy_scenario: No Inflation Reduction Act

-- Scenario: No Inflation Reduction Act --
              scenario_prefix: f'preIRA_mp10_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_preIRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_preIRA'


Determining lifetime private impacts for category: heating with lifetime: 15

Calculating costs for heating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for heating with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp10_


Determining lifetime private impacts for category: waterHeating with lifetime: 12

Calculating costs for waterHeating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for waterHeating with lifetime: 12 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp10_


Determining lifetime private impacts for category: clothesDrying with lifetime: 13

Calculating costs for clothesDrying...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for clothesDrying with lifetime: 13 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp10_


Determining lifetime private impacts for category: cooking with lifetime: 15

Calculating costs for cooking...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: No Inflation Reduction Act
Calculating Private NPV for cooking with lifetime: 15 years
          policy_scenario: No Inflation Reduction Act --> scenario_prefix: preIRA_mp10_


Comparing results for policy scenario: No Inflation Reduction Act

===== COMPARISON RESULTS =====
✓ Column 'bldg_id' matches between implementations
✓ Column 'mp10_clothesDrying_installationCost' matches between implementations
✓ Column 'mp10_clothesDrying_rebate_amount' matches between implementations
✓ Column 'mp10_clothesDrying_replacementCost' matches between implementations
✓ Column 'mp10_cooking_installationCost' matches between implementations
✓ Column 'mp10_cooking_rebate_amount' matches between implementations
✓ Column 'mp10_cooking_replacementCost' matches between implementations
✓ Column 'mp10_enclosure_upgradeCost' matches between implementations
✓ Column 'mp10_heating_installationCost' matches between implementations
✓ Column 'mp10_heating_installation_premium' matches between implementations
✓ Column 'mp10_heating_rebate_amount' matches between implementations
✓ Column 'mp10_heating_replacementCost' matches between implementations
✓ Column 'mp10_waterHeating_installationCost' matches between implementations
✓ Column 'mp10_waterHeating_rebate_amount' matches between implementations
✓ Column 'mp10_waterHeating_replacementCost' matches between implementations
✓ Column 'preIRA_mp10_clothesDrying_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp10_clothesDrying_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp10_clothesDrying_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp10_clothesDrying_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp10_cooking_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp10_cooking_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp10_cooking_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp10_cooking_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp10_heating_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp10_heating_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp10_heating_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp10_heating_total_capitalCost' matches between implementations
✓ Column 'preIRA_mp10_waterHeating_net_capitalCost' matches between implementations
✓ Column 'preIRA_mp10_waterHeating_private_npv_lessWTP' matches between implementations
✓ Column 'preIRA_mp10_waterHeating_private_npv_moreWTP' matches between implementations
✓ Column 'preIRA_mp10_waterHeating_total_capitalCost' matches between implementations
✓ Column 'weatherization_rebate_amount' matches between implementations

Summary: 32/32 columns match exactly between implementations
Equivalence rate: 100.00%

✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓


Testing with policy scenario: AEO2023 Reference Case, menu_mp=10, input_mp=upgrade10
-- Scenario: Inflation Reduction Act (IRA) Reference --
              scenario_prefix: 'iraRef_mp10_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'


Calculating costs for heating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for heating...
          lifetime: 15, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for waterHeating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for waterHeating...
          lifetime: 12, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for clothesDrying...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for clothesDrying...
          lifetime: 13, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case


Calculating costs for cooking...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case

Calculating Private NPV for cooking...
          lifetime: 15, interest_rate: 0.07, policy_scenario: AEO2023 Reference Case

-- Scenario: Inflation Reduction Act (IRA) Reference --
              scenario_prefix: 'iraRef_mp10_', cambium_scenario: 'MidCase', lookup_emissions_fossil_fuel: 'lookup_emissions_fossil_fuel',
              lookup_emissions_electricity_climate: 'lookup_emissions_electricity_climate_IRA', lookup_emissions_electricity_health: 'lookup_emissions_electricity_health', lookup_fuel_prices: 'lookup_fuel_prices_iraRef'


Determining lifetime private impacts for category: heating with lifetime: 15

Calculating costs for heating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for heating with lifetime: 15 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp10_


Determining lifetime private impacts for category: waterHeating with lifetime: 12

Calculating costs for waterHeating...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for waterHeating with lifetime: 12 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp10_


Determining lifetime private impacts for category: clothesDrying with lifetime: 13

Calculating costs for clothesDrying...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for clothesDrying with lifetime: 13 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp10_


Determining lifetime private impacts for category: cooking with lifetime: 15

Calculating costs for cooking...
          input_mp: upgrade10, menu_mp: 10, policy_scenario: AEO2023 Reference Case
Calculating Private NPV for cooking with lifetime: 15 years
          policy_scenario: AEO2023 Reference Case --> scenario_prefix: iraRef_mp10_


Comparing results for policy scenario: AEO2023 Reference Case

===== COMPARISON RESULTS =====
✓ Column 'bldg_id' matches between implementations
✓ Column 'iraRef_mp10_clothesDrying_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp10_clothesDrying_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp10_clothesDrying_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp10_clothesDrying_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp10_cooking_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp10_cooking_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp10_cooking_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp10_cooking_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp10_heating_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp10_heating_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp10_heating_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp10_heating_total_capitalCost' matches between implementations
✓ Column 'iraRef_mp10_waterHeating_net_capitalCost' matches between implementations
✓ Column 'iraRef_mp10_waterHeating_private_npv_lessWTP' matches between implementations
✓ Column 'iraRef_mp10_waterHeating_private_npv_moreWTP' matches between implementations
✓ Column 'iraRef_mp10_waterHeating_total_capitalCost' matches between implementations
✓ Column 'mp10_clothesDrying_installationCost' matches between implementations
✓ Column 'mp10_clothesDrying_rebate_amount' matches between implementations
✓ Column 'mp10_clothesDrying_replacementCost' matches between implementations
✓ Column 'mp10_cooking_installationCost' matches between implementations
✓ Column 'mp10_cooking_rebate_amount' matches between implementations
✓ Column 'mp10_cooking_replacementCost' matches between implementations
✓ Column 'mp10_enclosure_upgradeCost' matches between implementations
✓ Column 'mp10_heating_installationCost' matches between implementations
✓ Column 'mp10_heating_installation_premium' matches between implementations
✓ Column 'mp10_heating_rebate_amount' matches between implementations
✓ Column 'mp10_heating_replacementCost' matches between implementations
✓ Column 'mp10_waterHeating_installationCost' matches between implementations
✓ Column 'mp10_waterHeating_rebate_amount' matches between implementations
✓ Column 'mp10_waterHeating_replacementCost' matches between implementations
✓ Column 'weatherization_rebate_amount' matches between implementations

Summary: 32/32 columns match exactly between implementations
Equivalence rate: 100.00%

✓✓✓ BOTH IMPLEMENTATIONS ARE FUNCTIONALLY EQUIVALENT ✓✓✓


✅ ALL TESTS PASSED! Both implementations are functionally equivalent across all test cases.