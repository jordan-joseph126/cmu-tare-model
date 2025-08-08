# Private Impact Analysis Code Review Findings

## Section A: Lifetime Fuel Costs Module

### Implementation of the 5-step validation framework

#### Step 1: Mask Initialization with `initialize_validation_tracking()`
- Function call: `df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(df_copy, category, menu_mp, verbose=verbose)`
- Creates a boolean Series (`valid_mask`) indicating which homes have valid data
- Initializes dictionaries for tracking columns to mask
- Accounts for both data quality and retrofit status

#### Step 2: Series Initialization with `create_retrofit_only_series()`
- Function call: `fuel_costs_template = create_retrofit_only_series(df_copy, valid_mask)`
- Creates a template Series with zeros for valid homes and NaN for invalid homes
- Used as a foundation for yearly fuel cost calculations

#### Step 3: Valid-Only Calculation for fuel costs
- Passes `valid_mask` to `calculate_annual_fuel_costs()`
- Inside `calculate_annual_fuel_costs()`, applies masking to consumption calculations:
  ```python
  if valid_mask is not None and not valid_mask.all():
      consumption = consumption.copy()
      consumption.loc[~valid_mask] = 0.0
  ```
- Ensures calculations are only performed for valid homes

#### Step 4: Valid-Only Updates using list-based collection
- Creates a list to store yearly costs: `yearly_costs_list = []`
- Appends annual costs to the list: `yearly_costs_list.append(annual_cost_values)`
- Combines at the end: `costs_df = pd.concat(yearly_costs_list, axis=1)`
- Sums to get lifetime costs: `lifetime_fuel_costs = costs_df.sum(axis=1)`
- This approach is more efficient than incremental updates

#### Step 5: Final Masking with `apply_final_masking()`
- For main DataFrame: `df_main = apply_temporary_validation_and_mask(df_copy, df_lifetime, all_columns_to_mask, verbose=verbose)`
- For detailed DataFrame: `df_detailed = apply_final_masking(df_detailed, all_columns_to_mask, verbose=verbose)`
- Ensures consistent masking across all result columns

### Key computational logic

#### Validation-aware calculations in `calculate_annual_fuel_costs`
- Function accepts a `valid_mask` parameter to filter calculations
- Only performs calculations for valid homes, improving efficiency
- Returns properly masked results 

#### List-based collection for yearly costs
- Avoids inefficient incremental updates to DataFrame
- Collects all yearly costs in a list, then combines at the end
- Improves performance for large DataFrames
- Enables vectorized operations

#### Handling of avoided costs
- For measure packages (menu_mp != 0), calculates avoided costs at lifetime level
- Uses `calculate_avoided_values()` utility for consistency with other modules
- Properly masks avoided calculations
- Handles missing baseline cost data gracefully

#### Year and category looping structure
- Outer loop over categories and their lifetimes
- Inner loop over years within each category's lifetime
- Maintains category-specific validation masking
- Tracks columns for final masking

## Section B: Private Impact (NPV) Module

### Implementation of the 5-step validation framework

#### Step 1: Mask Initialization with `initialize_validation_tracking()`
- Function call: `df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(df_copy, category, menu_mp, verbose=verbose)`
- Used inside the category loop to get category-specific validation masks
- Properly initializes tracking dictionaries

#### Step 2: Series Initialization with `create_retrofit_only_series()`
- Function call: `discounted_savings_template = create_retrofit_only_series(df_measure_costs, valid_mask)`
- Initializes a template Series for discounted savings with zeros for valid homes and NaN for invalid homes
- Used as a fallback if no fuel cost data is found

#### Step 3: Valid-Only Calculation for NPV components
- Calculates capital costs with validation masking:
  ```python
  total_capital_cost, net_capital_cost = calculate_capital_costs(
      df_copy=df_copy,
      category=category,
      input_mp=input_mp,
      menu_mp=menu_mp,
      policy_scenario=policy_scenario,
      valid_mask=valid_mask
  )
  ```
- Uses `calculate_avoided_values()` for fuel cost savings with proper masking

#### Step 4: Valid-Only Updates using list-based collection
- Creates a list for yearly avoided costs: `yearly_avoided_costs = []`
- Appends to the list: `yearly_avoided_costs.append(avoided_costs)`
- Combines at the end: `avoided_costs_df = pd.concat(yearly_avoided_costs, axis=1)`
- Sums to get total: `total_discounted_savings = avoided_costs_df.sum(axis=1)`

#### Step 5: Final Masking
- Uses `apply_temporary_validation_and_mask()` to ensure consistent masking:
  ```python
  df_result = apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=verbose)
  ```

### Key computational logic

#### Capital cost masking approach
- Applies validation masking directly in the `calculate_capital_costs()` function:
  ```python
  total_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)
  net_capital_cost_masked = pd.Series(np.nan, index=df_copy.index)
  total_capital_cost_masked.loc[valid_mask] = total_capital_cost.loc[valid_mask]
  net_capital_cost_masked.loc[valid_mask] = net_capital_cost.loc[valid_mask]
  ```
- Returns properly masked cost Series

#### NPV calculation improvements
- Uses the `calculate_avoided_values()` utility for consistency
- Properly handles missing fuel cost data with informative warnings
- Returns a dictionary of columns instead of directly updating the DataFrame
- Includes numerical stability improvements with `replace_small_values_with_nan()`

#### Discounting implementation
- Pre-calculates discount factors for efficiency: 
  ```python
  discount_factors: Dict[int, float] = {}
  for year in range(1, max_lifetime + 1):
      year_label = year + (base_year - 1)
      discount_factors[year_label] = calculate_discount_factor(base_year, year_label, discounting_method)
  ```
- Applies discount factors to yearly avoided costs
- Tracks processed years for error reporting

#### Rebate application logic
- Different logic based on policy scenario:
  ```python
  if policy_scenario == 'No Inflation Reduction Act':
      # No rebates applied
  else:
      # Apply rebates based on category
      rebate_amount = df_copy[f'mp{menu_mp}_{category}_rebate_amount'].fillna(0)
      total_capital_cost = installation_cost - rebate_amount
  ```
- Special handling for weatherization costs in the heating category
- Properly handles missing rebate data with `fillna(0)`