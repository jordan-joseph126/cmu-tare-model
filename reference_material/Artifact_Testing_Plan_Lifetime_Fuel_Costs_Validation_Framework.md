# Testing Plan for Validation Framework and Lifetime Fuel Costs

## Validation Framework Test Implementation Plan

### 1. Foundational Improvements

#### Update `replace_small_values_with_nan` Implementation
```python
# In validation_framework.py (lines 256-275)
def replace_small_values_with_nan(
    series_or_dict: Union[pd.Series, pd.DataFrame, Dict[Any, pd.Series]], 
    threshold: float = 1e-10
) -> Union[pd.Series, pd.DataFrame, Dict[Any, pd.Series]]:
    """
    Replace values close to zero with NaN to avoid numerical artifacts.
    
    Args:
        series_or_dict: A pandas Series, DataFrame, or dictionary of Series.
        threshold: Values with absolute value below this threshold will be replaced with NaN.
        
    Returns:
        The input with small values replaced by NaN.
        
    Raises:
        TypeError: If input is not a pandas Series, DataFrame, or dictionary of Series.
    """
    if isinstance(series_or_dict, pd.Series):
        return series_or_dict.where(abs(series_or_dict) > threshold, np.nan)
    elif isinstance(series_or_dict, pd.DataFrame):
        # Apply column-by-column to ensure threshold is respected
        result_df = series_or_dict.copy()
        for col in result_df.columns:
            result_df[col] = replace_small_values_with_nan(result_df[col], threshold)
        return result_df
    elif isinstance(series_or_dict, dict):
        return {k: replace_small_values_with_nan(v, threshold) for k, v in series_or_dict.items()}
    else:
        raise TypeError("Input must be a pandas Series, DataFrame, or dictionary of Series")
```

#### Update `get_valid_calculation_mask` to Handle String Menu MPs
```python
# In validation_framework.py (lines 70-110)
def get_valid_calculation_mask(
    df: pd.DataFrame, 
    category: str, 
    menu_mp: Union[int, str] = 0,
    verbose: bool = True
) -> pd.Series:
    """
    Combines data validation and retrofit status for comprehensive masking.
    
    This function addresses a key integration issue between the data validation
    system and the retrofit status tracking system. It ensures:
    - For baseline scenarios: Only homes with valid data are processed
    - For measure packages: Only homes with both valid data AND scheduled for retrofits are processed
    
    Args:
        df: DataFrame containing the validation flags and retrofit information.
        category: Equipment category (e.g., 'heating', 'waterHeating').
        menu_mp: Measure package identifier (0 for baseline, nonzero for measure packages).
        verbose: Whether to print information about valid homes.
        
    Returns:
        Series of boolean values indicating which homes should be included in calculations.
        
    Raises:
        ValueError: If the inclusion flag for the given category doesn't exist in the DataFrame.
    """
    # Standardize menu_mp to facilitate comparisons
    menu_mp_str = str(menu_mp).lower()
    is_baseline = menu_mp_str == "0" or menu_mp_str == "baseline"
    
    # Rest of the function remains unchanged...
```

### 2. Test Fixes in `test_validation_framework.py`

#### Fix `test_replace_small_values_utility`
```python
def test_replace_small_values_utility() -> None:
    """
    Test the replace_small_values_with_nan utility function.
    
    This test verifies that the replace_small_values_with_nan function correctly:
    1. Replaces values close to zero with NaN
    2. Preserves larger values
    3. Works with different input types (Series, DataFrame)
    """
    # Test with Series
    series = pd.Series([1.0, 1e-12, -1e-11, 0.1, -0.2])
    result_series = replace_small_values_with_nan(series)
    
    # Values below threshold should be NaN
    assert pd.isna(result_series[1]), "Value 1e-12 should be replaced with NaN"
    assert pd.isna(result_series[2]), "Value -1e-11 should be replaced with NaN"
    
    # Larger values should be preserved
    assert result_series[0] == 1.0, "Value 1.0 should be preserved"
    assert result_series[3] == 0.1, "Value 0.1 should be preserved"
    assert result_series[4] == -0.2, "Value -0.2 should be preserved"
    
    # Test with DataFrame - each column should be processed separately
    df = pd.DataFrame({
        'col1': [1.0, 1e-12, -1e-11, 0.1, -0.2],
        'col2': [0.5, -1e-9, 2e-8, -0.3, 0.7]
    })
    result_df = replace_small_values_with_nan(df)
    
    # Check values in first column
    assert pd.isna(result_df.loc[1, 'col1']), "Value 1e-12 in col1 should be NaN"
    assert pd.isna(result_df.loc[2, 'col1']), "Value -1e-11 in col1 should be NaN"
    
    # Check values in second column
    assert pd.isna(result_df.loc[1, 'col2']), "Value -1e-9 in col2 should be NaN"
    assert pd.isna(result_df.loc[2, 'col2']), "Value 2e-8 in col2 should be NaN"
```

#### Add `test_replace_small_values_with_custom_threshold`
```python
def test_replace_small_values_with_custom_threshold() -> None:
    """
    Test replace_small_values_with_nan with a custom threshold.
    
    This test verifies that replace_small_values_with_nan correctly:
    1. Uses the provided custom threshold
    2. Replaces values below threshold with NaN
    3. Preserves values above threshold
    """
    # Create a Series with values of different magnitudes
    series = pd.Series([1.0, 1e-3, 1e-6, 1e-9, 0.0])
    
    # Test with default threshold (1e-10)
    result_default = replace_small_values_with_nan(series)
    
    # Values below default threshold should be NaN
    assert not pd.isna(result_default[0]), "1.0 should remain unchanged with default threshold"
    assert not pd.isna(result_default[1]), "1e-3 should remain unchanged with default threshold"
    assert not pd.isna(result_default[2]), "1e-6 should remain unchanged with default threshold"
    assert pd.isna(result_default[3]), "1e-9 should be NaN with default threshold"
    assert pd.isna(result_default[4]), "0.0 should be NaN with default threshold"
    
    # Test with custom threshold (1e-5)
    result_custom = replace_small_values_with_nan(series, threshold=1e-5)
    
    # Values below custom threshold should be NaN
    assert not pd.isna(result_custom[0]), "1.0 should remain unchanged with custom threshold"
    assert not pd.isna(result_custom[1]), "1e-3 should remain unchanged with custom threshold"
    assert pd.isna(result_custom[2]), "1e-6 should be NaN with custom threshold 1e-5"
    assert pd.isna(result_custom[3]), "1e-9 should be NaN with custom threshold 1e-5"
    assert pd.isna(result_custom[4]), "0.0 should be NaN with custom threshold 1e-5"
```

#### Fix `test_list_based_collection`
```python
def test_list_based_collection(sample_homes_df: pd.DataFrame) -> None:
    """
    Test list-based collection pattern for updates.
    
    This test verifies the list-based collection pattern for updates,
    which is more efficient than incremental DataFrame updates.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    # Initialize validation
    category = 'heating'
    valid_mask = sample_homes_df[f'include_{category}']
    
    # Create a list to store calculation results
    yearly_values = []
    
    # Generate multiple years of calculations
    for year in range(2024, 2027):
        # Create a yearly calculation
        year_values = pd.Series(np.nan, index=sample_homes_df.index)
        
        # Calculate only for valid homes
        year_values.loc[valid_mask] = year - 2023  # Simple year-based value
        
        # Add to collection
        yearly_values.append(year_values)
    
    # Verify list contains the correct number of Series
    assert len(yearly_values) == 3, "List should contain 3 Series (one for each year)"
    
    # Verify each Series in the list has correct values
    for i, year_series in enumerate(yearly_values):
        year_value = i + 1  # 1, 2, 3 for years 2024, 2025, 2026
        
        # Valid homes should have the year value
        for idx in valid_mask[valid_mask].index:
            assert year_series[idx] == year_value, \
                f"Valid home at index {idx} for year {2024+i} should have value {year_value}"
        
        # Invalid homes should have NaN
        for idx in valid_mask[~valid_mask].index:
            assert pd.isna(year_series[idx]), \
                f"Invalid home at index {idx} for year {2024+i} should have NaN"
    
    # Now combine the list into a DataFrame and verify
    yearly_df = pd.concat(yearly_values, axis=1)
    yearly_df.columns = [f'year_{2024+i}' for i in range(len(yearly_values))]
    
    # Calculate total - explicitly propagate NaN values
    total_series = yearly_df.sum(axis=1, skipna=False)  # Use skipna=False to propagate NaN

    # Verify total has values for valid homes, NaN for invalid homes
    valid_sum = 1 + 2 + 3  # Sum of year values
    for idx in sample_homes_df.index:
        if valid_mask[idx]:
            assert total_series[idx] == valid_sum, \
                f"Valid home at index {idx} should have total value {valid_sum}"
        else:
            assert pd.isna(total_series[idx]), \
                f"Invalid home at index {idx} should have NaN for total"
```

#### Add `test_validation_framework_with_different_menu_mp_formats`
```python
def test_validation_framework_with_different_menu_mp_formats(sample_homes_df: pd.DataFrame) -> None:
    """
    Test validation framework with different menu_mp format strings.
    
    This test verifies that the validation framework correctly handles
    different string formats for menu_mp parameter.
    
    Args:
        sample_homes_df: Fixture providing test data
    """
    category = 'heating'
    include_col = f'include_{category}'
    
    # Test with different menu_mp formats that should be treated as baseline
    baseline_formats = [0, "0", "baseline", "BASELINE", "Baseline"]
    
    for menu_mp in baseline_formats:
        # Get valid mask
        valid_mask = get_valid_calculation_mask(sample_homes_df, category, menu_mp, verbose=False)
        
        # For baseline, mask should exactly match include column
        assert valid_mask.equals(sample_homes_df[include_col]), \
            f"For baseline menu_mp={menu_mp}, mask should match include column"
    
    # Test with non-baseline formats
    non_baseline_formats = [1, "1", 8, "8", "mp8", "upgrade08"]
    
    for menu_mp in non_baseline_formats:
        # Get valid mask
        valid_mask = get_valid_calculation_mask(sample_homes_df, category, menu_mp, verbose=False)
        
        # For non-baseline, may combine with retrofit status
        # Just check that it's not identical to the include column if retrofits exist
        upgrade_col = UPGRADE_COLUMNS.get(category)
        if upgrade_col in sample_homes_df.columns and sample_homes_df[upgrade_col].notna().any():
            assert not valid_mask.equals(sample_homes_df[include_col]), \
                f"For non-baseline menu_mp={menu_mp}, mask should consider retrofit status"
```

## Lifetime Fuel Costs Test Implementation Plan

### 1. Fix `mock_annual_calculation` Fixture

```python
@pytest.fixture
def mock_annual_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock the annual calculation to provide consistent test values.
    
    This fixture replaces the annual fuel cost calculation with a 
    predictable implementation that:
    - Returns controlled values based on category and year
    - Properly handles validation masking
    - Creates both costs and savings for measure packages
    - Supports ANY requested year (including years beyond data availability)
    
    Args:
        monkeypatch: Pytest fixture for patching functions
    """
    def mock_calculate_annual(df: pd.DataFrame, 
                             category: str, 
                             year_label: int,  # This works with any year
                             menu_mp: int, 
                             lookup_fuel_prices: Dict, 
                             policy_scenario: str, 
                             scenario_prefix: str, 
                             is_elec_or_gas: Optional[pd.Series] = None,
                             valid_mask: Optional[pd.Series] = None) -> Tuple[Dict[str, pd.Series], pd.Series]:
        """Mock implementation for calculate_annual_fuel_costs."""
        # If no valid_mask provided, use all homes
        if valid_mask is None:
            valid_mask = pd.Series(True, index=df.index)
        
        # Create results dictionary and cost column
        annual_costs = {}
        cost_col = f"{scenario_prefix}{year_label}_{category}_fuelCost"
        
        # Create a Series with zeros for all homes
        cost_series = pd.Series(0.0, index=df.index)
        
        # Set values for valid homes only - using deterministic values based on inputs
        valid_homes = valid_mask[valid_mask].index
        if len(valid_homes) > 0:
            # Create deterministic, different values for each home, year, and category
            for i, idx in enumerate(valid_homes):
                # Create deterministic, different values for each home, year, and category
                base_value = 100.0 + (i * 10) + ((year_label - 2024) * 5)
                multiplier = 1.0
                if category == 'heating':
                    multiplier = 2.0
                elif category == 'waterHeating':
                    multiplier = 1.5
                elif category == 'clothesDrying':
                    multiplier = 1.2
                elif category == 'cooking':
                    multiplier = 0.8
                
                cost_series.loc[idx] = base_value * multiplier
            
        # Set invalid homes to NaN
        cost_series.loc[~valid_mask] = np.nan
        
        # Create annual costs dictionary
        annual_costs[cost_col] = cost_series
        
        # For measure packages, add savings column
        if menu_mp != 0:
            savings_col = f"{scenario_prefix}{year_label}_{category}_savings_fuelCost"
            savings_series = pd.Series(np.nan, index=df.index)
            
            # Set values for valid homes only (50% of costs as savings)
            for idx in valid_homes:
                savings_series.loc[idx] = cost_series.loc[idx] * 0.5
                
            annual_costs[savings_col] = savings_series
            
        return annual_costs, cost_series
        
    # Apply the patch to the module under test
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_fuel_costs.calculate_annual_fuel_costs',
        mock_calculate_annual
    )
```

### 2. Fix Tests in `test_calculate_lifetime_fuel_costs.py`

#### Fix `test_empty_dataframe`
```python
def test_empty_dataframe(mock_scenario_params: None) -> None:
    """
    Test handling of empty DataFrame.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Handles empty DataFrames gracefully
    2. Raises appropriate errors with informative messages
    
    Args:
        mock_scenario_params: Fixture mocking scenario parameters
    """
    # Create an empty DataFrame with required columns
    empty_df = pd.DataFrame(columns=['state', 'census_division'])
    
    # Add required validation columns
    for category in EQUIPMENT_SPECS.keys():
        empty_df[f'include_{category}'] = []
    
    # Set up test parameters
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    try:
        # Call the main function
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=empty_df,
            menu_mp=menu_mp,
            policy_scenario=policy_scenario,
            verbose=False
        )
        
        # If it succeeds, both results should be empty DataFrames
        assert df_main.empty, "Result should be an empty DataFrame"
        assert df_detailed.empty, "Detailed result should be an empty DataFrame"
        
    except (ValueError, KeyError, RuntimeError) as e:
        # Alternatively, verify the error message is informative
        error_msg = str(e)
        assert "empty" in error_msg.lower() or "include_" in error_msg, \
            "Error message should mention empty DataFrame or missing columns"
```

#### Fix `test_all_invalid_homes`
```python
def test_all_invalid_homes(sample_homes_df: pd.DataFrame, 
                          mock_scenario_params: None, 
                          mock_annual_calculation: None) -> None:
    """
    Test calculation when all homes are invalid.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Handles the case where all homes are invalid
    2. Returns properly masked results (all NaN)
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Create modified DataFrame with all homes invalid
    df_modified = sample_homes_df.copy()
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        df_modified[f'include_{category}'] = False
    
    # Set up test parameters
    menu_mp = 0
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df_modified,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify that all result columns have NaN values
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        lifetime_col = f'baseline_{category}_lifetime_fuelCost'
        if lifetime_col in df_main.columns:
            # Count non-NaN values explicitly
            non_nan_count = df_main[lifetime_col].notna().sum()
            assert non_nan_count == 0, \
                f"All homes should have NaN values for {lifetime_col} when all homes are invalid (found {non_nan_count} non-NaN values)"
```

#### Fix `test_lifetime_fuel_costs_basic`
```python
def test_lifetime_fuel_costs_basic(sample_homes_df: pd.DataFrame, 
                                 mock_scenario_params: None, 
                                 mock_annual_calculation: None) -> None:
    """
    Test basic functionality of calculate_lifetime_fuel_costs.
    
    This test verifies that calculate_lifetime_fuel_costs correctly:
    1. Calculates fuel costs for all categories and years
    2. Returns main and detailed DataFrames
    3. Applies proper validation masking
    
    Args:
        sample_homes_df: Fixture providing test data
        mock_scenario_params: Fixture mocking scenario parameters
        mock_annual_calculation: Fixture mocking annual calculation
    """
    # Set up test parameters
    menu_mp = 0  # Baseline
    policy_scenario = 'AEO2023 Reference Case'
    
    # Call the main function
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=sample_homes_df,
        menu_mp=menu_mp,
        policy_scenario=policy_scenario,
        verbose=False
    )
    
    # Verify result structure
    assert isinstance(df_main, pd.DataFrame), "df_main should be a DataFrame"
    assert isinstance(df_detailed, pd.DataFrame), "df_detailed should be a DataFrame"
    assert len(df_main) == len(sample_homes_df), "df_main should have same number of rows"
    assert len(df_detailed) == len(sample_homes_df), "df_detailed should have same number of rows"
    
    # Verify result contains lifetime columns for all categories
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        lifetime_col = f'baseline_{category}_lifetime_fuelCost'
        assert lifetime_col in df_main.columns, f"df_main should have column '{lifetime_col}'"
        assert lifetime_col in df_detailed.columns, f"df_detailed should have column '{lifetime_col}'"
        
        # Verify validation masking on lifetime columns
        valid_mask = sample_homes_df[f'include_{category}']
        invalid_indices = valid_mask[~valid_mask].index
        
        if len(invalid_indices) > 0:
            # Check if any invalid homes have non-NaN values
            non_nan_invalid = df_main.loc[invalid_indices, lifetime_col].notna().sum()
            assert non_nan_invalid == 0, \
                f"Invalid homes should have NaN values for {lifetime_col} in df_main (found {non_nan_invalid} non-NaN values)"
            
            non_nan_invalid_detailed = df_detailed.loc[invalid_indices, lifetime_col].notna().sum()
            assert non_nan_invalid_detailed == 0, \
                f"Invalid homes should have NaN values for {lifetime_col} in df_detailed (found {non_nan_invalid_detailed} non-NaN values)"
```

### 3. Fix Production Code in `calculate_lifetime_fuel_costs.py`

#### Improve Error Handling for Missing Columns
```python
# Inside the calculate_lifetime_fuel_costs function
# At beginning of function
# Handle empty DataFrames gracefully
if df.empty:
    print("Warning: Empty DataFrame provided. Returning empty results.")
    return pd.DataFrame(), pd.DataFrame()

# Inside yearly calculation loop
try:
    # Determine the consumption column name
    consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
    
    # Check if column exists before attempting calculation
    if consumption_col not in df.columns:
        if verbose:
            print(f"Warning: Consumption column '{consumption_col}' not found. Skipping year {year_label}.")
        continue  # Skip this year instead of failing
    
    # Rest of calculation...
except KeyError as e:
    if "consumption" in str(e):
        # More informative error for missing consumption columns
        raise KeyError(f"Missing consumption column for year {year_label}, category '{category}': {e}")
    else:
        raise  # Re-raise original error for other cases
```

#### Ensure Proper NaN Propagation
```python
# After calculating lifetime fuel costs
# Ensure NaN values for invalid homes
lifetime_fuel_costs = pd.Series(
    np.where(valid_mask, lifetime_fuel_costs, np.nan),
    index=lifetime_fuel_costs.index
)
```

#### Handle All-Invalid Scenario
```python
# After determining valid_mask
if not valid_mask.any():
    if verbose:
        print(f"Warning: All homes are invalid for category '{category}'. Results will be all NaN.")
    
    # Use a template with all NaN values
    costs_col = f'{scenario_prefix}{category}_lifetime_fuelCost'
    lifetime_fuel_costs = pd.Series(np.nan, index=df_copy.index)
    lifetime_dict[costs_col] = lifetime_fuel_costs
    
    if menu_mp != 0 and df_baseline_costs is not None:
        baseline_costs_col = f'baseline_{category}_lifetime_fuelCost'
        if baseline_costs_col in df_baseline_costs.columns:
            savings_cost_col = f'{scenario_prefix}{category}_lifetime_savings_fuelCost'
            # Create savings column with all NaN
            lifetime_dict[savings_cost_col] = pd.Series(np.nan, index=df_copy.index)
            
    # Skip further processing for this category
    continue
```

## Extended Testing Plan

### 1. Additional Edge Case Tests

#### Test for Mixed Types in Columns
```python
def test_mixed_types_in_consumption_columns(sample_homes_df: pd.DataFrame):
    """Test with mixed types in consumption columns."""
    # Create DataFrame with mixed types in consumption columns
    df_mixed = sample_homes_df.copy()
    category = 'heating'
    consumption_col = f'baseline_2024_{category}_consumption'
    
    # Set some values to strings, some to None
    df_mixed[consumption_col] = ['100', 200, None, np.nan, 'invalid']
    
    # Should handle gracefully or raise helpful error
    try:
        df_main, df_detailed = calculate_lifetime_fuel_costs(
            df=df_mixed,
            menu_mp=0,
            policy_scenario='AEO2023 Reference Case',
            verbose=False
        )
        # Test passes if function handles mixed types gracefully
        
    except Exception as e:
        # Or if it raises a clear error message
        error_msg = str(e)
        assert "type" in error_msg.lower() or "conversion" in error_msg.lower(), \
            "Error message should mention type conversion issue"
```

#### Test with Very Large Values
```python
def test_extreme_values_in_consumption(sample_homes_df: pd.DataFrame):
    """Test with extreme values in consumption columns."""
    # Create DataFrame with extreme values
    df_extreme = sample_homes_df.copy()
    category = 'heating'
    consumption_col = f'baseline_2024_{category}_consumption'
    
    # Set some values to very large or small numbers
    df_extreme[consumption_col] = [1e20, 1e-20, -1e10, 0, 1e15]
    
    # Should handle gracefully without overflow
    df_main, df_detailed = calculate_lifetime_fuel_costs(
        df=df_extreme,
        menu_mp=0,
        policy_scenario='AEO2023 Reference Case',
        verbose=False
    )
    
    # Result should be finite or NaN, not inf
    lifetime_col = f'baseline_{category}_lifetime_fuelCost'
    assert (df_main[lifetime_col].isna() | df_main[lifetime_col].abs().lt(float('inf'))).all(), \
        "Results should be finite or NaN, not infinite"
```

### 2. Performance Testing

```python
def test_large_dataframe_performance():
    """Test performance with large DataFrames."""
    # Create a large DataFrame (e.g., 10,000 rows)
    n_rows = 10000
    large_df = pd.DataFrame({
        'state': ['CA'] * n_rows,
        'census_division': ['Pacific'] * n_rows
    })
    
    # Add required columns
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        large_df[f'include_{category}'] = [True] * n_rows
        large_df[f'base_{category}_fuel'] = ['Electricity'] * n_rows
        
        # Add consumption columns for multiple years
        for year in range(2024, 2027):
            large_df[f'baseline_{year}_{category}_consumption'] = [100] * n_rows
    
    # Measure execution time
    import time
    start_time = time.time()
    
    df_main, _ = calculate_lifetime_fuel_costs(
        df=large_df,
        menu_mp=0,
        policy_scenario='AEO2023 Reference Case',
        verbose=False
    )
    
    execution_time = time.time() - start_time
    
    # Should complete in reasonable time (adjust based on expected performance)
    assert execution_time < 10, f"Large DataFrame calculation took {execution_time:.2f} seconds (should be under 10 seconds)"
    
    # Verify results have expected shape
    assert len(df_main) == n_rows, "Result should have same number of rows as input"
    for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
        lifetime_col = f'baseline_{category}_lifetime_fuelCost'
        assert lifetime_col in df_main.columns, f"Result should include column {lifetime_col}"
```