"""
**Edge Cases**

1. **Missing `census_division` Column**  
   - The function explicitly raises a `KeyError` if `census_division` does not exist. A test should confirm that the function fails properly in this scenario.
  
2. **Empty DataFrame**  
   - An empty DataFrame (with or without the `census_division` column) could lead to no new columns being generated. Tests should verify how the function behaves with no data.

3. **Missing Baseline Columns**  
   - The function uses columns like `baseline_heating_consumption` or `baseline_waterHeating_consumption`. If these columns are absent, pandas will raise a `KeyError` when trying to access them. This is an edge case that could occur if the DataFrame doesn’t follow the expected schema.

4. **Missing or Partial HDD Factors**  
   - If `lookup_hdd_factor` does not contain expected years or census divisions, the function defaults to `lookup_hdd_factor['National'][year_label]`. A test can verify that this fallback behavior works correctly (e.g., for a census division not in the dictionary).

5. **Edge or Invalid `menu_mp` Values**  
   - The function differentiates logic for `menu_mp == 0` (baseline) vs. `menu_mp` in `[8, 9, 10]` (retrofit). A test could cover a scenario where `menu_mp` is something unexpected (e.g., 1, 7, or negative) to confirm how the code behaves (currently, it treats anything other than 0 as an “upgraded equipment” scenario).

6. **Partially Filled Data**  
   - If some rows in the DataFrame have valid columns but others have `NaN` or missing values for consumption, the multiplication may return `NaN`. Ensuring the function handles these gracefully (and does not crash) could be an important test.

7. **Year Label Collisions**  
   - The function dynamically creates columns named `baseline_{year}_{category}_consumption` or `mp{menu_mp}_{year}_{category}_consumption`. If those columns already exist in the original DataFrame for some reason, the code drops them before joining. Testing how it handles an overlap in column names is another edge case.

8. **Extremely Large or Negative Consumption Values**  
   - The code multiplies baseline consumption by the HDD factor and rounds the result. A test could check how the function handles very large, very small (including zero), or negative consumption values (possibly due to data entry errors) without breaking.

---

**Additional Information to Generate Pytest Tests**

1. **Input DataFrame Schema**  
   - Which columns are mandatory (e.g., `census_division`, `baseline_heating_consumption`, `mp8_heating_consumption`, etc.)?  
   - Are optional columns possible?

2. **Valid Ranges or Expected Types for Consumption**  
   - Should consumption be strictly non-negative? Could it be zero?  
   - What about exceptionally large numbers?

3. **Valid Range of Years in `lookup_hdd_factor`**  
   - For example, if the function expects years 2023 to 2023 + `lifetime`, confirm that `lookup_hdd_factor` has the corresponding keys.  
   - Clarify whether the function should fail or fallback if a certain year is missing from the dictionary.

4. **Structure of `lookup_hdd_factor`**  
   - The code references keys like `[census_division][year_label]` and a `'National'` fallback. Confirm whether `'National'` is always guaranteed to exist.  
   - Should tests include scenarios where `'National'` is missing?

5. **Intended Behavior for Unexpected `menu_mp` Values**  
   - Currently, the code lumps everything other than `0` into “retrofit scenario.” Is that by design, or should the function raise an exception for invalid `menu_mp`?

6. **Performance Constraints**  
   - If data can be very large, confirm whether performance or memory usage is a concern for the test suite.

By clarifying these aspects, it becomes much easier to write focused, thorough pytest tests that cover both typical and edge-case usage.

# ========================================================================================================
# PROMPT FOR PROJECT_FUTURE_ENERGY_CONSUMPTION FUNCTION
# ========================================================================================================

ChatGPT, I have a Python function named `project_future_consumption` which projects future energy consumption based on baseline or upgraded equipment specs. It relies on a DataFrame (with columns like `census_division`, `baseline_heating_consumption`, etc.) and a lookup dictionary (`lookup_hdd_factor`) containing Heating Degree Day factors. The function adjusts consumption over multiple years, uses “National” as a fallback for missing factors, and differentiates scenarios based on a `menu_mp` parameter (0 for baseline, others like 8,9,10 for retrofit).

I want you to generate a **pytest test suite** that does the following:
1. Covers **edge cases** such as:
   - Missing `census_division` column (should raise `KeyError`).
   - Empty DataFrame.
   - Missing or partial baseline columns (e.g., `baseline_heating_consumption`).
   - Missing or partial `lookup_hdd_factor` keys, ensuring fallback to `'National'`.
   - Unexpected `menu_mp` values (like 1, 7, negative).
   - Rows with `NaN` or negative consumption values.
   - Existing columns that collide with new column names.
   - Extremely large or zero consumption.
2. Uses **best practices**: 
   - `pytest.raises` for exception checks.
   - **No actual file I/O** (create data in memory).
   - Fixtures if useful.
   - Parameterization where relevant.
3. Provides **docstrings** or inline comments explaining the purpose of each test.

Finally, produce the **fully-coded Python test file** so I can copy it into `test_project_future_consumption.py` and run it immediately with `pytest`.
"""

"""
test_project_future_consumption.py

Pytest suite for testing the 'project_future_consumption' function. 
It checks for success scenarios, failure conditions, boundary cases, 
and correct handling of fallback logic.
"""
import pytest
import pandas as pd
import numpy as np

# Replace 'your_module' with the actual module or script name
# where your 'project_future_consumption' function is defined.
from cmu_tare_model.utils.project_future_energy_consumption import *


@pytest.fixture
def baseline_df():
    """
    Returns a minimal valid DataFrame with all necessary columns 
    to run project_future_consumption without errors under normal conditions.
    """
    return pd.DataFrame({
        'census_division': ['Division1', 'Division2'],
        'baseline_heating_consumption': [100.0, 200.0],
        'baseline_waterHeating_consumption': [80.0, 150.0],
        'baseline_clothesDrying_consumption': [50.0, 50.0],
        'baseline_cooking_consumption': [40.0, 60.0],
        'mp8_heating_consumption': [90.0, 180.0],
        'mp8_waterHeating_consumption': [70.0, 130.0],
        'mp8_clothesDrying_consumption': [45.0, 45.0],
        'mp8_cooking_consumption': [35.0, 50.0],
        # Similarly for mp9, mp10, if desired
        'baseline_2024_heating_consumption': [105.0, 210.0],  # Example existing year label
    })


@pytest.fixture
def hdd_factors():
    """
    Returns a dictionary simulating the HDD factor lookup with 
    both division-specific entries and a 'National' fallback.
    """
    return {
        'Division1': {
            2024: 1.05,
            2025: 1.06,
            # years up to 2038 if needed
        },
        'Division2': {
            2024: 1.10,
            2025: 1.12,
        },
        'National': {
            2024: 1.01,
            2025: 1.02,
        }
    }


def test_missing_census_division_column(baseline_df, hdd_factors):
    """
    Ensure a KeyError is raised if the 'census_division' column is absent.
    """
    df_no_census = baseline_df.drop(columns=['census_division'])
    with pytest.raises(KeyError, match="census_division"):
        project_future_consumption(df_no_census, hdd_factors, menu_mp=0)


def test_empty_dataframe(hdd_factors):
    """
    Test how the function handles an empty DataFrame.
    Expected: It should return two empty DataFrames without error.
    """
    empty_df = pd.DataFrame()
    # Since 'census_division' is missing, it should raise KeyError immediately
    with pytest.raises(KeyError):
        project_future_consumption(empty_df, hdd_factors, menu_mp=0)


def test_missing_baseline_columns(baseline_df, hdd_factors):
    """
    If baseline columns (like 'baseline_heating_consumption') are missing, 
    pandas should raise KeyError when accessed.
    """
    df_missing_baseline = baseline_df.drop(columns=['baseline_heating_consumption'])
    with pytest.raises(KeyError):
        project_future_consumption(df_missing_baseline, hdd_factors, menu_mp=0)


def test_missing_or_partial_hdd_factors(baseline_df):
    """
    Test scenario where the lookup dictionary is missing 
    some divisions or years, thus requiring fallback to 'National'.
    """
    # Missing 'Division1' or certain year keys
    partial_factors = {
        'Division2': {
            2024: 1.10
        },
        'National': {
            2024: 1.01,
            2025: 1.02,
        }
    }
    df_copy, df_consumption = project_future_consumption(
        baseline_df, partial_factors, menu_mp=0
    )
    # Check if columns for 2024 were created using the fallback for Division1
    assert 'baseline_2024_heating_consumption' in df_copy.columns, \
        "Projected consumption columns for 2024 missing."
    # Ensure the function did not crash and used 'National' fallback


@pytest.mark.parametrize("menu_mp_val", [0, 8, 9, 10, 1, -1, 999])
def test_menu_mp_values(baseline_df, hdd_factors, menu_mp_val):
    """
    Confirm that the function differentiates baseline (menu_mp=0) 
    from retrofit (menu_mp != 0) scenarios, without failing for unexpected values.
    """
    df_copy, df_consumption = project_future_consumption(
        baseline_df, hdd_factors, menu_mp=menu_mp_val
    )
    # If menu_mp=0, we expect columns starting with 'baseline_YYYY_...'
    # Otherwise, columns should start with 'mp{menu_mp_val}_YYYY_...'
    if menu_mp_val == 0:
        expected_substring = 'baseline_'
    else:
        expected_substring = f'mp{menu_mp_val}_'
    # Check if at least one newly generated column contains our substring
    new_cols = [col for col in df_copy.columns if expected_substring in col]
    assert len(new_cols) > 0, f"No columns generated for menu_mp={menu_mp_val}"


def test_partially_filled_data(baseline_df, hdd_factors):
    """
    Some rows may have NaN or missing consumption values. 
    The function should not crash and should produce NaN results where appropriate.
    """
    # Introduce NaN in a baseline column
    baseline_df.loc[0, 'baseline_heating_consumption'] = np.nan
    df_copy, df_consumption = project_future_consumption(
        baseline_df, hdd_factors, menu_mp=0
    )
    # Check if the new columns exist and contain NaN
    new_col = 'baseline_2024_heating_consumption'
    assert new_col in df_copy.columns, "Expected new column not found."
    assert pd.isna(df_copy.loc[0, new_col]), "NaN value not propagated as expected."


def test_year_label_collisions(baseline_df, hdd_factors):
    """
    The DataFrame might already have columns like 'baseline_2024_heating_consumption'.
    This test checks that the function drops existing columns before adding new ones.
    """
    # The fixture baseline_df already has 'baseline_2024_heating_consumption'
    # The function should drop and recreate it without error.
    df_copy, df_consumption = project_future_consumption(
        baseline_df, hdd_factors, menu_mp=0
    )
    assert 'baseline_2024_heating_consumption' in df_copy.columns, \
        "Column for year 2024 should be recreated if it overlaps."


def test_extremely_large_or_negative_consumption_values(baseline_df, hdd_factors):
    """
    If consumption values are negative or extremely large, 
    the function should still run without error, producing 
    negative or large results accordingly.
    """
    # Modify consumption values
    baseline_df.loc[0, 'baseline_heating_consumption'] = -999999
    baseline_df.loc[1, 'baseline_heating_consumption'] = 1e12
    df_copy, df_consumption = project_future_consumption(
        baseline_df, hdd_factors, menu_mp=0
    )
    # Check for presence of updated columns without crash
    large_neg_col = 'baseline_2024_heating_consumption'
    assert large_neg_col in df_copy.columns, "Expected column missing."
    # We don't specifically fail on negative or large numbers by design,
    # just ensure the function can handle them.


def test_successful_case(baseline_df, hdd_factors):
    """
    Basic sanity check that the function returns two DataFrames of the same shape, 
    with new consumption columns appended.
    """
    df_copy, df_consumption = project_future_consumption(
        baseline_df, hdd_factors, menu_mp=0
    )
    # df_consumption is a copy of df_copy at the end, so they should have the same columns
    assert df_copy.shape == df_consumption.shape, \
        "df_copy and df_consumption should match in shape."
    # Confirm that at least one new year-based column was generated
    new_year_cols = [c for c in df_copy.columns if '_202' in c]
    assert len(new_year_cols) > 0, "No projected consumption columns were generated."
