"""
Below are some potential edge cases for the calculate_fuel_costs function, along with the additional information that would help in generating more comprehensive Pytest tests.

Edge Cases
1. Empty DataFrame
    Description: The function is called with a completely empty df or one missing required columns.
    Expected Behavior: Depending on how robust you want the function to be, it could either return an empty DataFrame or raise an exception indicating missing columns.

2. Missing or Misnamed Columns
    Description: Some key columns (e.g., state, census_division, base_heating_fuel, or consumption columns like baseline_2024_heating_consumption) are missing or have unexpected names.
    Expected Behavior: The function may fail with a KeyError or produce incomplete results if columns are missing. Tests should check that the function handles or reports this situation in an expected way.

3. Invalid policy_scenario
    Description: A string other than 'No Inflation Reduction Act' or 'AEO2023 Reference Case' is passed.
    Expected Behavior: The function should raise a ValueError stating that an invalid scenario was provided.

4. menu_mp Edge Cases
    Description:
    menu_mp = 0 (baseline scenario) is already covered in the code, but what if negative or very large integers are passed?
    The code logic does not explicitly handle invalid or unexpected measure package IDs aside from 0 and “any other integer.”
    Expected Behavior: The function will default to the “MP scenario” branch for any non-zero integer. You may want to verify the function still produces correct columns or handle out-of-range measure package IDs.

5. Lookup for States Not in Fuel Price Dictionaries
    Description: The df includes a state or census_division that is not in the lookup_fuel_prices_preIRA or lookup_fuel_prices_iraRef dictionaries.
    Expected Behavior: The code uses .get(..., 0) if not found, so the resulting fuel cost might be 0. A test could verify that a missing lookup gracefully defaults to 0 rather than crashing.

6. Missing or Zero Consumption Values
    Description: Some (or all) consumption columns contain NaN, negative values, or zeros.
    Expected Behavior:
        NaN or negative consumption could lead to NaN or negative fuel cost.
        Zero consumption should lead to zero cost.
    Tests should confirm how these edge values propagate into the resulting fuel cost columns.

7. drop_fuel_cost_columns = True
    Description: After computing costs, dropping columns might create scenarios where baseline columns are removed before you can compute savings for a second run.
    Expected Behavior: Ensure that the columns are actually dropped as expected and that savings columns remain intact.

8. Large DataFrames
    Description: Testing performance or memory usage with large DataFrames.
    Expected Behavior: The function should handle large datasets efficiently without timing out or using excessive memory.

9. No Variation in Fuel Types
    Description: If the DataFrame has only one type of fuel for all rows (e.g., only Electricity or only Natural Gas).
    Expected Behavior: The function should correctly compute costs for the single fuel type and not fail due to missing others.

10. Unexpected Fuel Types
    Description: The base_heating_fuel column contains a fuel type not in {'Electricity', 'Natural Gas', 'Fuel Oil', 'Propane'}.
    Expected Behavior: Because of the .map(fuel_mapping), an unrecognized fuel type may become NaN. The subsequent lookups default to 0. Tests can confirm that unrecognized fuels default to 0 rather than causing a crash.

Additional Information for Test Generation

1. Complete Column Specifications
    A clear list of all columns required by the function, including those for baseline (e.g., base_heating_fuel, baseline_2024_heating_consumption, etc.) and measure package columns (e.g., mp1_2024_heating_consumption, etc.).
    This helps validate that test data sets include or exclude columns intentionally to test various scenarios.

2. Sample Fuel Price Dictionaries
    Examples or mocks of lookup_fuel_prices_preIRA and lookup_fuel_prices_iraRef structure.
    For instance, what states or census divisions are available, what years are covered, and what happens if a requested year is absent?
    This helps you verify correct lookups and the defaulting-to-zero logic.

3. Range of Possible Years
    The function currently calculates from 2024 to 2024 + lifetime for each end use. Knowing whether the dictionary covers all these years (e.g., 2024–2039 for heating) or if some years might be missing ensures tests check the zero-default logic.

4. Valid menu_mp Range
    Clarification on whether measure package IDs can be any integer (1, 2, 3, etc.) or if there’s a limited set. This could influence test scenarios for invalid or out-of-range IDs.

5. Expected DataFrame Size and Data Types
    Understanding typical row counts, performance constraints, and typical data types (e.g., floats vs. integers for consumption).
    You can then design stress tests (large DataFrames) and corner cases (very small or empty DataFrames).

6. Behavior With Null or NaN Values
    Clarification on whether NaN consumption values should be replaced with 0, raise errors, or simply propagate as NaN.
    This helps define how to handle incomplete or partially missing data.

With the above details, you can write more comprehensive and robust pytest tests, each targeting specific edge cases, ensuring the function handles real-world complexities without unexpected behavior.

# =====================================================================================================================================================================
# PROMPT FOR CALCUATE_FUEL_COSTS FUNCTION
# =====================================================================================================================================================================

Below is an **updated prompt** that provides clearer context, objectives, and requirements for generating pytest tests specifically for the `calculate_annual_fuel_cost` function, following best practices in prompt engineering.

---

## **Prompt**

I have a Python function named `calculate_annual_fuel_cost` with the following signature:

```python
calculate_annual_fuel_cost(df, menu_mp, policy_scenario, drop_fuel_cost_columns)
```

where:
- `df` is a `pandas.DataFrame` that must contain baseline and (optionally) measure-package-related fuel consumption data and location info (`state`, `census_division`).
- `menu_mp` is an integer indicating the measure package (0 = baseline scenario).
- `policy_scenario` is a string that must be either `"No Inflation Reduction Act"` or `"AEO2023 Reference Case"`.
- `drop_fuel_cost_columns` is a boolean that indicates whether to drop the annual fuel cost columns after calculating savings.

The function:
1. Calculates annual fuel costs for baseline (when `menu_mp = 0`) or for a specific measure package (when `menu_mp != 0`).
2. Looks up fuel prices from dictionaries (`lookup_fuel_prices_preIRA` or `lookup_fuel_prices_iraRef`), keyed by state or census division.
3. Multiplies the per-year fuel consumption by the respective price to derive a cost column (e.g., `baseline_2024_heating_fuel_cost`).
4. Optionally calculates savings when comparing baseline to measure packages.
5. May drop intermediate columns if `drop_fuel_cost_columns` is `True`.

### **Context**
- The function is designed to handle different policy scenarios, fuel types, and measure packages to produce a final DataFrame with additional columns for fuel costs and savings.
- It raises a `ValueError` if an invalid policy scenario is passed.
- It defaults to zero cost if the lookup dictionary does not contain an entry for a given state, fuel type, or year.

### **Objectives**
I need **pytest** tests that ensure:
- **Edge cases** and **core logic** are covered, including:
  1. **Empty DataFrame** or missing required columns (verify correct handling or exceptions).
  2. **Invalid `policy_scenario`** (e.g., raises `ValueError`).
  3. **`menu_mp` edge cases** (e.g., negative integers, very large integers).
  4. **Missing or zero consumption values** (ensuring fuel costs remain zero or handle `NaN`).
  5. **Fuel-price lookups missing** for a given state or year (ensure default `0` cost).
  6. **Dropping columns** when `drop_fuel_cost_columns` is `True`.
  7. **Multiple categories (heating, waterHeating, etc.)** with partial or inconsistent year coverage.
- **Exception handling** with `pytest.raises` for invalid inputs.
- **Parametrized tests** wherever it helps reduce repetition (e.g., multiple `menu_mp` or `policy_scenario` values).
- **No real I/O** or external dependencies—mock out or provide minimal data inline if needed.
- **Clear coverage** of main logic pathways (baseline vs. measure package).

### **Constraints/Requirements**
1. Use **pytest**.
2. Test the function’s **success** paths, **failure** paths, and **boundary conditions**.
3. **No actual file I/O**—all data for tests should be embedded or generated in the test code.
4. Where relevant, use **fixtures** to set up test data in a clean, reusable way.
5. Provide **short docstrings** or **inline comments** in the test code explaining what each test checks.

**Your final answer** should be the fully-written test file in Python syntax (e.g., `test_annual_fuel_cost.py`) that I can copy and run immediately with `pytest`.

"""
