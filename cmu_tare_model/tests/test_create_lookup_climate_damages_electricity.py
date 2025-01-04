"""
Below are some potential edge cases to consider and additional information that would help in generating comprehensive pytest tests for these two functions.

## Edge Cases for `calculate_electricity_co2e_cambium(df_cambium_import)`

1. **Insufficient Data Points for Interpolation**  
   - If a given scenario–GEA region group has fewer than 2 unique years, linear interpolation cannot be performed. This might raise a `ValueError` from `interp1d`.
2. **Missing Required Columns**  
   - If columns such as `scenario`, `gea_region`, `year`, `lrmer_co2e_kg_per_MWh`, or `srmer_co2e_kg_per_MWh` are missing, a `KeyError` would be raised.
3. **Non-Numeric or Invalid Data**  
   - If the emission factor columns contain non-numeric values (e.g., strings) or have invalid entries (e.g., `NaN`, negative values for emission rates), interpolation might fail or produce unexpected results.
4. **Duplicate Rows**  
   - Multiple rows with the same `(scenario, gea_region, year)` combination could lead to ambiguous interpolation.  
5. **Discontinuous or Missing Years**  
   - Scenarios in which the available years are not consecutive (e.g., 2020, 2022, 2025 with missing 2021, 2023, 2024) rely on interpolation to fill gaps. Large gaps may lead to unexpected interpolation outcomes or errors.
6. **Very Large Range of Years**  
   - A large range of years (e.g., 1900–2100) could create a performance issue or a very large output DataFrame, though not necessarily an error.

## Edge Cases for `create_cambium_co2e_lookup(df_cambium_processed)`

1. **Empty or Minimal DataFrame**  
   - If the DataFrame is empty or has only one row, the nested dictionary should be handled gracefully.
2. **Missing Required Columns**  
   - If columns such as `scenario`, `gea_region`, `year`, `lrmer_co2e_ton_per_kWh`, or `srmer_co2e_ton_per_kWh` are missing, a `KeyError` is possible.
3. **Duplicate Rows**  
   - Multiple entries with the same `(scenario, gea_region, year)` combination might lead to the last one overwriting the previous values or unwanted collisions in the lookup dictionary.
4. **Non-Numeric or Invalid Data**  
   - If the emission factors in `lrmer_co2e_ton_per_kWh` or `srmer_co2e_ton_per_kWh` contain non-numeric values or negative numbers, it might represent invalid states that should be tested.

## Additional Information to Generate Pytest Tests

1. **Sample Data**  
   - A representative dataset (or multiple small DataFrames) that include various scenarios, GEA regions, and years. Ideally, the sample data would demonstrate normal cases, corner cases, and erroneous entries (e.g., missing columns, negative values).
2. **Expected Behavior/Outcomes**  
   - Clear expectations about what should happen when:
     - Interpolation fails or when there is only one year of data.  
     - A column is missing.  
     - Data contains `NaN` or invalid emission rates.  
   - This helps specify whether an exception should be raised or if the function should handle it gracefully.
3. **Performance Expectations**  
   - Guidance on whether the functions need to handle very large DataFrames. This helps in writing stress tests or verifying speed and memory usage (if needed).
4. **Allowed Year Ranges**  
   - If there is a valid year range (e.g., 2020–2050) that the data is expected to cover, test cases can be designed to see how the functions handle data outside those boundaries.
5. **Type Checking Requirements**  
   - Information on whether or not to strictly enforce types (e.g., years must be integers, emission factors must be floats). This allows for specific tests to validate data type handling or conversions.

With these edge cases and clarifications, you can craft thorough `pytest` test suites for both normal and abnormal data scenarios, ensuring the functions behave consistently and predictably under various conditions.
"""