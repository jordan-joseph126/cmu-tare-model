# Pytest Test Suite Improvement for Energy Retrofit Model

## PROJECT OVERVIEW
Based on the test results and code files you provided, I need to create a comprehensive prompt that will help you fix failing tests in your energy retrofit model's validation framework and lifetime fuel costs calculation modules.

## TEST RESULTS ANALYSIS

The test results show several failures:

### calculate_lifetime_fuel_costs.py (2 failures):
1. `test_final_masking_implementation` - Error: "Lifetime column 'iraRef_mp8_heating_lifetime_fuelCost' should be tracked for final masking"
2. `test_lifetime_fuel_costs_list_collection` - Error: "Lifetime cost should match sum of yearly costs for home at index 0."

### validation_framework.py (3 failures):
1. `test_replace_small_values_utility` - Error: "Value -1e-9 in col2 should be NaN"
2. `test_replace_small_values_with_custom_threshold` - Error: "1e-9 should be NaN with default threshold"
3. `test_full_validation_framework` - Error: "Validation step 'mask_initialization' was not executed"

## PROMPT FOR CLAUDE

```
I need help fixing failing tests in my energy retrofit model's validation framework and fuel costs calculations. The test failures suggest specific implementation issues that need to be addressed while maintaining the integrity of our 5-step validation framework.

## Current Test Status
The test results show 5 failing tests across two modules:

### calculate_lifetime_fuel_costs.py (2 failures):
1. `test_final_masking_implementation` - Error: "Lifetime column 'iraRef_mp8_heating_lifetime_fuelCost' should be tracked for final masking"
2. `test_lifetime_fuel_costs_list_collection` - Error: "Lifetime cost should match sum of yearly costs for home at index 0."

### validation_framework.py (3 failures):
1. `test_replace_small_values_utility` - Error: "Value -1e-9 in col2 should be NaN"
2. `test_replace_small_values_with_custom_threshold` - Error: "1e-9 should be NaN with default threshold"
3. `test_full_validation_framework` - Error: "Validation step 'mask_initialization' was not executed"

## Source Code
I've attached the relevant source files:
1. `calculate_lifetime_fuel_costs.py` - Implementation of lifetime fuel cost calculations
2. `validation_framework.py` - Core utilities for the 5-step validation framework
3. `test_calculate_lifetime_fuel_costs.py` - Tests for fuel cost calculations
4. `test_validation_framework.py` - Tests for the validation framework

## 5-Step Validation Framework Requirements
All calculations must follow our validation framework:
1. Mask Initialization with initialize_validation_tracking()
2. Series Initialization with create_retrofit_only_series()
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates with list-based collection instead of incremental updates
5. Final Masking with apply_final_masking()

## What I Need Help With
1. Detailed analysis of each failing test to identify the root causes
2. Specific code fixes for each issue, with both original and replacement code shown
3. Clear explanation of WHY each change fixes the underlying problem
4. Suggestions for improving test reliability

Please organize your response by issue, with clear Before/After code comparisons and explanations for each fix. Focus particularly on:

1. **Column Tracking for Final Masking**: The first failure suggests lifetime columns aren't being tracked properly for masking.

2. **NaN Propagation in Yearly Summation**: The second failure implies an issue with how NaN values are handled when summing yearly costs.

3. **Small Values Replacement in DataFrames**: The third and fourth failures indicate problems with replacing small values with NaN in DataFrames, possibly related to column-by-column processing.

4. **Validation Step Tracking**: The fifth failure suggests an issue with how validation steps are detected or executed in the test.

I value thorough, educational responses that help me understand both WHAT is wrong and WHY the proposed fixes work. Please focus on maintaining our data validation principles while solving these specific issues.
```