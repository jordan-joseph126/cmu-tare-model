#TEST RESULTS FOR test_validation_framework.py:

============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0 -- c:\Users\14128\anaconda3\python.exe
cachedir: .pytest_cache
rootdir: c:\Users\14128\Research\cmu-tare-model
plugins: anyio-4.2.0
collecting ... collected 34 items

test_validation_framework.py::test_mask_initialization_basic Project root directory: c:\users\14128\research\cmu-tare-model
Fuel sources in df_grid_mix: {'Coal', 'Nuclear', 'Natural Gas', 'Renewable', 'Oil'}
Fuel sources in df_grid_emis_factors: {'Coal', 'Nuclear', 'Renewables', 'Natural Gas', 'Oil'}
Retrieved data for filename: bls_cpiu_2005-2023.xlsx
Located at filepath: c:\users\14128\research\cmu-tare-model\cmu_tare_model\data\inflation_data\bls_cpiu_2005-2023.xlsx
    year  cpiu_annual
0   2005      195.300
1   2006      201.600
2   2007      207.342
3   2008      215.303
4   2009      214.537
5   2010      218.056
6   2011      224.939
7   2012      229.594
8   2013      232.957
9   2014      236.736
10  2015      237.017
11  2016      240.007
12  2017      245.120
13  2018      251.107
14  2019      255.657
15  2020      258.811
16  2021      270.970
17  2022      292.655
18  2023      304.702
PASSED
test_validation_framework.py::test_mask_initialization_different_categories[heating] PASSED
test_validation_framework.py::test_mask_initialization_different_categories[waterHeating] PASSED
test_validation_framework.py::test_mask_initialization_different_categories[clothesDrying] PASSED
test_validation_framework.py::test_mask_initialization_different_categories[cooking] PASSED
test_validation_framework.py::test_mask_initialization_missing_include_flag PASSED
test_validation_framework.py::test_mask_initialization_empty_dataframe PASSED
test_validation_framework.py::test_retrofit_status_integration PASSED
test_validation_framework.py::test_series_initialization_basic[heating] PASSED
test_validation_framework.py::test_series_initialization_basic[waterHeating] PASSED
test_validation_framework.py::test_series_initialization_basic[clothesDrying] PASSED
test_validation_framework.py::test_series_initialization_basic[cooking] PASSED
test_validation_framework.py::test_series_initialization_all_valid PASSED
test_validation_framework.py::test_series_initialization_all_invalid PASSED
test_validation_framework.py::test_series_initialization_empty_dataframe PASSED
test_validation_framework.py::test_series_initialization_non_standard_index PASSED
test_validation_framework.py::test_series_initialization_direct_mask_derivation PASSED
test_validation_framework.py::test_valid_only_calculation_basic PASSED
test_validation_framework.py::test_valid_only_calculation_compound_mask PASSED
test_validation_framework.py::test_valid_only_calculation_edge_cases PASSED
test_validation_framework.py::test_valid_only_calculation_multi_column PASSED
test_validation_framework.py::test_list_based_collection PASSED
test_validation_framework.py::test_calculate_avoided_values_utility PASSED
test_validation_framework.py::test_replace_small_values_utility FAILED

================================== FAILURES ===================================
______________________ test_replace_small_values_utility ______________________

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
    
        # Test with DataFrame
        df = pd.DataFrame({
            'col1': [1.0, 1e-12, -1e-11, 0.1, -0.2],
            'col2': [0.5, -1e-9, 2e-8, -0.3, 0.7]
        })
        result_df = replace_small_values_with_nan(df)
    
        # Check values in first column
        assert pd.isna(result_df.loc[1, 'col1']), "Value 1e-12 in col1 should be NaN"
        assert pd.isna(result_df.loc[2, 'col1']), "Value -1e-11 in col1 should be NaN"
    
        # Check values in second column
>       assert pd.isna(result_df.loc[1, 'col2']), "Value -1e-9 in col2 should be NaN"
E       AssertionError: Value -1e-9 in col2 should be NaN
E       assert False
E        +  where False = <function isna at 0x000002D2031F3880>(-1e-09)
E        +    where <function isna at 0x000002D2031F3880> = pd.isna

test_validation_framework.py:935: AssertionError
=========================== short test summary info ===========================
FAILED test_validation_framework.py::test_replace_small_values_utility - AssertionError: Value -1e-9 in col2 should be NaN
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!
======================== 1 failed, 23 passed in 4.36s =========================

#TEST RESULTS FOR test_calculate_lifetime_fuel_costs.py:
=================================================================== warnings summary ==================================================================== 
..\..\..\..\anaconda3\Lib\site-packages\jupyter_client\connect.py:22
  C:\Users\14128\anaconda3\Lib\site-packages\jupyter_client\connect.py:22: DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
  given by the platformdirs library.  To remove this warning and
  see the appropriate new directories, set the environment variable
  `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
  The use of platformdirs will be the default in `jupyter_core` v6
    from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
================================================================ short test summary info ================================================================ 
FAILED test_calculate_lifetime_fuel_costs.py::test_final_masking_implementation - AssertionError: Lifetime column 'iraRef_mp8_heating_lifetime_fuelCost' should be tracked for final masking
FAILED test_calculate_lifetime_fuel_costs.py::test_lifetime_fuel_costs_basic - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_lifetime_fuel_costs_with_baseline - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_lifetime_fuel_costs_list_collection - AssertionError: Lifetime cost should match sum of yearly costs for home at index 0
FAILED test_calculate_lifetime_fuel_costs.py::test_different_categories[heating] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_different_categories[waterHeating] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_different_categories[clothesDrying] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_different_categories[cooking] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_different_menu_mps[0] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_different_menu_mps[8] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_different_policy_scenarios[No Inflation Reduction Act] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
FAILED test_calculate_lifetime_fuel_costs.py::test_different_policy_scenarios[AEO2023 Reference Case] - RuntimeError: Error processing category 'heating': Error processing year 2024 for category 'heating': mock_annual_calculation.<locals>.mock_calculate...
======================================================= 12 failed, 13 passed, 1 warning in 4.97s ========================================================