# ========================================================================================================================================================================================================================================================================
# TEST RESULTS FOR test_calculate_lifetime_climate_impacts_sensitivity_8May2025.py
# ========================================================================================================================================================================================================================================================================
```
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_success[No Inflation Reduction Act-0] - RuntimeError: Error processing category 'heating': Inclusion flag 'include_heating' not found in DataFrame. Ensure identify_valid_homes() has been ca...
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_success[No Inflation Reduction Act-8] - RuntimeError: Error processing category 'heating': Inclusion flag 'include_heating' not found in DataFrame. Ensure identify_valid_homes() has been ca...
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_success[AEO2023 Reference Case-0] - RuntimeError: Error processing category 'heating': Inclusion flag 'include_heating' not found in DataFrame. Ensure identify_valid_homes() has been ca...
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_success[AEO2023 Reference Case-8] - RuntimeError: Error processing category 'heating': Inclusion flag 'include_heating' not found in DataFrame. Ensure identify_valid_homes() has been ca...
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_boundary_lifetime - RuntimeError: Error processing category 'test_cat': Inclusion flag 'include_test_cat' not found in DataFrame. Ensure identify_valid_homes() has been ...
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_invalid_policy_scenario[SomeUnknownPolicy] - AssertionError: Regex pattern did not match.
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_invalid_policy_scenario[InvalidScenario] - AssertionError: Regex pattern did not match.
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_missing_hdd_factor_year - AssertionError: Regex pattern did not match.
FAILED test_calculate_lifetime_climate_impacts_sentivity_8May2025.py::test_calculate_lifetime_climate_impacts_with_baseline_damages - RuntimeError: Error processing category 'heating': Inclusion flag 'include_heating' not found in DataFrame. Ensure identify_valid_homes() has been ca...
```

# ========================================================================================================================================================================================================================================================================
# TEST RESULTS FOR test_calculate_lifetime_climate_impacts_sensitivity.py
# ========================================================================================================================================================================================================================================================================
================================================================== test session starts ==================================================================
platform win32 -- Python 3.11.7, pytest-7.4.0, pluggy-1.0.0
rootdir: C:\Users\14128\Research\cmu-tare-model
plugins: anyio-4.2.0
collected 29 items                                                                                                                                       

test_calculate_lifetime_climate_impacts_sensitivity.py EEEEEEEEEEEEEEEEEEEEEEEEEEEEE                                                               [100%]

======================================================================== ERRORS =========================================================================
_______________________________________________ ERROR at setup of test_mask_initialization_implementation _______________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
______________________________________________ ERROR at setup of test_series_initialization_implementation ______________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_____________________________________________ ERROR at setup of test_valid_only_calculation_implementation ______________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
______________________________________________ ERROR at setup of test_list_based_collection_implementation ______________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
__________________________________________________ ERROR at setup of test_final_masking_implementation __________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
________________________________________________ ERROR at setup of test_all_validation_steps_integrated _________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
________________________________________ ERROR at setup of test_calculate_climate_emissions_and_damages_success _________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
____________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_success[No Inflation Reduction Act-0] ____________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
____________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_success[No Inflation Reduction Act-8] ____________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
______________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_success[AEO2023 Reference Case-0] ______________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
______________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_success[AEO2023 Reference Case-8] ______________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
____________________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_with_baseline_damages ____________________________________
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
__________________________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_empty_df ___________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_______________________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_missing_column ________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_invalid_policy_scenario[SomeUnknownPolicy] __________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
__________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_invalid_policy_scenario[InvalidScenario] ___________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
____________________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_missing_region_factor ____________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
______________________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_boundary_lifetime ______________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
___________________________________ ERROR at setup of test_calculate_lifetime_climate_impacts_missing_hdd_factor_year ___________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_________________________________________________ ERROR at setup of test_different_categories[heating] __________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_______________________________________________ ERROR at setup of test_different_categories[waterHeating] _______________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
______________________________________________ ERROR at setup of test_different_categories[clothesDrying] _______________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_________________________________________________ ERROR at setup of test_different_categories[cooking] __________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_____________________________________________________ ERROR at setup of test_different_menu_mps[0] ______________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_____________________________________________________ ERROR at setup of test_different_menu_mps[8] ______________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_____________________________________ ERROR at setup of test_different_policy_scenarios[No Inflation Reduction Act] _____________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_______________________________________ ERROR at setup of test_different_policy_scenarios[AEO2023 Reference Case] _______________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
_______________________________________________________ ERROR at setup of test_all_invalid_homes ________________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
____________________________________________________ ERROR at setup of test_missing_required_columns ____________________________________________________ 
Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
but are created automatically when test functions request them as parameters.
See https://docs.pytest.org/en/stable/explanation/fixtures.html for more information about fixtures, and
https://docs.pytest.org/en/stable/deprecations.html#calling-fixtures-directly about how to update your code.
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
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_mask_initialization_implementation - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_series_initialization_implementation - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_valid_only_calculation_implementation - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_list_based_collection_implementation - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_final_masking_implementation - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_all_validation_steps_integrated - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_climate_emissions_and_damages_success - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_success[No Inflation Reduction Act-0] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_success[No Inflation Reduction Act-8] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_success[AEO2023 Reference Case-0] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_success[AEO2023 Reference Case-8] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_with_baseline_damages - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_empty_df - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_missing_column - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_invalid_policy_scenario[SomeUnknownPolicy] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_invalid_policy_scenario[InvalidScenario] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_missing_region_factor - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_boundary_lifetime - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_calculate_lifetime_climate_impacts_missing_hdd_factor_year - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_categories[heating] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_categories[waterHeating] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_categories[clothesDrying] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_categories[cooking] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_menu_mps[0] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_menu_mps[8] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_policy_scenarios[No Inflation Reduction Act] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_different_policy_scenarios[AEO2023 Reference Case] - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_all_invalid_homes - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
ERROR test_calculate_lifetime_climate_impacts_sensitivity.py::test_missing_required_columns - Failed: Fixture "dummy_scc_lookup" called directly. Fixtures are not meant to be called directly,
============================================================= 1 warning, 29 errors in 3.83s ============================================================= 
