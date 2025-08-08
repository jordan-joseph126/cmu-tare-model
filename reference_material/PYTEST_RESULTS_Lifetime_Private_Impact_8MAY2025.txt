#TEST RESULTS FOR test_calculate_lifetime_private_impact.py:

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
FAILED test_lifetime_private_impact.py::test_mask_initialization[heating] - AssertionError: Valid mask should match include_heating column
FAILED test_lifetime_private_impact.py::test_mask_initialization[waterHeating] - AssertionError: Valid mask should match include_waterHeating column      
FAILED test_lifetime_private_impact.py::test_mask_initialization[clothesDrying] - AssertionError: Valid mask should match include_clothesDrying column    
FAILED test_lifetime_private_impact.py::test_mask_initialization[cooking] - AssertionError: Valid mask should match include_cooking column
FAILED test_lifetime_private_impact.py::test_calculate_and_update_npv_basic - AssertionError: NPV (less WTP) should be roughly proportional to expected value
FAILED test_lifetime_private_impact.py::test_calculate_private_npv_basic - AssertionError: NPV with more WTP should be higher than with less WTP for home at index 1
FAILED test_lifetime_private_impact.py::test_rebate_application - AssertionError: Capital costs should be lower with IRA rebates for home at index 1      
FAILED test_lifetime_private_impact.py::test_weatherization_costs - AssertionError: Difference in capital costs should equal net weatherization cost for home at index 1
FAILED test_lifetime_private_impact.py::test_across_categories[heating] - AssertionError: Valid home at index 1 should have a value for column 'iraRef_mp8_heating_private_npv_lessWTP'
FAILED test_lifetime_private_impact.py::test_across_categories[waterHeating] - AssertionError: Valid home at index 1 should have a value for column 'iraRef_mp8_waterHeating_private_npv_lessWTP'
FAILED test_lifetime_private_impact.py::test_across_categories[clothesDrying] - AssertionError: Valid home at index 0 should have a value for column 'iraRef_mp8_clothesDrying_private_npv_lessWTP'
FAILED test_lifetime_private_impact.py::test_across_categories[cooking] - AssertionError: Valid home at index 1 should have a value for column 'iraRef_mp8_cooking_private_npv_lessWTP'
FAILED test_lifetime_private_impact.py::test_empty_dataframe - AssertionError: Error message should mention empty DataFrame or missing data
FAILED test_lifetime_private_impact.py::test_missing_fuel_cost_data - AssertionError: NPV should be negative when fuel cost savings are missing for home at index 1
FAILED test_lifetime_private_impact.py::test_negative_cost_scenarios - AssertionError: NPV should be positive when installation costs are negative for home at index 1
======================================================= 15 failed, 12 passed, 1 warning in 4.04s ======================================================== 