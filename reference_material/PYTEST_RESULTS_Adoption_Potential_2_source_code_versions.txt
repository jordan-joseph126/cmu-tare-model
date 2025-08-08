USING V1 OF THE SOURCE CODE:

# Import the module to test
from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity import (
    adoption_decision,
    validate_input_parameters
)

...

========================================================================= warnings summary ========================================================================== 
..\..\..\..\anaconda3\Lib\site-packages\jupyter_client\connect.py:22
  C:\Users\14128\anaconda3\Lib\site-packages\jupyter_client\connect.py:22: DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
  given by the platformdirs library.  To remove this warning and
  see the appropriate new directories, set the environment variable
  `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
  The use of platformdirs will be the default in `jupyter_core` v6
    from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================================================================== short test summary info ====================================================================== 
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_error_handling_for_invalid_scc - RuntimeError: An unexpected error occurred in adoption_decision: SCC_ASSUMPTIONS is empty. Cannot perform climate sensitivity analysis.
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_climate_sensitivity_parameter - AssertionError: Should not have 'lower' SCC columns when climate_sensitivity=False
============================================================== 2 failed, 34 passed, 1 warning in 3.98s ============================================================== 


USING V2 OF THE SOURCE CODE:

# Import the module to test
from cmu_tare_model.adoption_potential.determine_adoption_potential_sensitivity_V2 import (
    adoption_decision,
    validate_input_parameters
)

...

========================================================================= warnings summary ========================================================================== 
..\..\..\..\anaconda3\Lib\site-packages\jupyter_client\connect.py:22
  C:\Users\14128\anaconda3\Lib\site-packages\jupyter_client\connect.py:22: DeprecationWarning: Jupyter is migrating its paths to use standard platformdirs
  given by the platformdirs library.  To remove this warning and
  see the appropriate new directories, set the environment variable
  `JUPYTER_PLATFORM_DIRS=1` and then run `jupyter --paths`.
  The use of platformdirs will be the default in `jupyter_core` v6
    from jupyter_core.paths import jupyter_data_dir, jupyter_runtime_dir, secure_write

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================================================================== short test summary info ====================================================================== 
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_mask_initialization_implementation - AssertionError: initialize_validation_tracking() should be called for category 'heating'
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_series_initialization_implementation - AssertionError: create_retrofit_only_series() should be called at least once
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_final_masking_implementation - AssertionError: apply_final_masking() should be called
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_all_validation_steps - AssertionError: Step 1: initialize_validation_tracking() should be called for at least one category
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_error_handling_for_invalid_scc - Failed: DID NOT RAISE <class 'ValueError'>
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_category_error_graceful_continuation - Failed: adoption_decision should handle category errors gracefully, but raised: At least some categories should be processed
FAILED test_determine_adoption_potential_sensitivity_V2.py::test_climate_sensitivity_parameter - AssertionError: Should not have 'lower' SCC columns when climate_sensitivity=False
============================================================== 7 failed, 29 passed, 1 warning in 4.08s ============================================================== 
