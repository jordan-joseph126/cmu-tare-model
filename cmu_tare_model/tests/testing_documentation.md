# ChatGPT Prompt - Documentation for Code/Functions:

file.py uses {}, {}, and other imports from the cmu_tare_model package. Some functions lack adequate docstrings and inline comments.  

    Please:

    1. Add Google-style docstrings to each function, including:
    - A brief one-line summary.
    - Parameter descriptions with types.
    - Return descriptions with type.
    - Any exceptions raised.
    2. Add inline comments only for non-trivial or complex lines to clarify logic.
    3. Add typehints to all functions
    4. Do not rename any functions, parameters, or variables.
    5. Do not change or remove any existing lines of code.
    6. Do not rearrange or refactor the code structure.



# ChatGPT Prompt - Additional Information for Unit Tests (Edge Cases): 

    What are some edge cases for this set of functions? What additional information would help you generate pytest tests?


# ChatGPT Prompt for Requesting ChatGPT to Write a Pytest Test Generation Prompt

Remember the important edge case and additional info for writing the pytest tests for the code mentioned above:

    """
    TEXT HERE
    """

 Write a prompt that I can use to ask ChatGPT to generate pytest tests for the code above. Please follow **best practices** in prompt engineering by clearly stating: 
- **Context**: What the function does.  
- **Objectives**: What kind of tests are needed (edge-case coverage, exception handling, etc.).  
- **Constraints/Requirements**: Use of pytest.raises, parametrization, no actual I/O, etc.

Please:
1. Write **pytest functions** that thoroughly test success, failure, and boundary conditions.
2. Use **fixtures** if that helps simplify code or avoid repetition.
3. Provide short **docstrings** or inline comments within the test code to describe the purpose of each test.

Your final answer should be the fully-written test file in Python syntax so that I can copy and run immediately with pytest.


"""
test_process_fuel_price_data.py

Pytest test suite for the following functions in process_fuel_price_data.py:
1. project_future_prices
2. create_lookup_fuel_price
3. map_location_to_census_division

This test suite covers:
- Success cases for valid inputs
- Edge cases with missing or invalid inputs
- Failure cases with mismatched data types or missing columns
- Boundary conditions such as extreme values and missing years
- Ensures correct return types and outputs

Author: Jordan Joseph
"""

import pytest
import pandas as pd
from typing import Dict, Any
from cmu_tare_model.functions.process_fuel_price_data import (
    project_future_prices,
    create_lookup_fuel_price,
    map_location_to_census_division
)

# --------------------------------------------------------------------------
#                                FIXTURES
# --------------------------------------------------------------------------
@pytest.fixture
def sample_fuel_data() -> pd.DataFrame:
    """
    Returns a sample DataFrame simulating raw fuel price data that might
    be used as input for create_lookup_fuel_price and project_future_prices.
    """
    data = {
        "Year": [2020, 2020, 2021, 2021],
        "Fuel_Type": ["Gasoline", "Diesel", "Gasoline", "Diesel"],
        "Location": ["CA", "CA", "NY", "NY"],
        "Price": [2.80, 3.00, 2.90, 3.10],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_projection_factors() -> Dict[str, Dict[str, float]]:
    """
    Returns a sample dictionary simulating projection factors for different
    policy scenarios, keyed by Location/Census Division and Year.
    """
    # Example structure: { "CA": {"2022": 1.05, "2023": 1.07}, ... }
    return {
        "CA": {
            "2022": 1.05,
            "2023": 1.07,
            "2050": 1.60,  # extreme future
        },
        "NY": {
            "2022": 1.04,
            "2023": 1.06,
            "2050": 1.55,  # extreme future
        },
    }


@pytest.fixture
def sample_census_division_map() -> Dict[str, str]:
    """
    Returns a sample dictionary mapping states or locations to census divisions.
    """
    return {
        "CA": "West",
        "NY": "Northeast",
        "TX": "South",
    }


# --------------------------------------------------------------------------
#                               TESTS: project_future_prices
# --------------------------------------------------------------------------
def test_project_future_prices_basic_success(sample_fuel_data, sample_projection_factors):
    """
    Test that project_future_prices returns a valid DataFrame with expected columns
    and projected values when valid inputs are given.
    """

def test_project_future_prices_missing_factors(sample_fuel_data):
    """
    Test that project_future_prices handles missing projection factors appropriately.
    Expect an empty or partially filled DataFrame, or an exception.
    """

def test_project_future_prices_invalid_data_type():
    """
    Test that project_future_prices raises TypeError or ValueError if the input DataFrame
    or projection_factors are of invalid types.
    """

@pytest.mark.parametrize(
    "policy_scenario",
    ["reference", "high_growth", "low_growth"]
)
def test_project_future_prices_multiple_scenarios(
    sample_fuel_data, sample_projection_factors, policy_scenario
):
    """
    Parametrized test to ensure function handles multiple policy scenarios
    gracefully. Assumes that the function internally adjusts logic based on
    scenario.
    """
    projected_df = project_future_prices(
        df=sample_fuel_data,
        projection_factors=sample_projection_factors,
        policy_scenario=policy_scenario
    )

    # Very basic checks for demonstration
    assert "Year" in projected_df.columns
    assert "Price" in projected_df.columns


# --------------------------------------------------------------------------
#                             TESTS: create_lookup_fuel_price
# --------------------------------------------------------------------------
def test_create_lookup_fuel_price_basic_success(sample_fuel_data):
    """
    Test that create_lookup_fuel_price returns a nested dictionary with the expected structure.
    """

def test_create_lookup_fuel_price_missing_columns():
    """
    Test that create_lookup_fuel_price raises a KeyError if the input DataFrame
    lacks required columns.
    """

def test_create_lookup_fuel_price_empty_data():
    """
    Test that create_lookup_fuel_price handles an empty DataFrame by returning
    an empty dictionary.
    """

# --------------------------------------------------------------------------
#                     TESTS: map_location_to_census_division
# --------------------------------------------------------------------------
def test_map_location_to_census_division_success(sample_census_division_map):
    """
    Test that map_location_to_census_division correctly maps location strings
    to their census division.
    """


def test_map_location_to_census_division_unknown(sample_census_division_map):
    """
    Test that map_location_to_census_division returns a fallback value (e.g., "Unknown")
    when the location is not in the map.
    """


def test_map_location_to_census_division_invalid_input(sample_census_division_map):
    """
    Test that map_location_to_census_division handles invalid input types gracefully.
    """


GENERAL PROMPT

### **Files for Review:**
- Main script: `[INSERT_MAIN_FILE_NAME_HERE.py]`
- Additional module/script(s) imported:
  - Module: `[MODULE_NAME_HERE.py]` – Imported as `[alias or module name in main script]`
  - Module: `[ADDITIONAL_MODULE_NAME_HERE.py]` – Imported as `[alias or module name in main script]`

### **Task Description:**
Carefully review the provided main script and the additional imported module(s). Ensure a comprehensive understanding of their logic, functionality, and interactions. Pay particular attention to the creation of lookup dictionaries within the attached scripts, ensuring you understand fully:
- The input data structure.
- The transformation logic and algorithms applied.
- The resulting output structures.
- Functional differences, if any, between the scripts.

### **Review and Analysis Objectives:**
- Confirm your complete understanding of all attached code (main file and modules).
- Clearly note and document:
  - Any similarities or differences between multiple scripts performing similar functions.
  - Potential redundancies, optimization points, or areas of confusion.

### **Document Changes Clearly:**
- Clearly highlight every change made to the code since the version from `[INSERT_PREVIOUS_VERSION_DATE]`.
- Provide a detailed changelog including:
  - Specific descriptions of modifications, additions, and removals.
  - Explicit code snippets where relevant to clearly indicate what has changed.

**Example Changelog Format:**
```
CHANGELOG
----------

Date: [Insert Date Here]
- Modified function `[function_name]()`:
  - Changed [brief description of change]
  - Reason: [explain why this change was necessary]

- Added new variable `[variable_name]`:
  - Purpose: [describe the function or purpose of this variable]

- Removed redundant loop in `[section or function]`:
  - Reason: [justify the removal clearly and concisely]
```

### **Naming Conventions:**
- Verify that all naming conventions precisely match those used in the original codebase. Identify and document any discrepancies or inconsistencies.

### **Deliverable:**
Provide a comprehensive report structured as follows:

- **Understanding and Summary:**
  - Clear overview and summary of the logic, functionality, and purpose of all provided scripts and their interactions.

- **Detailed Script Comparison:**
  - Table or structured comparison clearly indicating differences and similarities between scripts, especially around lookup dictionary creation logic.

- **Documented Changes:**
  - Comprehensive changelog documenting every modification since `[INSERT_PREVIOUS_VERSION_DATE]`.

- **Naming Convention Consistency Report:**
  - Explicitly document any naming convention inconsistencies or confirm consistency.

---

### **Future Tasks Placeholder:**

Clearly indicate here the tasks you'd like to accomplish in the next iteration:

- `[INSERT TASK 1 HERE]`
- `[INSERT TASK 2 HERE]`
- `[INSERT TASK 3 HERE]`

---

