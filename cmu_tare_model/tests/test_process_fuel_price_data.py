# """
# Updated test_process_fuel_price_data.py

# Enhanced Pytest test suite for the following functions in process_fuel_price_data.py:
# 1. project_future_prices
# 2. create_lookup_fuel_price
# 3. map_location_to_census_division

# This test suite covers:
# - Success cases for valid inputs
# - Edge cases with missing or invalid inputs
# - Failure cases with mismatched data types or missing columns
# - Boundary conditions such as extreme values and missing years
# - Ensures correct return types and outputs
# - Additional edge cases and performance validations

# Author: Jordan Joseph
# """

# import pytest
# import pandas as pd
# from typing import Dict, Any
# from cmu_tare_model.private_impact.process_fuel_price_data import (
#     project_future_prices,
#     create_lookup_fuel_price,
#     map_location_to_census_division
# )

# # --------------------------------------------------------------------------
# #                                FIXTURES
# # --------------------------------------------------------------------------
# @pytest.fixture
# def sample_fuel_data() -> pd.DataFrame:
#     """
#     Returns a sample DataFrame simulating raw fuel price data.
#     """
#     data = {
#         "Year": [2022, 2022],
#         "Fuel_Type": ["Gasoline", "Diesel"],
#         "Location": ["CA", "NY"],
#         "Price": [2.80, 3.00],
#     }
#     return pd.DataFrame(data)

# @pytest.fixture
# def sample_projection_factors() -> Dict[str, Dict[str, float]]:
#     """
#     Returns a sample dictionary simulating projection factors.
#     """
#     return {
#         "CA": {
#             "2022": 1.05,
#             "2023": 1.07,
#             "2050": 1.60,
#         },
#         "NY": {
#             "2022": 1.04,
#             "2023": 1.06,
#             "2050": 1.55,
#         },
#     }

# @pytest.fixture
# def sample_census_division_map() -> Dict[str, str]:
#     """
#     Returns a sample dictionary mapping locations to census divisions.
#     """
#     return {
#         "CA": "West",
#         "NY": "Northeast",
#         "TX": "South",
#     }

# # --------------------------------------------------------------------------
# #                               TESTS: project_future_prices
# # --------------------------------------------------------------------------
# def test_project_future_prices_basic_success(sample_fuel_data, sample_projection_factors):
#     """
#     Validates success for valid input data and projection factors.
#     """
#     df = sample_fuel_data
#     projection_factors = sample_projection_factors
#     policy_scenario = "reference"

#     result = project_future_prices(
#         row=df.iloc[0],
#         factor_dict=projection_factors,
#         policy_scenario=policy_scenario,
#     )
#     assert isinstance(result, pd.Series), "Output should be a pandas Series"
#     assert len(result) > 0, "Projection should contain future years"
#     assert all(year in result.index for year in range(2022, 2051)), "Missing years in projection"

# def test_project_future_prices_partial_missing_factors(sample_fuel_data):
#     """
#     Handles partial projection factors gracefully.
#     """
#     projection_factors = {
#         "CA": {"2022": 1.05, "2023": 1.07},  # Missing 2050
#     }
#     row = sample_fuel_data.iloc[0]

#     result = project_future_prices(
#         row=row,
#         factor_dict=projection_factors,
#         policy_scenario="reference",
#     )
#     assert "2050_fuelPrice_perkWh" not in result.index, "Missing factors should result in skipped years"

# def test_project_future_prices_invalid_row_format(sample_projection_factors):
#     """
#     Validates that improperly formatted rows raise errors.
#     """
#     row = {"invalid_key": "invalid_value"}
#     with pytest.raises(KeyError):
#         project_future_prices(row=row, factor_dict=sample_projection_factors, policy_scenario="reference")

# def test_project_future_prices_national_fallback(sample_projection_factors):
#     """
#     Ensures fallback to national factors when regional factors are missing.
#     """
#     factors = {"National": {"Gasoline": {"2022": 1.05, "2050": 1.50}}}
#     row = pd.Series({
#         "census_division": "UnknownRegion",
#         "fuel_type": "Gasoline",
#         "2022_fuelPrice_perkWh": 3.00,
#     })

#     result = project_future_prices(
#         row=row,
#         factor_dict=factors,
#         policy_scenario="reference",
#     )
#     assert "2050_fuelPrice_perkWh" in result.index, "Fallback should calculate prices based on national factors"

# # --------------------------------------------------------------------------
# #                             TESTS: create_lookup_fuel_price
# # --------------------------------------------------------------------------
# def test_create_lookup_fuel_price_success(sample_fuel_data):
#     """
#     Validates that the lookup dictionary is correctly structured.
#     """
#     result = create_lookup_fuel_price(sample_fuel_data, policy_scenario="reference")
#     assert isinstance(result, dict), "Output should be a dictionary"
#     assert "CA" in result, "Location should exist in dictionary"
#     assert "Gasoline" in result["CA"], "Fuel type should exist in nested dictionary"

# def test_create_lookup_fuel_price_duplicate_rows():
#     """
#     Validates behavior with duplicate rows in input.
#     """
#     data = pd.DataFrame({
#         "Location": ["CA", "CA"],
#         "Fuel_Type": ["Gasoline", "Gasoline"],
#         "2022_fuelPrice_perkWh": [2.8, 2.9],
#     })
#     result = create_lookup_fuel_price(data, policy_scenario="reference")
#     assert len(result["CA"]["Gasoline"]["reference"]) == 1, "Duplicate rows should not cause duplicate keys"

# # --------------------------------------------------------------------------
# #                     TESTS: map_location_to_census_division
# # --------------------------------------------------------------------------
# def test_map_location_to_census_division_whitespace(sample_census_division_map):
#     """
#     Validates behavior for whitespace or empty location inputs.
#     """
#     result = map_location_to_census_division("   ", sample_census_division_map)
#     assert result == "   ", "Whitespace input should return as is"

# def test_map_location_to_census_division_case_insensitivity(sample_census_division_map):
#     """
#     Validates that location mapping is case-insensitive.
#     """
#     result = map_location_to_census_division("ca", sample_census_division_map)
#     assert result == "West", "Mapping should be case-insensitive"

"""
Updated test_process_fuel_price_data.py

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
- Additional edge cases and performance validations

Author: Jordan Joseph
"""

import pytest
import pandas as pd
from typing import Dict
from cmu_tare_model.private_impact.data_processing.create_lookup_fuel_prices import (
    project_future_prices,
    create_lookup_fuel_price,
    map_location_to_census_division,
)

# --------------------------------------------------------------------------
#                                FIXTURES
# --------------------------------------------------------------------------
@pytest.fixture
def sample_fuel_data() -> pd.DataFrame:
    """
    Returns a sample DataFrame simulating raw fuel price data for testing.
    """
    data = {
        "location_map": ["CA", "NY"],
        "fuel_type": ["Gasoline", "Diesel"],
        "2022_fuelPrice_perkWh": [2.80, 3.00],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_projection_factors() -> Dict:
    """
    Returns a dictionary simulating projection factors for different locations, fuels, and years.
    """
    return {
        ("CA", "Gasoline", "reference"): {2022: 1.05, 2050: 1.60},
        ("NY", "Diesel", "reference"): {2022: 1.04, 2050: 1.55},
    }


@pytest.fixture
def sample_census_division_map() -> Dict[str, str]:
    """
    Returns a dictionary mapping states to census divisions.
    """
    return {"CA": "West", "NY": "Northeast"}


# --------------------------------------------------------------------------
#                               TESTS: project_future_prices
# --------------------------------------------------------------------------
def test_project_future_prices_basic_success(sample_fuel_data, sample_projection_factors):
    """
    Test project_future_prices with valid input data.
    """
    row = sample_fuel_data.iloc[0]  # Test with the first row
    factors = sample_projection_factors
    policy_scenario = "reference"

    result = project_future_prices(row=row, factor_dict=factors, policy_scenario=policy_scenario)

    assert isinstance(result, pd.Series), "Result should be a pandas Series"
    assert "2022_fuelPrice_perkWh" in result.index, "Missing projected price for 2022"
    assert "2050_fuelPrice_perkWh" in result.index, "Missing projected price for 2050"


def test_project_future_prices_missing_factors(sample_fuel_data):
    """
    Test project_future_prices with missing projection factors.
    """
    row = sample_fuel_data.iloc[0]
    factors = {}  # No projection factors
    policy_scenario = "reference"

    result = project_future_prices(row=row, factor_dict=factors, policy_scenario=policy_scenario)

    assert result.empty, "Expected an empty Series when factors are missing"


def test_project_future_prices_partial_missing_factors(sample_fuel_data, sample_projection_factors):
    """
    Test project_future_prices with partial projection factors.
    """
    row = sample_fuel_data.iloc[0]
    # Remove 2050 factors
    factors = {
        ("CA", "Gasoline", "reference"): {2022: 1.05},
    }
    policy_scenario = "reference"

    result = project_future_prices(row=row, factor_dict=factors, policy_scenario=policy_scenario)

    assert "2050_fuelPrice_perkWh" not in result.index, "Missing factors should result in skipped years"


def test_project_future_prices_invalid_data_type():
    """
    Test project_future_prices with invalid data types.
    """
    row = "InvalidRow"
    factors = {"CA": {2022: 1.05}}
    policy_scenario = "reference"

    with pytest.raises(TypeError):
        project_future_prices(row=row, factor_dict=factors, policy_scenario=policy_scenario)


# --------------------------------------------------------------------------
#                             TESTS: create_lookup_fuel_price
# --------------------------------------------------------------------------
def test_create_lookup_fuel_price_basic_success(sample_fuel_data):
    """
    Test create_lookup_fuel_price with valid data.
    """
    policy_scenario = "reference"
    result = create_lookup_fuel_price(sample_fuel_data, policy_scenario)

    assert isinstance(result, dict), "Result should be a dictionary"
    assert "CA" in result, "Location key missing in the result"
    assert "Gasoline" in result["CA"], "Fuel type missing in the nested dictionary"


def test_create_lookup_fuel_price_empty_data():
    """
    Test create_lookup_fuel_price with an empty DataFrame.
    """
    df_empty = pd.DataFrame(columns=["location_map", "fuel_type", "2022_fuelPrice_perkWh"])
    policy_scenario = "reference"
    result = create_lookup_fuel_price(df_empty, policy_scenario)

    assert result == {}, "Expected an empty dictionary for empty DataFrame"


def test_create_lookup_fuel_price_missing_columns():
    """
    Test create_lookup_fuel_price with missing required columns.
    """
    df_missing = pd.DataFrame({
        "location_map": ["CA"],
        "fuel_type": ["Gasoline"],
        # Missing '2022_fuelPrice_perkWh'
    })
    policy_scenario = "reference"

    with pytest.raises(KeyError):
        create_lookup_fuel_price(df_missing, policy_scenario)


# --------------------------------------------------------------------------
#                     TESTS: map_location_to_census_division
# --------------------------------------------------------------------------
def test_map_location_to_census_division_success(sample_census_division_map):
    """
    Test map_location_to_census_division with known mappings.
    """
    assert map_location_to_census_division("CA") == "West"
    assert map_location_to_census_division("NY") == "Northeast"


def test_map_location_to_census_division_unknown():
    """
    Test map_location_to_census_division with an unmapped location.
    """
    result = map_location_to_census_division("UnknownState")
    assert result == "UnknownState", "Expected input value to be returned if not mapped"


def test_map_location_to_census_division_invalid_input():
    """
    Test map_location_to_census_division with invalid input.
    """
    with pytest.raises(TypeError):
        map_location_to_census_division(123)