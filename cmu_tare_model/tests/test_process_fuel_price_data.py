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
from cmu_tare_model.utils.process_fuel_price_data import (
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
    projected_df = project_future_prices(
        df=sample_fuel_data,
        projection_factors=sample_projection_factors,
        policy_scenario="reference"  # hypothetical scenario
    )
    
    # Check that result is a DataFrame
    assert isinstance(projected_df, pd.DataFrame), "Output should be a DataFrame"
    
    # Check that certain columns exist
    expected_columns = {"Year", "Fuel_Type", "Location", "Price"}
    assert expected_columns.issubset(projected_df.columns), "Missing expected columns"

    # Check that future years are included (e.g., 2022 to 2050)
    assert projected_df["Year"].max() >= 2050, "Projection did not extend to 2050"

    # Check that prices are projected (i.e., higher than original in this sample)
    original_prices = sample_fuel_data["Price"].tolist()
    projected_prices = projected_df[projected_df["Year"] == 2050]["Price"].tolist()
    assert all(p_new > p_old for p_new, p_old in zip(projected_prices, original_prices)), \
        "Projected prices for 2050 should be greater than original prices"


def test_project_future_prices_missing_factors(sample_fuel_data):
    """
    Test that project_future_prices handles missing projection factors appropriately.
    Expect an empty or partially filled DataFrame, or an exception.
    """
    # Provide an empty projection_factors dict
    projection_factors_empty = {}

    with pytest.raises(KeyError):
        # Expect a KeyError or similar if location/years can't be found
        project_future_prices(
            df=sample_fuel_data,
            projection_factors=projection_factors_empty,
            policy_scenario="reference"
        )


def test_project_future_prices_invalid_data_type():
    """
    Test that project_future_prices raises TypeError or ValueError if the input DataFrame
    or projection_factors are of invalid types.
    """
    with pytest.raises(TypeError):
        # Invalid type for df
        project_future_prices(
            df=["invalid", "data"],
            projection_factors={"CA": {"2022": 1.05}},
            policy_scenario="reference"
        )

    with pytest.raises(TypeError):
        # Invalid type for projection_factors
        df = pd.DataFrame({"Year": [2020], "Price": [2.5]})
        project_future_prices(
            df=df,
            projection_factors=["invalid_factors"],
            policy_scenario="reference"
        )


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
    lookup_dict = create_lookup_fuel_price(sample_fuel_data)

    assert isinstance(lookup_dict, dict), "Output should be a dictionary"

    # Example nested dict structure check
    # {
    #   'CA': {
    #       'Gasoline': {2020: price, 2021: price, ...},
    #       'Diesel':   {2020: price, 2021: price, ...},
    #       ...
    #   },
    #   'NY': {
    #       ...
    #   },
    #   ...
    # }
    for location, fuel_data in lookup_dict.items():
        assert isinstance(fuel_data, dict), "Second-level value should be a dict"
        for fuel_type, year_dict in fuel_data.items():
            assert isinstance(year_dict, dict), "Third-level value should be a dict"
            for year, price in year_dict.items():
                assert isinstance(year, int), "Year should be an integer"
                assert isinstance(price, float), "Price should be a float"


def test_create_lookup_fuel_price_missing_columns():
    """
    Test that create_lookup_fuel_price raises a KeyError if the input DataFrame
    lacks required columns.
    """
    df_missing = pd.DataFrame({
        # Missing 'Location' and 'Fuel_Type'
        "Year": [2020, 2021],
        "Price": [2.80, 2.90]
    })

    with pytest.raises(KeyError):
        create_lookup_fuel_price(df_missing)


def test_create_lookup_fuel_price_empty_data():
    """
    Test that create_lookup_fuel_price handles an empty DataFrame by returning
    an empty dictionary.
    """
    df_empty = pd.DataFrame(columns=["Year", "Fuel_Type", "Location", "Price"])
    lookup_dict = create_lookup_fuel_price(df_empty)
    assert lookup_dict == {}, "Expected an empty dictionary for empty DataFrame"


# --------------------------------------------------------------------------
#                     TESTS: map_location_to_census_division
# --------------------------------------------------------------------------
def test_map_location_to_census_division_success(sample_census_division_map):
    """
    Test that map_location_to_census_division correctly maps location strings
    to their census division.
    """
    # Testing known mappings
    assert map_location_to_census_division("CA", sample_census_division_map) == "West"
    assert map_location_to_census_division("NY", sample_census_division_map) == "Northeast"
    assert map_location_to_census_division("TX", sample_census_division_map) == "South"


def test_map_location_to_census_division_unknown(sample_census_division_map):
    """
    Test that map_location_to_census_division returns a fallback value (e.g., "Unknown")
    when the location is not in the map.
    """
    result = map_location_to_census_division("FL", sample_census_division_map)
    assert result == "Unknown", "Expected 'Unknown' for unmapped location"


def test_map_location_to_census_division_invalid_input(sample_census_division_map):
    """
    Test that map_location_to_census_division handles invalid input types gracefully.
    """
    with pytest.raises(TypeError):
        map_location_to_census_division(123, sample_census_division_map)

    with pytest.raises(TypeError):
        # Passing None should raise an error if not handled
        map_location_to_census_division(None, sample_census_division_map)