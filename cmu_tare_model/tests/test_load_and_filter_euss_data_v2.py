"""
test_app.py

Pytest tests for the module that handles user prompts, fuel standardization,
and filtering/masking rows based on fuel types and technology types.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Assuming all functions are defined in a file named my_module.py
# Adjust the import statement to match your actual filename:
from cmu_tare_model.energy_consumption_and_metadata.load_and_filter_euss_data_v2 import(
    get_menu_choice,
    get_state_choice,
    get_city_choice,
    standardize_fuel_name,
    preprocess_fuel_data,
    apply_fuel_filter,
    apply_technology_filter,
    debug_filters,
    extract_city_name,
    df_enduse_refactored
)

# -------------------------------------------------------------------------
#                           FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """
    Returns a sample DataFrame containing ALL columns expected by df_enduse_refactored().
    This dummy data can be modified or expanded to suit specific testing needs.
    """
    data = {
        # Basic structure & metadata columns
        'in.sqft': [1000, 2000, 1500, 1200, 900],
        'in.census_region': ['West', 'South', 'South', 'Northeast', 'Midwest'],
        'in.census_division': ['Division9', 'Division7', 'Division7', 'Division1', 'Division4'],
        'in.census_division_recs': ['Division9A', 'Division7B', 'Division7C', 'Division1A', 'Division4A'],
        'in.building_america_climate_zone': ['Marine', 'Hot-Dry', 'Hot-Humid', 'Cold', 'Mixed-Humid'],
        'in.reeds_balancing_area': ['Area1', 'Area2', 'Area2', 'Area3', 'Area4'],
        'in.generation_and_emissions_assessment_region': ['GEA1', 'GEA2', 'GEA2', 'GEA1', 'GEA4'],
        'in.state': ['CA', 'TX', 'TX', 'NY', 'IL'],
        'in.city': ['CA, San Diego', 'TX, Dallas', 'TX, Houston', 'NY, Albany', 'IL, Chicago'],
        'in.county': ['CountyA', 'CountyB', 'CountyB', 'CountyC', 'CountyD'],
        'in.puma': ['PUMA1', 'PUMA2', 'PUMA2', 'PUMA3', 'PUMA4'],
        'in.county_and_puma': [
            'CountyA, PUMA1', 
            'CountyB, PUMA2', 
            'CountyB, PUMA2', 
            'CountyC, PUMA3', 
            'CountyD, PUMA4'
        ],
        'in.weather_file_city': ['SanDiego', 'Dallas', 'Houston', 'Albany', 'Chicago'],
        'in.weather_file_longitude': [-117.1611, -96.7970, -95.3698, -73.7562, -87.6298],
        'in.weather_file_latitude': [32.7157, 32.7767, 29.7604, 42.6526, 41.8781],
        'in.geometry_building_type_recs': [
            'Single-Family', 'Single-Family', 'Multi-Family', 'Mobile Home', 'Single-Family'
        ],
        'in.income': [70000, 120000, 50000, 35000, 92000],
        'in.federal_poverty_level': [1.2, 2.0, 1.5, 0.9, 1.8],
        'in.occupants': [3, 4, 2, 1, 2],
        'in.tenure': ['Owned', 'Owned', 'Rented', 'Rented', 'Owned'],
        'in.vacancy_status': ['Occupied', 'Occupied', 'Occupied', 'Vacant', 'Occupied'],

        # Heating & HVAC columns
        'in.heating_fuel': ['Electric', 'Gas', 'Oil', 'Propane', 'Coal'],  # includes invalid 'Coal'
        'in.hvac_heating_type_and_fuel': [
            'Electricity ASHP', 'Natural Gas Fuel Furnace', 
            'Fuel Oil Fuel Furnace', 'Propane Fuel Furnace', 
            'Wood Stove'  # invalid
        ],
        'in.hvac_cooling_type': [
            'Central AC', 'Room AC', None, 'Central AC', 'None'
        ],
        'in.vintage': ['Pre-1980', '1990s', '2000s', '1980s', '2000s'],
        'in.hvac_heating_efficiency': [0.9, 0.85, 0.8, 0.95, 0.7],

        # Heating consumption
        'out.electricity.heating.energy_consumption.kwh': [200, 0, 10, 100, 0],
        'out.fuel_oil.heating.energy_consumption.kwh': [0, 0, 50, 0, 10],
        'out.natural_gas.heating.energy_consumption.kwh': [0, 300, 0, 0, 50],
        'out.propane.heating.energy_consumption.kwh': [0, 0, 0, 200, 0],

        # Water heating columns
        'in.water_heater_fuel': ['Electric', 'Gas', 'Propane', 'Electric', 'Oil'],
        'in.water_heater_efficiency': [
            'Electric Standard', 'Natural Gas Standard', 
            'Propane Premium', 'Electric Standard', 
            'Fuel Oil Standard'
        ],
        'out.electricity.hot_water.energy_consumption.kwh': [20, 15, 0, 25, 5],
        'out.fuel_oil.hot_water.energy_consumption.kwh': [0, 0, 0, 0, 12],
        'out.natural_gas.hot_water.energy_consumption.kwh': [0, 30, 0, 0, 0],
        'out.propane.hot_water.energy_consumption.kwh': [0, 0, 18, 0, 0],

        # Clothes drying columns
        'in.clothes_dryer': ['Electric', 'Gas', None, 'Propane', 'Oil'],
        'out.electricity.clothes_dryer.energy_consumption.kwh': [8, 0, 0, 0, 5],
        'out.natural_gas.clothes_dryer.energy_consumption.kwh': [0, 10, 0, 0, 0],
        'out.propane.clothes_dryer.energy_consumption.kwh': [0, 0, 0, 10, 0],

        # Cooking columns
        'in.cooking_range': ['Electric Range', 'Propane Range', 'Gas Range', 'Oil Range', np.nan],
        'out.electricity.range_oven.energy_consumption.kwh': [5, 0, 0, 0, 1],
        'out.natural_gas.range_oven.energy_consumption.kwh': [0, 0, 5, 0, 0],
        'out.propane.range_oven.energy_consumption.kwh': [0, 10, 0, 0, 0],
    }
    return pd.DataFrame(data)

# -------------------------------------------------------------------------
#                  TESTS FOR USER INPUT FUNCTIONS
# -------------------------------------------------------------------------

def test_get_menu_choice_valid():
    """Test that get_menu_choice() returns valid input from a set of choices."""
    with patch('builtins.input', side_effect=['n', 'N']):
        # First try 'n' (lowercase) => uppercase it => 'N'
        result = get_menu_choice("Prompt", ['N', 'Y'])
        assert result == 'N'

def test_get_menu_choice_invalid_then_valid():
    """Test re-prompting until user provides a valid choice."""
    with patch('builtins.input', side_effect=['X', 'Z', 'y']):
        result = get_menu_choice("Prompt", ['N', 'Y'])
        assert result == 'Y'

def test_get_state_choice_valid(sample_df):
    """Test picking a valid two-letter state abbreviation."""
    with patch('builtins.input', side_effect=['tx']):
        result = get_state_choice(sample_df)
        assert result == 'TX'

def test_get_state_choice_invalid_then_valid(sample_df):
    """Test re-prompting until a valid state is found in df_copy."""
    with patch('builtins.input', side_effect=['ZZ', 'Ca']):
        # 'ZZ' is invalid; 'Ca' becomes 'CA' which should be valid
        result = get_state_choice(sample_df)
        assert result == 'CA'

def test_get_city_choice_valid(sample_df):
    """Test choosing a valid city for a known state in df_copy."""
    with patch('builtins.input', side_effect=['Dallas']):
        # Should find "TX, Dallas" in the DataFrame
        result = get_city_choice(sample_df, 'TX')
        assert result == 'Dallas'

def test_get_city_choice_invalid(sample_df):
    """Test re-prompting until valid city name is given."""
    with patch('builtins.input', side_effect=['InvalidCity', 'Houston']):
        result = get_city_choice(sample_df, 'TX')
        assert result == 'Houston'

# -------------------------------------------------------------------------
#                TESTS FOR FUEL / TECHNOLOGY FUNCTIONS
# -------------------------------------------------------------------------

@pytest.mark.parametrize("input_val,expected", [
    ("Electric Heater", "Electricity"),
    ("GAS furnace", "Natural Gas"),
    ("Propane 123", "Propane"),
    ("oil-based", "Fuel Oil"),
    ("Coal", None),
    (np.nan, None),
    (123, None),
    ("", None),
])
def test_standardize_fuel_name(input_val, expected):
    """Test standardize_fuel_name() with various possible inputs."""
    assert standardize_fuel_name(input_val) == expected

def test_preprocess_fuel_data(sample_df):
    """Test that preprocess_fuel_data() applies standardize_fuel_name to a column."""
    col_name = 'in.clothes_dryer'
    modified_df = preprocess_fuel_data(sample_df, col_name)
    
    # Check if each row in the processed column is one of the standardized types or None
    allowed_types = {"Electricity", "Natural Gas", "Propane", "Fuel Oil", None}
    assert all(x in allowed_types for x in modified_df[col_name].unique())

def test_apply_fuel_filter_filter_mode(sample_df):
    """
    Test apply_fuel_filter in 'filter' mode for the 'heating' category 
    to ensure only rows with valid fuels remain.
    """
    df_copy = sample_df.copy()
    df_copy['base_heating_fuel'] = df_copy['in.heating_fuel'].apply(standardize_fuel_name)
    
    # Enable is 'Yes', so the filter is applied
    filtered = apply_fuel_filter(df_copy, 'heating', enable='Yes', mode='filter')
    valid_fuels = {'Electricity', 'Natural Gas', 'Propane', 'Fuel Oil'}
    assert all(fuel in valid_fuels for fuel in filtered['base_heating_fuel'])

def test_apply_fuel_filter_mask_mode(sample_df):
    """
    Test apply_fuel_filter in 'mask' mode for 'heating' category.
    Invalid fuels lead to NaN in consumption columns, but rows remain.
    """
    df_copy = sample_df.copy()
    df_copy['base_heating_fuel'] = df_copy['in.heating_fuel'].apply(standardize_fuel_name)
    original_len = len(df_copy)

    masked = apply_fuel_filter(df_copy, 'heating', enable='Yes', mode='mask')
    assert len(masked) == original_len  # No rows dropped in 'mask' mode

    # Check that rows with invalid fuel have NaN in base_* consumption
    invalid_idx = masked[~masked['base_heating_fuel'].isin(['Electricity','Natural Gas','Propane','Fuel Oil'])].index
    for fuel in ['electricity', 'fuelOil', 'naturalGas', 'propane']:
        col_name = f'base_{fuel}_heating_consumption'
        assert masked.loc[invalid_idx, col_name].isna().all()

def test_apply_fuel_filter_disabled(sample_df):
    """If enable != 'Yes', the function should return the df unchanged."""
    df_copy = sample_df.copy()
    filtered = apply_fuel_filter(df_copy, 'heating', enable='No', mode='filter')
    assert filtered.equals(df_copy)

def test_apply_technology_filter_filter_mode(sample_df):
    """
    Test apply_technology_filter() for 'heating' in filter mode. 
    Only valid heating technologies should remain.
    """
    df_copy = sample_df.copy()
    # Use 'in.hvac_heating_type_and_fuel' as 'heating_type'
    df_copy['heating_type'] = df_copy['in.hvac_heating_type_and_fuel']
    filtered = apply_technology_filter(df_copy, 'heating', 'Yes', mode='filter')
    
    allowed_tech = {
        'Electricity ASHP', 'Electricity Baseboard', 'Electricity Electric Boiler', 'Electricity Electric Furnace',
        'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace',
        'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
        'Propane Fuel Boiler', 'Propane Fuel Furnace'
    }
    assert all(t in allowed_tech for t in filtered['heating_type'])

def test_apply_technology_filter_mask_mode(sample_df):
    """
    Test apply_technology_filter() for 'waterHeating' in mask mode.
    Invalid technologies get NaN in consumption columns.
    """
    df_copy = sample_df.copy()
    df_copy['waterHeating_type'] = df_copy['in.water_heater_efficiency']
    masked = apply_technology_filter(df_copy, 'waterHeating', 'Yes', mode='mask')

    # Rows with unknown or invalid water heater tech should have NaN in these consumption columns
    invalid_mask = ~masked['waterHeating_type'].isin([
        'Electric Heat Pump, 80 gal', 'Electric Premium', 'Electric Standard',
        'Fuel Oil Premium', 'Fuel Oil Standard',
        'Natural Gas Premium', 'Natural Gas Standard',
        'Propane Premium', 'Propane Standard'
    ])
    invalid_indices = masked[invalid_mask].index

    for fuel in ['electricity', 'fuelOil', 'naturalGas', 'propane']:
        col_name = f'base_{fuel}_waterHeating_consumption'
        assert masked.loc[invalid_indices, col_name].isna().all()

def test_apply_technology_filter_disabled(sample_df):
    """If category not in ['heating','waterHeating'] or enable != 'Yes', no changes should apply."""
    df_copy = sample_df.copy()
    modified = apply_technology_filter(df_copy, 'clothesDrying', 'Yes', mode='filter')
    assert modified.equals(df_copy)

# -------------------------------------------------------------------------
#                          TESTS FOR OTHER UTILS
# -------------------------------------------------------------------------

def test_debug_filters_empty(capfd):
    """Test debug_filters() with an empty DataFrame."""
    empty_df = pd.DataFrame()
    debug_filters(empty_df, "test_filter")
    captured = capfd.readouterr()
    assert "No rows left after applying test_filter" in captured.out

def test_debug_filters_non_empty(capfd, sample_df):
    """Test debug_filters() with a non-empty DataFrame."""
    debug_filters(sample_df, "test_filter")
    captured = capfd.readouterr()
    assert "rows remain after applying test_filter" in captured.out

@pytest.mark.parametrize("input_str,expected", [
    ("CA, San Diego", "San Diego"),
    ("TX, Dallas", "Dallas"),
    ("NY,Albany", "NY,Albany"),      # no space after comma => fails regex
    ("JustCity", "JustCity"),        # doesn't match pattern
    ("", ""),
])
def test_extract_city_name(input_str, expected):
    """Test extracting city name from 'ST, City' format."""
    assert extract_city_name(input_str) == expected

# -------------------------------------------------------------------------
#               TESTS FOR df_enduse_refactored() FUNCTION
# -------------------------------------------------------------------------

def test_df_enduse_refactored_empty():
    """If df_baseline is empty, function should return it and print a warning."""
    df_empty = pd.DataFrame()
    result = df_enduse_refactored(df_empty)
    assert result.empty

def test_df_enduse_refactored_basic(sample_df):
    """
    Test that df_enduse_refactored runs without error and produces expected columns.
    Also verify that filters apply or mask accordingly if default arguments are used.
    """
    enduse_df = df_enduse_refactored(sample_df)
    # Check presence of key output columns
    expected_cols = [
        'square_footage', 'city', 'state', 'base_heating_fuel',
        'base_electricity_heating_consumption', 'baseline_heating_consumption',
        'base_waterHeating_fuel', 'baseline_waterHeating_consumption',
        'base_clothesDrying_fuel', 'baseline_clothesDrying_consumption',
        'base_cooking_fuel', 'baseline_cooking_consumption'
    ]
    for col in expected_cols:
        assert col in enduse_df.columns, f"Missing column {col} in enduse_df"

    # Check that invalid fuels or technologies are masked for heating/waterHeating by default
    # Example: 'Coal' in original for heating => consumption columns should be NaN
    coal_rows = enduse_df[enduse_df['base_heating_fuel'].isna()].index
    assert enduse_df.loc[coal_rows, 'base_electricity_heating_consumption'].isna().all()

    # Because mode='mask' is default, the row should still exist
    # The 'Coal' row from sample_df is not entirely dropped
    assert len(enduse_df) == len(sample_df), "Rows appear to have been dropped in mask mode"

def test_df_enduse_refactored_filter_mode(sample_df):
    """
    Test df_enduse_refactored with 'invalid_row_handling'='filter'
    to ensure rows with invalid fuels/technologies are dropped.
    """
    enduse_df = df_enduse_refactored(sample_df, invalid_row_handling='filter')
    # Now we expect the row with 'Coal' fuel, and row with 'Wood Stove' technology to be removed
    assert len(enduse_df) < len(sample_df), "Filtering did not remove invalid rows"