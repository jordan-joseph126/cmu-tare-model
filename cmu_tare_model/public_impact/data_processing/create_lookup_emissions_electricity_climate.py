import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# import from cmu-tare-model package
from config import PROJECT_ROOT

# ====================================================================================================================================================================================
# Set print_verbose to True for detailed output, or False for minimal output
# By default, print_verbose is set to False because define_scenario_params is imported multiple times in the codebase
# and we don't want to print the same information multiple times.
print_verbose = False
# ====================================================================================================================================================================================

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ELECTRICITY - CLIMATE RELATED EMISSIONS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED MARCH 26, 2025 @ 6:45 PM
def calculate_electricity_co2e_cambium(df_cambium_import: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates Cambium electricity emission factors and converts units.

    This function linearly interpolates the Long Run Marginal Emissions Rate (LRMER) and 
    Short Run Marginal Emissions Rate (SRMER) for each scenario and GEA region across the 
    years provided in the input DataFrame. It then converts the units from kg to metric 
    tons (mt) and from MWh to kWh.

    Args:
        df_cambium_import (pd.DataFrame): 
            Input dataframe containing Cambium electricity emission factors with the following columns:
            - 'scenario': Scenario name or identifier.
            - 'gea_region': GEA region identifier.
            - 'year': Year of the data.
            - 'lrmer_co2e_kg_per_MWh': Long Run Marginal Emissions Rate in kg CO2e per MWh.
            - 'srmer_co2e_kg_per_MWh': Short Run Marginal Emissions Rate in kg CO2e per MWh.

    Returns:
        pd.DataFrame: 
            A DataFrame with interpolated LRMER and SRMER values for each year and additional
            columns for emission factors converted to tons per MWh and tons per kWh.

    Raises:
        ValueError: 
            If there are not enough data points to perform interpolation (e.g., fewer than 
            2 distinct years).
        KeyError: 
            If necessary columns (e.g., 'scenario', 'gea_region', 'year', 'lrmer_co2e_kg_per_MWh',
            'srmer_co2e_kg_per_MWh') are missing in the input.

    Additional Notes:
        - The interpolation is performed linearly between the available years for each unique 
          combination of scenario and GEA region.
        - The converted emission factors are added as new columns:
            'lrmer_co2e_mt_per_MWh', 'lrmer_co2e_mt_per_kWh',
            'srmer_co2e_mt_per_MWh', 'srmer_co2e_mt_per_kWh'.
        - The conversion from kg to metric tons is done by dividing by 1,000 (1 ton = 1,000 kg).
        - The conversion from MWh to kWh is done by dividing by 1,000 (1 MWh = 1,000 kWh).
    """
    # Create a copy of the dataframe to avoid mutating the original
    df_cambium_import_copy = df_cambium_import.copy()

    # Create a list to store all interpolated results
    interpolated_data = []

    # Group the data by 'scenario' and 'gea_region' to handle interpolation separately
    grouped = df_cambium_import_copy.groupby(['scenario', 'gea_region'])

    for (scenario, gea_region), group in grouped:
        years = group['year'].values

        # Create linear interpolation functions for LRMER and SRMER
        lrmer_values = group['lrmer_co2e_kg_per_MWh'].values
        lrmer_interp_func = interp1d(years, lrmer_values, kind='linear')  # Interpolate LRMER across years

        srmer_values = group['srmer_co2e_kg_per_MWh'].values
        srmer_interp_func = interp1d(years, srmer_values, kind='linear')  # Interpolate SRMER across years

        # Generate new years in 1-year increments to fill any gaps
        new_years = np.arange(years.min(), years.max() + 1)

        # Compute interpolated values for LRMER and SRMER
        new_lrmer_values = lrmer_interp_func(new_years)
        new_srmer_values = srmer_interp_func(new_years)

        # Store the interpolated results in a temporary DataFrame
        interpolated_group = pd.DataFrame({
            'scenario': scenario,
            'gea_region': gea_region,
            'year': new_years,
            'lrmer_co2e_kg_per_MWh': new_lrmer_values,
            'srmer_co2e_kg_per_MWh': new_srmer_values
        })

        interpolated_data.append(interpolated_group)

    # Concatenate all the interpolated data into a single DataFrame
    df_cambium_import_copy = pd.concat(interpolated_data).reset_index(drop=True)

    # Convert LRMER and SRMER values to tons per MWh and tons per kWh
    df_cambium_import_copy['lrmer_co2e_mt_per_MWh'] = df_cambium_import_copy['lrmer_co2e_kg_per_MWh'] / 1000
    df_cambium_import_copy['lrmer_co2e_mt_per_kWh'] = df_cambium_import_copy['lrmer_co2e_kg_per_MWh'] / 1_000_000

    df_cambium_import_copy['srmer_co2e_mt_per_MWh'] = df_cambium_import_copy['srmer_co2e_kg_per_MWh'] / 1000
    df_cambium_import_copy['srmer_co2e_mt_per_kWh'] = df_cambium_import_copy['srmer_co2e_kg_per_MWh'] / 1_000_000

    return df_cambium_import_copy

# LAST UPDATED MARCH 26, 2025 @ 6:45 PM
def create_cambium_co2e_lookup(df_cambium_processed: pd.DataFrame) -> dict:
    """
    Creates a nested dictionary for quick lookup of Cambium LRMER and SRMER emission factors.

    The outer keys of the dictionary are tuples of the form (scenario, gea_region).
    Each such key maps to another dictionary keyed by year, which then holds the 
    LRMER and SRMER values in metric tons CO2e per kWh.

    Args:
        df_cambium_processed (pd.DataFrame): 
            DataFrame containing processed Cambium emission factors with the following columns:
            - 'scenario': Scenario name or identifier.
            - 'gea_region': GEA region identifier.
            - 'year': Year of the data.
            - 'lrmer_co2e_mt_per_kWh': LRMER in tons CO2e per kWh.
            - 'srmer_co2e_mt_per_kWh': SRMER in tons CO2e per kWh.

    Returns:
        dict: 
            Nested dictionary structured as:
            {
                (scenario, gea_region): {
                    year: {
                        'lrmer_mt_per_kWh_co2e': <float>,
                        'srmer_mt_per_kWh_co2e': <float>
                    },
                    ...
                },
                ...
            }

    Raises:
        KeyError: 
            If expected columns are missing from the dataframe (e.g. 'scenario', 'gea_region', 
            'year', 'lrmer_co2e_mt_per_kWh', 'srmer_co2e_mt_per_kWh').
    """
    # Create a copy to avoid mutating the original DataFrame
    df_cambium_processed_copy = df_cambium_processed.copy()

    # Create the nested lookup dictionary for both LRMER and SRMER in metric tons CO2e/kWh
    emis_scenario_cambium_lookup = {}

    # Populate the dictionary by iterating over rows in the processed DataFrame
    for _, row in df_cambium_processed_copy.iterrows():
        outer_key = (row['scenario'], row['gea_region'])
        year = row['year']

        # Extract LRMER and SRMER in tons per kWh
        lrmer_value = row['lrmer_co2e_mt_per_kWh']
        srmer_value = row['srmer_co2e_mt_per_kWh']

        # Initialize the outer key if it doesn't exist
        if outer_key not in emis_scenario_cambium_lookup:
            emis_scenario_cambium_lookup[outer_key] = {}

        # Assign LRMER and SRMER to the inner dictionary for each year
        emis_scenario_cambium_lookup[outer_key][year] = {
            'lrmer_mt_per_kWh_co2e': lrmer_value,
            'srmer_mt_per_kWh_co2e': srmer_value
        }

    return emis_scenario_cambium_lookup

### Climate-Related Emissions from CAMBIUM LRMER/SRMER 
### Includes pre-combustion (fugitive) and combustion

if print_verbose:
    print("""
    -------------------------------------------------------------------------------------------------------
    PRE-IRA LONG RUN AND SHORT RUN MARGINAL EMISSIONS RATES (LRMER, SRMER) FROM CAMBIUM 2021 RELEASE
    -------------------------------------------------------------------------------------------------------

    """)

# CAMBIUM 2021 FOR PRE-IRA SCENARIO
filename = 'cambium21_midCase_annual_gea.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_cambium21_margEmis_electricity = pd.read_excel(io=file_path, sheet_name='proc_Cambium21_MidCase_gea')

if print_verbose:
    print(f"""
    Retrieved data for filename: {filename}
    Located at filepath: {file_path}

    Loading dataframe ...

    {df_cambium21_margEmis_electricity}

    Creating lookup dictionary for LRMER and SRMER ...
    -------------------------------------------------------------------------------------------------------
    """)

# Calculate electricity emission factors for Cambium 2021
df_cambium21_processed = calculate_electricity_co2e_cambium(df_cambium21_margEmis_electricity)

# Create the lookup dictionary
lookup_emissions_electricity_climate_preIRA = create_cambium_co2e_lookup(df_cambium21_processed)

if print_verbose:
    print("""
    -------------------------------------------------------------------------------------------------------
    IRA LONG RUN AND SHORT RUN MARGINAL EMISSIONS RATES (LRMER, SRMER) FROM CAMBIUM 2022 RELEASE
    -------------------------------------------------------------------------------------------------------
    """)

# CAMBIUM 2022 FOR IRA SCENARIO
filename = 'cambium22_allScenarios_annual_gea.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_cambium22_margEmis_electricity = pd.read_excel(io=file_path, sheet_name='proc_Cambium22_MidCase_gea')

if print_verbose:
    print(f"""
    Retrieved data for filename: {filename}
    Located at filepath: {file_path}

    Loading dataframe ...

    {df_cambium22_margEmis_electricity}

    Creating lookup dictionary for 2024 LRMER and SRMER ...
    -------------------------------------------------------------------------------------------------------
    """)

# Calculate electricity emission factors for Cambium 2022
df_cambium22_processed = calculate_electricity_co2e_cambium(df_cambium22_margEmis_electricity)

# Create the lookup dictionary
lookup_emissions_electricity_climate_IRA = create_cambium_co2e_lookup(df_cambium22_processed)

# Display the dictionaries (commented out, but left unchanged)
# lookup_emissions_electricity_climate_preIRA
# lookup_emissions_electricity_climate_IRA
