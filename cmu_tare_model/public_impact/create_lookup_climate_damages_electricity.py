import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

# import from cmu-tare-model package
from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ELECTRICITY - CLIMATE RELATED EMISSIONS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED/TESTED NOV 24, 2024 @ 5 PM
def calculate_electricity_co2e_cambium(df_cambium_import):
    """
    Interpolates Cambium electricity emission factors and converts units.

    Args:
        df_cambium_import (pandas.DataFrame): Input dataframe containing Cambium electricity emission factors with the following columns:
            - 'scenario': Scenario name or identifier.
            - 'gea_region': GEA region identifier.
            - 'year': Year of the data.
            - 'lrmer_co2e_kg_per_MWh': Long Run Marginal Emissions Rate in kg CO2e per MWh.
            - 'srmer_co2e_kg_per_MWh': Short Run Marginal Emissions Rate in kg CO2e per MWh.

    Returns:
        df_cambium_import_copy (pandas.DataFrame): DataFrame with interpolated LRMER and SRMER values for each year and additional columns for emission factors
        converted to tons per MWh and tons per kWh.


    Raises:
        ValueError: If there are not enough data points to perform interpolation (e.g., fewer than 2 distinct years).
        KeyError: If necessary columns (e.g., 'scenario', 'gea_region', 'year', 'lrmer_co2e_kg_per_MWh',
                  'srmer_co2e_kg_per_MWh') are missing in the input.

    Additional Notes:
    -----
    - The interpolation is performed linearly between the available years for each unique combination of scenario and GEA region.
    - The converted emission factors are added as new columns:
        - 'lrmer_co2e_ton_per_MWh'
        - 'lrmer_co2e_ton_per_kWh'
        - 'srmer_co2e_ton_per_MWh'
        - 'srmer_co2e_ton_per_kWh'
    - The conversion from kg to tons is done by dividing by 1,000 (1 ton = 1,000 kg).
    - The conversion from MWh to kWh is done by dividing by 1,000 (1 MWh = 1,000 kWh).
    """

    # Create a copy of the dataframe
    df_cambium_import_copy = df_cambium_import.copy()

    # Create a new DataFrame to store interpolated results
    interpolated_data = []

    # Group the data by 'scenario' and 'gea_region' to handle interpolation for each group separately
    grouped = df_cambium_import_copy.groupby(['scenario', 'gea_region'])

    for (scenario, gea_region), group in grouped:
        years = group['year'].values

        # Initialize the linear interpolation function for LRMER
        lrmer_values = group['lrmer_co2e_kg_per_MWh'].values
        lrmer_interp_func = interp1d(years, lrmer_values, kind='linear')

        # Initialize the linear interpolation function for SRMER
        srmer_values = group['srmer_co2e_kg_per_MWh'].values
        srmer_interp_func = interp1d(years, srmer_values, kind='linear')

        # Generate new years in 1-year increments, ensuring no gaps in time coverage
        new_years = np.arange(years.min(), years.max() + 1)

        # Interpolate the LRMER and SRMER values for these new years
        new_lrmer_values = lrmer_interp_func(new_years)
        new_srmer_values = srmer_interp_func(new_years)

        # Store the results in a temporary DataFrame to be concatenated
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

    # Convert both LRMER and SRMER values to tons per MWh and tons per kWh
    df_cambium_import_copy['lrmer_co2e_ton_per_MWh'] = df_cambium_import_copy['lrmer_co2e_kg_per_MWh'] / 1000
    df_cambium_import_copy['lrmer_co2e_ton_per_kWh'] = df_cambium_import_copy['lrmer_co2e_kg_per_MWh'] / 1_000_000

    df_cambium_import_copy['srmer_co2e_ton_per_MWh'] = df_cambium_import_copy['srmer_co2e_kg_per_MWh'] / 1000
    df_cambium_import_copy['srmer_co2e_ton_per_kWh'] = df_cambium_import_copy['srmer_co2e_kg_per_MWh'] / 1_000_000

    return df_cambium_import_copy


def create_cambium_co2e_lookup(df_cambium_processed):
    """
    Creates a nested dictionary that allows quick lookup of Cambium LRMER and SRMER emission factors based on scenario, GEA region, and year.

    Args:
        df_cambium_processed (pandas.DataFrame): DataFrame containing processed Cambium emission factors with the following columns:
            - 'scenario': Scenario name or identifier.
            - 'gea_region': GEA region identifier.
            - 'year': Year of the data.
            - 'lrmer_co2e_ton_per_kWh': Long Run Marginal Emissions Rate in tons CO2e per kWh.
            - 'srmer_co2e_ton_per_kWh': Short Run Marginal Emissions Rate in tons CO2e per kWh.

    Returns:
        emis_scenario_cambium_lookup (dict): A nested dictionary for quick emission factor lookups by scenario, region, and year.
            Nested dictionary structured as:
            {
                (scenario, gea_region): {
                    year: {
                        'lrmer_co2e': lrmer_value,
                        'srmer_co2e': srmer_value
                    },
                    ...
                },
                ...
            }

    Raises:
        KeyError: If expected columns (e.g., 'scenario', 'gea_region', 'year', 'lrmer_co2e_ton_per_kWh',
                  'srmer_co2e_ton_per_kWh') are missing from the dataframe.

    Notes
    -----
    - The outer keys of the dictionary are tuples containing (scenario, gea_region).
    - The inner dictionary maps years to a dictionary containing both LRMER and SRMER values.
    - This structure allows efficient retrieval of emission factors based on scenario, location, and year.
    """

    # Create a copy of the dataframe
    df_cambium_processed_copy = df_cambium_processed.copy()

    # Create the nested lookup dictionary for both LRMER and SRMER in tons CO2e per kWh
    emis_scenario_cambium_lookup = {}

    # Populate the dictionary by iterating over rows in the processed DataFrame
    for _, row in df_cambium_processed_copy.iterrows():
        outer_key = (row['scenario'], row['gea_region'])
        year = row['year']

        # Extract both LRMER and SRMER values in tons per kWh
        lrmer_value = row['lrmer_co2e_ton_per_kWh']
        srmer_value = row['srmer_co2e_ton_per_kWh']

        # Initialize the outer key if not already present
        if outer_key not in emis_scenario_cambium_lookup:
            emis_scenario_cambium_lookup[outer_key] = {}

        # Assign both LRMER and SRMER values in the inner dictionary for each year
        emis_scenario_cambium_lookup[outer_key][year] = {
            'lrmer_ton_per_kWh_co2e': lrmer_value,
            'srmer_ton_per_kWh_co2e': srmer_value
        }

    return emis_scenario_cambium_lookup

### Climate-Related Emissions from CAMBIUM LRMER/SRMER 
### Includes pre-combustion (fugitive) and combustion

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

print(f"""
Retrieved data for filename: {filename}
Located at filepath: {file_path}

Loading dataframe ...
Creating lookup dictionary for LRMER and SRMER ...
-------------------------------------------------------------------------------------------------------
""")

# Calculate electricity emission factors for Cambium 2021
# Process the data using the provided function to interpolate and convert units
df_cambium21_processed = calculate_electricity_co2e_cambium(df_cambium21_margEmis_electricity)

# # Display the processed DataFrame
# df_cambium21_processed

# Create the lookup dictionary using the create_cambium_co2e_lookup function
lookup_co2e_emis_electricity_preIRA = create_cambium_co2e_lookup(df_cambium21_processed)

# Display the lookup dictionary
lookup_co2e_emis_electricity_preIRA

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

print(f"""
Retrieved data for filename: {filename}
Located at filepath: {file_path}

Loading dataframe ...
Creating lookup dictionary for 2024 LRMER and SRMER ...
-------------------------------------------------------------------------------------------------------
""")

# Calculate electricity emission factors for Cambium 2021
# Process the data using the provided function to interpolate and convert units
df_cambium22_processed = calculate_electricity_co2e_cambium(df_cambium22_margEmis_electricity)

# # Display the processed DataFrame
# df_cambium22_processed

# Create the lookup dictionary using the create_cambium_co2e_lookup function
lookup_co2e_emis_electricity_IRA = create_cambium_co2e_lookup(df_cambium22_processed)

# Display the lookup dictionary
lookup_co2e_emis_electricity_IRA