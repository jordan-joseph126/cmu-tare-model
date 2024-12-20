import pandas as pd
import os
from cmu_tare_model.functions.load_and_filter_euss_data import project_root

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FOSSIL FUEL EMISSIONS FACTORS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# LAST UPDATED DECEMBER 4, 2024
def calculate_fossil_fuel_emission_factor(fuel_type, so2_factor, nox_factor, pm25_factor, conversion_factor1, conversion_factor2):
    """
    Calculate Emission Factors for Fossil Fuels.

    Parameters:
    -----------
    fuel_type : str
        Type of fuel (e.g., "naturalGas", "fuelOil", "propane").
    so2_factor : float
        SO2 emission factor in lb/Mbtu.
    nox_factor : float
        NOx emission factor in lb/Mbtu.
    pm25_factor : float
        PM2.5 emission factor in lb per volume unit (varies by fuel).
    conversion_factor1 : int
        Conversion factor for volume units to gallons/thousand gallons.
    conversion_factor2 : int
        Conversion factor for energy content (e.g., BTU per gallon/cf).
    
    Returns:
    --------
    dict
        Dictionary containing emission factors for the given fuel type in lb/kWh or mt/kWh.
    """

    # Correct conversion factor from Mbtu to kWh
    # 1 Mbtu = 1,000,000 Btu
    # 1 kWh = 3,412 Btu
    # So, 1 Mbtu = 1,000,000 / 3,412 kWh
    mbtu_to_kwh = 1_000_000 / 3412  # Approximately 293.07107 kWh/Mbtu

    # Emission factors in lb/kWh
    emission_factors = {
        f"{fuel_type}_so2": so2_factor * (1/mbtu_to_kwh),
        f"{fuel_type}_nox": nox_factor * (1 / mbtu_to_kwh),
        f"{fuel_type}_pm25": pm25_factor * (1 / conversion_factor1) * (1 / conversion_factor2) * 3412,
    }

    # # Natural gas-specific CO2e calculation (including leakage)
    # leakage rate for natural gas infrastructure
    # 1 Therm = 29.30 kWh --> 1.27 kg CO2e/therm * (1 therm/29.30 kWh) = 0.043 kg CO2e/kWh = 0.095 lb CO2e/kWh
    naturalGas_leakage_mtCO2e_perkWh = 0.043 * (1 / 1000)

    if fuel_type == "naturalGas":
        # Convert units from kg/MWh to ton/MWh to ton/kWh
        emission_factors[f"{fuel_type}_co2e"] = (228.5 * (1 / 1000) * (1 / 1000)) + naturalGas_leakage_mtCO2e_perkWh

    # CO2e for propane and fuel oil
    # Convert units from kg/MWh to ton/MWh to ton/kWh
    elif fuel_type == "propane":
        emission_factors[f"{fuel_type}_co2e"] = 275.8 * (1 / 1000) * (1 / 1000)
    elif fuel_type == "fuelOil":
        emission_factors[f"{fuel_type}_co2e"] = 303.9 * (1 / 1000) * (1 / 1000)

    return emission_factors

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ELECTRICITY - CLIMATE RELATED EMISSIONS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED/TESTED NOV 24, 2024 @ 5 PM
from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def calculate_electricity_co2e_cambium(df_cambium_import):
    """
    Interpolates Cambium electricity emission factors and converts units.

    This function takes a dataframe containing Cambium electricity emission factors and performs the following:
    - Interpolates the Long Run Marginal Emissions Rates (LRMER) and Short Run Marginal Emissions Rates (SRMER)
      values for each scenario and GEA region on an annual basis.
    - Converts the LRMER and SRMER values from kg per MWh to tons per MWh and tons per kWh.

    Parameters
    ----------
    df_cambium_import : pandas.DataFrame
        DataFrame containing Cambium electricity emission factors with the following columns:
        - 'scenario': Scenario name or identifier.
        - 'gea_region': GEA region identifier.
        - 'year': Year of the data.
        - 'lrmer_co2e_kg_per_MWh': Long Run Marginal Emissions Rate in kg CO2e per MWh.
        - 'srmer_co2e_kg_per_MWh': Short Run Marginal Emissions Rate in kg CO2e per MWh.

    Returns
    -------
    df_cambium_import_copy : pandas.DataFrame
        DataFrame with interpolated LRMER and SRMER values for each year and additional columns for emission factors
        converted to tons per MWh and tons per kWh.

    Notes
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

    # Group by 'scenario', 'state', and 'gea_region'
    grouped = df_cambium_import_copy.groupby(['scenario', 'gea_region'])

    for (scenario, gea_region), group in grouped:
        years = group['year'].values

        # Interpolate for LRMER (Long Run Marginal Emissions Rates)
        lrmer_values = group['lrmer_co2e_kg_per_MWh'].values
        lrmer_interp_func = interp1d(years, lrmer_values, kind='linear')

        # Interpolate for SRMER (Short Run Marginal Emissions Rates)
        srmer_values = group['srmer_co2e_kg_per_MWh'].values
        srmer_interp_func = interp1d(years, srmer_values, kind='linear')

        # Generate new years in 1-year increments
        new_years = np.arange(years.min(), years.max() + 1)

        # Interpolate the LRMER and SRMER values for these new years
        new_lrmer_values = lrmer_interp_func(new_years)
        new_srmer_values = srmer_interp_func(new_years)

        # Store the results in a DataFrame
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
    Creates a nested lookup dictionary for Cambium emission factors.

    This function takes a processed dataframe containing Cambium emission factors and constructs a nested dictionary
    that allows quick lookup of LRMER and SRMER emission factors based on scenario, GEA region, and year.

    Parameters
    ----------
    df_cambium_processed : pandas.DataFrame
        DataFrame containing processed Cambium emission factors with the following columns:
        - 'scenario': Scenario name or identifier.
        - 'gea_region': GEA region identifier.
        - 'year': Year of the data.
        - 'lrmer_co2e_ton_per_kWh': Long Run Marginal Emissions Rate in tons CO2e per kWh.
        - 'srmer_co2e_ton_per_kWh': Short Run Marginal Emissions Rate in tons CO2e per kWh.

    Returns
    -------
    emis_scenario_cambium_lookup : dict
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

    # Populate the dictionary
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


"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ELECTRICITY - HEALTH RELATED EMISSIONS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED/TESTED NOV 24, 2024 @ 5 PM
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def calculate_coal_projection_factors(df_cambium):
    """
    Interpolates coal_MWh and calculates coal projection factors for each region from 2018 to 2050,
    using 2018 coal generation as the reference point.

    Parameters
    ----------
    df_cambium : pandas.DataFrame
        DataFrame containing Cambium coal generation data with the following columns:
        - 'scenario': Scenario name or identifier.
        - 'gea_region': GEA region identifier.
        - 'year': Year of the data.
        - 'coal_MWh': Coal generation in MWh.

    Returns
    -------
    df_coal_factors : pandas.DataFrame
        DataFrame with interpolated coal_MWh values and a new column 'coal_projection_factors'.
    """
    # Create a copy of the dataframe
    df_cambium_copy = df_cambium.copy()

    # Create a new DataFrame to store interpolated results
    interpolated_data = []

    # Group by 'scenario' and 'gea_region'
    grouped = df_cambium_copy.groupby(['scenario', 'gea_region'])

    for (scenario, gea_region), group in grouped:
        # Extract existing years and coal_MWh values
        years = group['year'].values
        coal_MWh_values = group['coal_MWh'].values

        # Create interpolation function, allowing extrapolation
        coal_MWh_interp_func = interp1d(years, coal_MWh_values, kind='linear', bounds_error=False, fill_value="extrapolate")

        # Generate years from 2018 to 2050
        all_years = np.arange(2018, 2051)

        # Interpolate the coal_MWh values for these years
        interpolated_values = coal_MWh_interp_func(all_years)

        # Store the results in a DataFrame
        interpolated_group = pd.DataFrame({
            'scenario': scenario,
            'gea_region': gea_region,
            'year': all_years,
            'coal_MWh': interpolated_values
        })

        interpolated_data.append(interpolated_group)

    # Concatenate all the interpolated data into a single DataFrame
    df_interpolated = pd.concat(interpolated_data).reset_index(drop=True)

    # Get the coal_MWh value in 2018 for each scenario and gea_region
    coal_MWh_2018 = df_interpolated[df_interpolated['year'] == 2018][['scenario', 'gea_region', 'coal_MWh']]
    coal_MWh_2018 = coal_MWh_2018.set_index(['scenario', 'gea_region'])['coal_MWh']

    # Map the 2018 coal_MWh values to the DataFrame
    df_interpolated['coal_MWh_2018'] = df_interpolated.set_index(['scenario', 'gea_region']).index.map(coal_MWh_2018)

    # Avoid division by zero by replacing zero coal_MWh_2018 with NaN
    df_interpolated['coal_MWh_2018'] = df_interpolated['coal_MWh_2018'].replace(0, np.nan)

    # Conditions for regions other than CAMX
    condition_regions = (df_interpolated['gea_region'] != 'CAMX')

    # Calculate coal projection factors for regions other than CAMX
    df_interpolated.loc[condition_regions, 'coal_projection_factors'] = (
        df_interpolated.loc[condition_regions, 'coal_MWh'] / df_interpolated.loc[condition_regions, 'coal_MWh_2018']
    )

    # For CAMX region, assign coal_projection_factors as 1
    condition_CAMX = (df_interpolated['gea_region'] == 'CAMX')
    df_interpolated.loc[condition_CAMX, 'coal_projection_factors'] = 1

    # Replace any NaN or infinite values resulting from division by zero with 0
    df_interpolated['coal_projection_factors'] = df_interpolated['coal_projection_factors'].replace([np.inf, -np.inf, np.nan], 0)

    # Drop temporary columns
    df_interpolated.drop(columns=['coal_MWh_2018'], inplace=True)

    return df_interpolated

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SCHMITT STUDY VALIDATION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED NOVEMBER 24 @ 5 PM
# HEALTH RELATED EMISSIONS VALIDATION
def process_Schmitt_emissions_data(df_grid_mix=None, df_grid_emis_factors=None):
    if df_grid_mix is None:
        df_grid_mix = pd.DataFrame({
            'cambium_gea_region': [],
            'fuel_source': [],
            'fraction_generation': []
        })
    if df_grid_emis_factors is None:
        df_grid_emis_factors = pd.DataFrame({
            'cambium_gea_region': [],
            'fuel_source': [],
            'emis_rate': []
        })
    # Check unique fuel sources in both dataframes
    fuel_sources_mix = set(df_grid_mix['fuel_source'].unique())
    fuel_sources_emis = set(df_grid_emis_factors['fuel_source'].unique())

    print("Fuel sources in df_grid_mix:", fuel_sources_mix)
    print("Fuel sources in df_grid_emis_factors:", fuel_sources_emis)

    # Merge the dataframes
    df_combined = pd.merge(
        df_grid_mix,
        df_grid_emis_factors,
        on=['cambium_gea_region', 'fuel_source'],
        how='inner'
    )

    # Calculate emissions contribution
    df_combined['emis_contribution'] = df_combined['fraction_generation'] * df_combined['emis_rate']

    # Sum emissions contributions
    df_emis_factors = df_combined.groupby(
        ['year', 'cambium_gea_region', 'pollutant']
    )['emis_contribution'].sum().reset_index()

    # Pivot the dataframe
    df_emis_factors_pivot = df_emis_factors.pivot_table(
        index=['year', 'cambium_gea_region'],
        columns='pollutant',
        values='emis_contribution'
    ).reset_index()

    # Rename columns
    df_emis_factors_pivot.rename(columns={
        'NH3': 'delta_egrid_nh3',
        'NOx': 'delta_egrid_nox',
        'PM25': 'delta_egrid_pm25',
        'SO2': 'delta_egrid_so2',
        'VOC': 'delta_egrid_voc'
    }, inplace=True)

    return df_emis_factors_pivot