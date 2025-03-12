import os
import pandas as pd

# import from cmu-tare-model package
from config import PROJECT_ROOT
print(f"Project root directory: {PROJECT_ROOT}")

print("""
==============================================================================================================================================================================
HEALTH RELATED EMISSIONS: 
      Calculated using the Schmitt et al. (2024) Long Run Marginal Emissions Rates (LRMER) for the MidCase Scenario
      Formerly used to validate CEDM Marginal Emissions/Damages Rates but we decided not to use those.
==============================================================================================================================================================================
- Assumes GEA Region and EPA eGRID subregions are the same. They are similar but not the same.
- Multiply emissions factors (pollutant / kWh) for the grid mix fuel sources (ef_pollutants_egrid: Coal, NG, Renewables) by the generation fraction (grid_mix_reg_full_delta: Fuel Source Generation / Total Generation)
- This creates a regional emissions factor. The delta scenario approximates long run marginal emissions rates by subtracting the MidCase generation from the High Electrification scenario generation
- The regional emissions factor (eGRID subregion/Cambium GEA Region) can then be multiplied by the EASIUR marginal social costs (Latitude/Longitude specific)
""")

# LAST UPDATED NOVEMBER 24 @ 5 PM
# HEALTH RELATED EMISSIONS VALIDATION
def process_Schmitt_emissions_data(df_grid_mix=None, df_grid_emis_factors=None):
    """
    Processes and merges grid mix data with emissions factors, then calculates
    the total emissions contribution by region and pollutant.

    Args:
        df_grid_mix (pandas.DataFrame, optional): DataFrame containing grid mix
            information with columns such as 'year', 'cambium_gea_region',
            'fuel_source', and 'fraction_generation'. Defaults to an empty
            DataFrame if None is provided.
        df_grid_emis_factors (pandas.DataFrame, optional): DataFrame containing
            emissions factor information with columns such as
            'cambium_gea_region', 'fuel_source', 'pollutant', and 'emis_rate'.
            Defaults to an empty DataFrame if None is provided.

    Returns:
        pandas.DataFrame: A pivoted DataFrame indexed by 'year' and
        'cambium_gea_region', with each pollutant's total emissions contribution
        in separate columns (e.g., 'delta_egrid_nh3', 'delta_egrid_nox', etc.).

    Raises:
        KeyError: If expected columns are missing from either DataFrame.
    """
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

    # Identify unique fuel sources for verification
    fuel_sources_mix = set(df_grid_mix['fuel_source'].unique())
    fuel_sources_emis = set(df_grid_emis_factors['fuel_source'].unique())
    print("Fuel sources in df_grid_mix:", fuel_sources_mix)
    print("Fuel sources in df_grid_emis_factors:", fuel_sources_emis)

    # Merge the dataframes on region and fuel source
    df_combined = pd.merge(
        df_grid_mix,
        df_grid_emis_factors,
        on=['cambium_gea_region', 'fuel_source'],
        how='inner'
    )

    # Calculate emissions contribution (fraction_generation * emis_rate)
    df_combined['emis_contribution'] = df_combined['fraction_generation'] * df_combined['emis_rate']

    # Group by year, region, and pollutant to sum total contributions
    df_emis_factors = df_combined.groupby(
        ['year', 'cambium_gea_region', 'pollutant']
    )['emis_contribution'].sum().reset_index()

    # Pivot so that each pollutant's emissions become separate columns
    df_emis_factors_pivot = df_emis_factors.pivot_table(
        index=['year', 'cambium_gea_region'],
        columns='pollutant',
        values='emis_contribution'
    ).reset_index()

    # Rename columns for clarity and consistency
    df_emis_factors_pivot.rename(columns={
        'NH3': 'delta_egrid_nh3',
        'NOx': 'delta_egrid_nox',
        'PM25': 'delta_egrid_pm25',
        'SO2': 'delta_egrid_so2',
        'VOC': 'delta_egrid_voc'
    }, inplace=True)

    return df_emis_factors_pivot


# ======================================================================================================================
# Load the data for grid mix fuel sources and emissions factors
# For example: Region ___ uses ___% Coal, ___% NG, ___% Renewables
# ======================================================================================================================

# Construct the absolute path to the .py file
filename = "grid_mix_reg_full_delta.csv"
relative_path = os.path.join("cmu_tare_model", "data", "projections", "schmitt_ev_study", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

# Load the data for grid mix fuel sources and generation fractions
df_grid_mix = pd.read_csv(file_path)

df_grid_mix = pd.DataFrame({
    'year': df_grid_mix['Year'],
    'cambium_gea_region': df_grid_mix['Cambium.GEA'],
    'fuel_source': df_grid_mix['Source'],
    'fraction_generation': df_grid_mix['Fraction'],
})

print(f"""
======================================================================================================================
Loading the data for electricity grid mix fuel sources and generation fractions ...
For example: Region ___ uses ___% Coal, ___% NG, ___% Renewables
======================================================================================================================

DATAFRAME: df_grid_mix

{df_grid_mix}
""")


# ======================================================================================================================
# Load the data for grid mix fuel sources and emissions factors
# For example: Using Fuel Source X in Region Y results in ___ mt/kWh of Pollutant Z
# ======================================================================================================================

filename = "ef_pollutants_egrid.csv"
relative_path = os.path.join("cmu_tare_model", "data", "projections", "schmitt_ev_study", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

# Load the data for grid mix fuel sources and emissions factors
df_grid_emis_factors = pd.read_csv(file_path)

df_grid_emis_factors = pd.DataFrame({
    'eGRID_subregion': df_grid_emis_factors['eGRID_subregion'],
    'cambium_gea_region': df_grid_emis_factors['eGRID_subregion'],
    'fuel_source': df_grid_emis_factors['Fuel'],
    'pollutant': df_grid_emis_factors['Pollutant'],
    'emis_rate': df_grid_emis_factors['Emission_rate'],
    'unit': df_grid_emis_factors['Unit'],
})

# Map eGRID subregions to Cambium GEA regions (some are excluded)
mapping = {
    'AKGD': None,       # Alaska Grid - Not included
    'AKMS': None,       # Alaska Miscellaneous - Not included
    'AZNM': 'AZNMc',    # Arizona/New Mexico Power Area
    'CAMX': 'CAMXc',    # California Mexico
    'ERCT': 'ERCTc',    # Electric Reliability Council of Texas
    'FRCC': 'FRCCc',    # Florida Reliability Coordinating Council
    'HIMS': None,       # Hawaii Maui Subregion - Not included
    'HIOA': None,       # Hawaii Oahu Subregion - Not included
    'MROE': 'MROEc',    # Midwest Reliability Organization East
    'MROW': 'MROWc',    # Midwest Reliability Organization West
    'NEWE': 'NEWEc',    # New England
    'NWPP': 'NWPPc',    # Northwest Power Pool
    'NYCW': 'NYSTc',    # New York City/Westchester mapped to New York State
    'NYLI': 'NYSTc',    # New York Long Island mapped to New York State
    'NYUP': 'NYSTc',    # New York Upstate mapped to New York State
    'PRMS': None,       # Puerto Rico Miscellaneous - Not included
    'RFCE': 'RFCEc',    # ReliabilityFirst Corporation East
    'RFCM': 'RFCMc',    # ReliabilityFirst Corporation Midwest
    'RFCW': 'RFCWc',    # ReliabilityFirst Corporation West
    'RMPA': 'RMPAc',    # Rocky Mountain Power Area
    'SPNO': 'SPNOc',    # Southwest Power Pool North
    'SPSO': 'SPSOc',    # Southwest Power Pool South
    'SRMV': 'SRMVc',    # SERC Reliability Corporation Mississippi Valley
    'SRMW': 'SRMWc',    # SERC Reliability Corporation Midwest
    'SRSO': 'SRSOc',    # SERC Reliability Corporation South
    'SRTV': 'SRTVc',    # SERC Reliability Corporation Tennessee Valley
    'SRVC': 'SRVCc',    # SERC Reliability Corporation Virginia/Carolina
}

# Apply the mapping to transform eGRID_subregion to Cambium GEA region
df_grid_emis_factors['cambium_gea_region'] = df_grid_emis_factors['cambium_gea_region'].map(mapping)

# Drop rows where 'cambium_gea_region' is None (regions not included in the mapping)
df_grid_emis_factors = df_grid_emis_factors.dropna(subset=['cambium_gea_region']).reset_index(drop=True)

# Conversion constants
lb_to_mt = 0.00045359237      # pounds to metric tons
perMWh_to_perkWh = 1 / 1000   # MWh to kWh

# Convert 'emis_rate' from lb/MWh to mt/kWh when needed
df_grid_emis_factors.loc[df_grid_emis_factors['unit'] == 'lb/MWh', 'emis_rate'] *= (lb_to_mt * perMWh_to_perkWh)
df_grid_emis_factors.loc[df_grid_emis_factors['unit'] == 'lb/MWh', 'unit'] = 'mt/kWh'

# Process the data to calculate emissions factors by region and pollutant
df_emis_factors_epa_egrid = process_Schmitt_emissions_data(df_grid_mix, df_grid_emis_factors)

# Create a lookup dictionary indexed by (year, region) for quick reference
lookup_electricity_emissions_egrid = df_emis_factors_epa_egrid.set_index(
    ['year', 'cambium_gea_region']
).to_dict('index')


print(f"""
======================================================================================================================
Load the data for grid mix fuel sources and emissions factors
For example: Using Fuel Source X in Region Y results in ___ mt/kWh of Pollutant Z
======================================================================================================================
DATAFRAME: df_grid_emis_factors
- Map eGRID subregions to Cambium GEA regions (some are excluded)
- Convert 'emis_rate' from lb/MWh to mt/kWh when needed

{df_grid_emis_factors}

DATAFRAME: df_emis_factors_epa_egrid
- Processed data using the process_Schmitt_emissions_data function
- Calculates emissions factors by region and pollutant

{df_emis_factors_epa_egrid}

LOOKUP DICTIONARY: lookup_electricity_emissions_egrid
- A lookup dictionary is created from df_emis_factors_epa_egrid
- Indexed by (year, region)

{lookup_electricity_emissions_egrid}
""")