
from config import PROJECT_ROOT
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ELECTRICITY - HEALTH RELATED EMISSIONS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS
"""

# LAST UPDATED/TESTED NOV 24, 2024 @ 5 PM
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
COAL USED IN ELECTRICITY GENERATION FROM CAMBIUM 2021 RELEASE
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HEALTH-RELATED EMISSIONS: Grid Emissions Intensity Projections

PROJECTION FACTORS FOR FUTURE GRID EMISSIONS INTENSITY (Coal Generation Reduction)
"""

# CAMBIUM 2022 FOR IRA SCENARIO
filename = 'cambium21_midCase_annual_gea.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_cambium21_COAL_processed = pd.read_excel(io=file_path, sheet_name='proc_Cambium21_coal_gea')

print(f"""
Retrieved data for filename: {filename}
Located at filepath: {file_path}
-------------------------------------------------------------------------------------------------------
""")
# df_cambium21_COAL_processed 

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
COAL USED IN ELECTRICITY GENERATION FROM CAMBIUM 2022 RELEASE
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HEALTH-RELATED EMISSIONS: Grid Emissions Intensity Projections

PROJECTION FACTORS FOR FUTURE GRID EMISSIONS INTENSITY (Coal Generation Reduction)
"""

# CAMBIUM 2022 FOR IRA SCENARIO
filename = 'cambium22_allScenarios_annual_gea.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_cambium22_COAL_processed = pd.read_excel(io=file_path, sheet_name='proc_Cambium22_coal_gea')

print(f"""
Retrieved data for filename: {filename}
Located at filepath: {file_path}
-------------------------------------------------------------------------------------------------------
""")
# df_cambium22_COAL_processed 

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

# EPA eGRID Coal Generation Data for 2018 to 2022 ()
filename = 'epa_eGRID_total_coal_generation_2018_2022.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_epa_eGRID_COAL_processed = pd.read_excel(io=file_path, sheet_name='coal_generation_2018_2022')

print(f"""
Retrieved data for filename: {filename}
Located at filepath: {file_path}
-------------------------------------------------------------------------------------------------------
""") 

# Apply the mapping to the 'cambium_gea_region' column
df_epa_eGRID_COAL_processed['cambium_gea_region'] = df_epa_eGRID_COAL_processed['eGRID_subregion'].map(mapping)

# Drop rows where 'cambium_gea_region' is None (regions not included in the mapping)
df_epa_eGRID_COAL_processed = df_epa_eGRID_COAL_processed.dropna(subset=['cambium_gea_region'])

# Group by 'cambium_gea_region' and aggregate
df_epa_eGRID_COAL_processed = (
    df_epa_eGRID_COAL_processed
    .groupby('cambium_gea_region', as_index=False)
    .agg({
        'eGRID_subregion': 'first',  # Retain the first value (or use a different strategy)
        'coal_MWh_2018': 'sum',
        'coal_MWh_2019': 'sum',
        'coal_MWh_2020': 'sum',
        'coal_MWh_2021': 'sum',
        'coal_MWh_2022': 'sum'
    })
)

# Step 1: Melt the DataFrame to Long Format
columns_to_melt = ['coal_MWh_2018', 'coal_MWh_2019', 'coal_MWh_2020', 'coal_MWh_2021', 'coal_MWh_2022']
df_melted = pd.melt(
    df_epa_eGRID_COAL_processed,
    id_vars=['eGRID_subregion', 'cambium_gea_region'],
    value_vars=columns_to_melt,
    var_name='year',
    value_name='coal_MWh'
)

# Step 2: Extract the Year from Column Names
df_melted['year'] = df_melted['year'].str.extract('(\d{4})').astype(int)
# Pre-IRA Scenario
# Step 3: Create Columns to Match the Target DataFrame
df_preIRA_transformed = pd.DataFrame({
    'data_source': 'EPA_eGRID',
    'scenario': 'MidCase',
    'eGRID_subregion': df_melted['eGRID_subregion'],
    'gea_region': df_melted['cambium_gea_region'],
    'year': df_melted['year'],
    'coal_MWh': df_melted['coal_MWh'],
})

# Step 4: Combine the DataFrames
df_preIRA_coal_generation = pd.concat([df_preIRA_transformed, df_cambium21_COAL_processed], ignore_index=True)
# df_preIRA_coal_generation

# Calculate Coal Generation Projection Factors
df_preIRA_coal_projection_factors = calculate_coal_projection_factors(df_preIRA_coal_generation)

print(f"""
=======================================================================================================
Projection Factors for Coal Generation Reduction (Pre-IRA Scenario):
=======================================================================================================
{df_preIRA_coal_projection_factors}

""")

# IRA Reference Scenario
# Step 3: Create Columns to Match the Target DataFrame
df_iraRef_transformed = pd.DataFrame({
    'data_source': 'EPA_eGRID',
    'scenario': 'MidCase',
    'eGRID_subregion': df_melted['eGRID_subregion'],
    'gea_region': df_melted['cambium_gea_region'],
    'year': df_melted['year'],
    'coal_MWh': df_melted['coal_MWh'],
})

# Step 4: Combine the DataFrames
df_iraRef_coal_generation = pd.concat([df_iraRef_transformed, df_cambium22_COAL_processed], ignore_index=True)
# df_iraRef_coal_generation

# Calculate Coal Generation Projection Factors
df_iraRef_coal_projection_factors = calculate_coal_projection_factors(df_iraRef_coal_generation)
print(f"""
=======================================================================================================
Projection Factors for Coal Generation Reduction (IRA-Reference Scenario):
=======================================================================================================
{df_iraRef_coal_projection_factors}

""")