import pandas as pd
import os

# Get the current working directory of the project
# project_root = os.path.abspath(os.getcwd())
project_root = "C:\\Users\\14128\\Research\\cmu-tare-model"
print(f"Project root directory: {project_root}")

print("""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SCHMITT STUDY VALIDATION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
HEALTH RELATED EMISSIONS: VALIDATION of Grid Emissions Intensity Projections
- Assumes GEA Region and EPA eGRID subregions are the same - which they aren't
- Multiply emissions factors for the grid mix fuel sources (ef_pollutants_egrid) by the generation fraction (grid_mix_reg_full_delta)
- This creates a regional emissions factor. The delta scenario approximates long run marginal emissions rates by subtracting the MidCase generation from the High Electrification scenario generation
- The regional emissions factor (eGRID subregion/Cambium GEA Region) can then be multiplied by the EASIUR marginal social costs (Latitude/Longitude specific)

Adjust for regional cost differences with RSMeans
""")

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

filename = "grid_mix_reg_full_delta.csv"
relative_path = os.path.join(r"projections\schmitt_ev_study", filename)
file_path = os.path.join(project_root, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

df_grid_mix = pd.read_csv(file_path)

df_grid_mix = pd.DataFrame({
    'year': df_grid_mix['Year'],
    'cambium_gea_region': df_grid_mix['Cambium.GEA'],
    'fuel_source': df_grid_mix['Source'],
    'fraction_generation': df_grid_mix['Fraction'],
})
df_grid_mix


# Adjust for regional cost differences with RSMeans
filename = "ef_pollutants_egrid.csv"
relative_path = os.path.join(r"projections\schmitt_ev_study", filename)
file_path = os.path.join(project_root, relative_path)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")
print("\n")

df_grid_emis_factors = pd.read_csv(file_path)

df_grid_emis_factors = pd.DataFrame({
    'cambium_gea_region': df_grid_emis_factors['eGRID_subregion'],
    'fuel_source': df_grid_emis_factors['Fuel'],
    'pollutant': df_grid_emis_factors['Pollutant'],
    'emis_rate': df_grid_emis_factors['Emission_rate'],
    'unit': df_grid_emis_factors['Unit'],
})

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

# Apply the mapping to the 'cambium_gea_region' column
df_grid_emis_factors['cambium_gea_region'] = df_grid_emis_factors['cambium_gea_region'].map(mapping)

# Drop rows where 'cambium_gea_region' is None (regions not included in the mapping)
df_grid_emis_factors = df_grid_emis_factors.dropna(subset=['cambium_gea_region']).reset_index(drop=True)

# Conversion factor from pounds to metric tons
lb_to_mt = 0.00045359237
perMWh_to_perkWh = 1/1000

# Apply the conversion where the unit is 'lb/MWh'
df_grid_emis_factors.loc[df_grid_emis_factors['unit'] == 'lb/MWh', 'emis_rate'] *= (lb_to_mt * perMWh_to_perkWh)
df_grid_emis_factors.loc[df_grid_emis_factors['unit'] == 'lb/MWh', 'unit'] = 'mt/kWh'

df_grid_emis_factors
# HEALTH RELATED EMISSIONS VALIDATION
df_emis_factors_epa_egrid = process_Schmitt_emissions_data(df_grid_mix, df_grid_emis_factors)
print(df_emis_factors_epa_egrid)

# Convert the emissions factors dataframe into a lookup dictionary
lookup_electricity_emissions_egrid = df_emis_factors_epa_egrid.set_index(['year', 'cambium_gea_region']).to_dict('index')

# Display the lookup dictionary
print(lookup_electricity_emissions_egrid)

# # Check unique fuel sources in both dataframes
# fuel_sources_mix = set(df_grid_mix['fuel_source'].unique())
# fuel_sources_emis = set(df_grid_emis_factors['fuel_source'].unique())

# print("Fuel sources in df_grid_mix:", fuel_sources_mix)
# print("Fuel sources in df_grid_emis_factors:", fuel_sources_emis)

# # Merge the dataframes
# df_combined = pd.merge(
#     df_grid_mix,
#     df_grid_emis_factors,
#     on=['cambium_gea_region', 'fuel_source'],
#     how='inner'
# )

# # Calculate emissions contribution
# df_combined['emis_contribution'] = df_combined['fraction_generation'] * df_combined['emis_rate']

# # Sum emissions contributions
# df_emis_factors = df_combined.groupby(
#     ['year', 'cambium_gea_region', 'pollutant']
# )['emis_contribution'].sum().reset_index()
# df_emis_factors