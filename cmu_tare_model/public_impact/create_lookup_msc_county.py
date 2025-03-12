
import os
import pandas as pd

# import from cmu-tare-model package
from config import PROJECT_ROOT
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2023_2010, cpi_ratio_2023_2020, cpi_ratio_2023_2021
from cmu_tare_model.public_impact.coal_projection_factors import mapping, df_preIRA_coal_projection_factors, df_iraRef_coal_projection_factors

"""
DAMAGES FROM CLIMATE RELATED EMISSIONS (CO2e):
    Use the updated Social Cost of Carbon (190 USD-2020/ton CO2) and inflate to USD-2023
        - EPA Median for 2% near term discount rate and most commonly mentioned value is 190 USD-2020 using the GIVE model.
        - 190 USD-2020 has some inconsistency with the VSL being used. An old study and 2008 VSL is noted
        - 190 USD value and inflate to USD 2023 because there is a clear source and ease of replicability.

    Adjustment for VSL
    - EASIUR uses a VSL of 8.8M USD-2010 
    - New EPA VSL is 11.3M USD-2021
    - INFLATE TO $USD-2023

    ALL DOLLAR VALUES ARE NOW IN USD2023, PREVIOUSLY USED $USD-2021
"""

# For CO2 adjust SCC
# Create an adjustment factor for the new Social Cost of Carbon (SCC)
epa_scc = 190 * cpi_ratio_2023_2020
old_scc = 40 * cpi_ratio_2023_2010
scc_adjustment_factor = epa_scc / old_scc

# For Health-Related Emissions Adjust for different Value of a Statistical Life (VSL) values
# Current VSL is $11.3 M USD2021
# INFLATE TO USD2022, PREVIOUSLY USD2021
current_VSL_USD2022 = 11.3 * cpi_ratio_2023_2021

# Easiur uses a VSL of $8.8 M USD2010
# INFLATE TO USD2022, PREVIOUSLY USD2021
easiur_VSL_USD2022 = 8.8 * (cpi_ratio_2023_2010)

# Calculate VSL adjustment factor
vsl_adjustment_factor = current_VSL_USD2022 / easiur_VSL_USD2022

# ============================================================================================================================================================================
# 


# print("""
# 
# ADJUSTED Electricity CEDM-EASIUR Marginal Damages: Pre-IRA and IRA-Reference
#     - Factor Type: Marginal
#     - Calculation Method: Regression
#     - Metric: Marginal Damages EASIUR [USD per MWh or kWh]
#     - Year: 2018
#     - Regional Aggregation: eGRID subregion (all regions)
#     - Pollutants: SO2, NOx, PM2.5 CO2

# SCC ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------Adjustment: We use the EPA suggested 190 USD-2020 value for the social cost of carbon and inflate to 2022 USD. 
#       **PREVIOUSLY USED 2021 USD**

# VSL: "We use a value of a statistical life (VSL) of USD 8.8 million (in 2010 dollars) for both our AP2 and EASIUR calculations. 
#       EASIUR reports damage intensities in USD/metric ton using this VSL and dollar year."
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# """)

filename = 'Generation-MARREG-DAMEASIUR-egrid-byYear_health2018.csv'
relative_path = os.path.join("cmu_tare_model", "data", "margDamages_EASIUR", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)
df_margDamages_health2018 = pd.read_csv(file_path, index_col=0)

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# Marginal damages [$/kWh]
# Inflate from 2010 to 2022
# Note only 2018 data available, used in place of 2021
df_margDamages_EASIUR_health = pd.DataFrame({
    'subregion_eGRID': df_margDamages_health2018['region'],
    'pollutant': df_margDamages_health2018['pollutant'],
    'unit': '[$USD2023/kWh]',
    'margDamages_dollarPerkWh_adjustVSL_ref': (df_margDamages_health2018['factor'] * (vsl_adjustment_factor) * (1/1000)) * (cpi_ratio_2023_2010),
    'margDamages_dollarPerkWh_adjustVSL_2018': (df_margDamages_health2018['factor'] * (vsl_adjustment_factor) * (1/1000)) * (cpi_ratio_2023_2010)
})

# Apply the mapping to the 'cambium_gea_region' column
df_margDamages_EASIUR_health['cambium_gea_region'] = df_margDamages_EASIUR_health['subregion_eGRID'].map(mapping)
df_margDamages_EASIUR_health

"""
===========================================================================================================================================================
Projection Factors
1. Create dictionaries mapping 'gea_region' to marginal damage factors for each pollutant
2. Map to the projection factors dataframe
3. Calculate the new columns by multiplying coal projection factors with marginal damages (THESE HAVE ALREADY BEEN INTERPOLATED)
4. Drop the intermediate marginal damage columns if they're no longer needed
5. Group the projection factors df by scenario and gea_region
6. Create a nested dictionary to serve as the lookup dictionary for pollutant damage factors ({pollutant}_dollarsPerKWh_adjustVSL)
===========================================================================================================================================================
Pre-IRA Projections
Cambium 2021 Coal Generation
===========================================================================================================================================================
"""

# Create a copy of the dataframe
df_preIRA_health_damages_factors = df_preIRA_coal_projection_factors.copy()

# For SO2
so2_marg_damages = df_margDamages_EASIUR_health[df_margDamages_EASIUR_health['pollutant'] == 'so2'] \
    .set_index('cambium_gea_region')['margDamages_dollarPerkWh_adjustVSL_ref'].to_dict()

# For NOx
nox_marg_damages = df_margDamages_EASIUR_health[df_margDamages_EASIUR_health['pollutant'] == 'nox'] \
    .set_index('cambium_gea_region')['margDamages_dollarPerkWh_adjustVSL_ref'].to_dict()

# For PM2.5
pm25_marg_damages = df_margDamages_EASIUR_health[df_margDamages_EASIUR_health['pollutant'] == 'pm25'] \
    .set_index('cambium_gea_region')['margDamages_dollarPerkWh_adjustVSL_ref'].to_dict()

# Map the marginal damage factors to the projection factors dataframe
df_preIRA_health_damages_factors['so2_marg_damage'] = df_preIRA_health_damages_factors['gea_region'].map(so2_marg_damages)
df_preIRA_health_damages_factors['nox_marg_damage'] = df_preIRA_health_damages_factors['gea_region'].map(nox_marg_damages)
df_preIRA_health_damages_factors['pm25_marg_damage'] = df_preIRA_health_damages_factors['gea_region'].map(pm25_marg_damages)

# Calculate the new columns by multiplying the coal projection factors with the marginal damages
df_preIRA_health_damages_factors['so2_dollarPerkWh_adjustVSL'] = (
    df_preIRA_health_damages_factors['coal_projection_factors'] * df_preIRA_health_damages_factors['so2_marg_damage']
)
df_preIRA_health_damages_factors['nox_dollarPerkWh_adjustVSL'] = (
    df_preIRA_health_damages_factors['coal_projection_factors'] * df_preIRA_health_damages_factors['nox_marg_damage']
)
df_preIRA_health_damages_factors['pm25_dollarPerkWh_adjustVSL'] = (
    df_preIRA_health_damages_factors['coal_projection_factors'] * df_preIRA_health_damages_factors['pm25_marg_damage']
)

# Optionally, drop the intermediate marginal damage columns if they're no longer needed
df_preIRA_health_damages_factors.drop(
    columns=['so2_marg_damage', 'nox_marg_damage', 'pm25_marg_damage'], 
    inplace=True
)
df_preIRA_health_damages_factors
# Initialize the lookup dictionary
lookup_health_damages_electricity_preIRA = {}

# Group the dataframe
grouped = df_preIRA_health_damages_factors.groupby(['scenario', 'gea_region'])

for (scenario, gea_region), group in grouped:
    year_dict = {}
    for _, row in group.iterrows():
        year = int(row['year'])
        data_dict = {
            'so2_dollarPerkWh_adjustVSL': row['so2_dollarPerkWh_adjustVSL'],
            'nox_dollarPerkWh_adjustVSL': row['nox_dollarPerkWh_adjustVSL'],
            'pm25_dollarPerkWh_adjustVSL': row['pm25_dollarPerkWh_adjustVSL']
        }
        year_dict[year] = data_dict
    lookup_health_damages_electricity_preIRA[(scenario, gea_region)] = year_dict

# Now, lookup_dict contains the desired nested dictionary
# damages_preIRA_health_damages_lookup
lookup_health_damages_electricity_preIRA

"""
===========================================================================================================================================================
IRA-Reference Projections
Cambium 2022 Coal Generation
===========================================================================================================================================================
"""
# Create a copy of the dataframe
df_iraRef_health_damages_factors = df_iraRef_coal_projection_factors.copy()

# For SO2
so2_marg_damages = df_margDamages_EASIUR_health[df_margDamages_EASIUR_health['pollutant'] == 'so2'] \
    .set_index('cambium_gea_region')['margDamages_dollarPerkWh_adjustVSL_ref'].to_dict()

# For NOx
nox_marg_damages = df_margDamages_EASIUR_health[df_margDamages_EASIUR_health['pollutant'] == 'nox'] \
    .set_index('cambium_gea_region')['margDamages_dollarPerkWh_adjustVSL_ref'].to_dict()

# For PM2.5
pm25_marg_damages = df_margDamages_EASIUR_health[df_margDamages_EASIUR_health['pollutant'] == 'pm25'] \
    .set_index('cambium_gea_region')['margDamages_dollarPerkWh_adjustVSL_ref'].to_dict()

# Map the marginal damage factors to the projection factors dataframe
df_iraRef_health_damages_factors['so2_marg_damage'] = df_iraRef_health_damages_factors['gea_region'].map(so2_marg_damages)
df_iraRef_health_damages_factors['nox_marg_damage'] = df_iraRef_health_damages_factors['gea_region'].map(nox_marg_damages)
df_iraRef_health_damages_factors['pm25_marg_damage'] = df_iraRef_health_damages_factors['gea_region'].map(pm25_marg_damages)

# Calculate the new columns by multiplying the coal projection factors with the marginal damages
df_iraRef_health_damages_factors['so2_dollarPerkWh_adjustVSL'] = (
    df_iraRef_health_damages_factors['coal_projection_factors'] * df_iraRef_health_damages_factors['so2_marg_damage']
)
df_iraRef_health_damages_factors['nox_dollarPerkWh_adjustVSL'] = (
    df_iraRef_health_damages_factors['coal_projection_factors'] * df_iraRef_health_damages_factors['nox_marg_damage']
)
df_iraRef_health_damages_factors['pm25_dollarPerkWh_adjustVSL'] = (
    df_iraRef_health_damages_factors['coal_projection_factors'] * df_iraRef_health_damages_factors['pm25_marg_damage']
)

# Optionally, drop the intermediate marginal damage columns if they're no longer needed
df_iraRef_health_damages_factors.drop(
    columns=['so2_marg_damage', 'nox_marg_damage', 'pm25_marg_damage'], 
    inplace=True
)
df_iraRef_health_damages_factors
# Initialize the lookup dictionary
lookup_health_damages_electricity_iraRef = {}

# Group the dataframe
grouped = df_iraRef_health_damages_factors.groupby(['scenario', 'gea_region'])

for (scenario, gea_region), group in grouped:
    year_dict = {}
    for _, row in group.iterrows():
        year = int(row['year'])
        data_dict = {
            'so2_dollarPerkWh_adjustVSL': row['so2_dollarPerkWh_adjustVSL'],
            'nox_dollarPerkWh_adjustVSL': row['nox_dollarPerkWh_adjustVSL'],
            'pm25_dollarPerkWh_adjustVSL': row['pm25_dollarPerkWh_adjustVSL']
        }
        year_dict[year] = data_dict
    lookup_health_damages_electricity_iraRef[(scenario, gea_region)] = year_dict

# Now, lookup_dict contains the desired nested dictionary
# damages_iraRef_health_damages_lookup
lookup_health_damages_electricity_iraRef