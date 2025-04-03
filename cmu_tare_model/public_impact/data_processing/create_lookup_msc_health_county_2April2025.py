# Description: This script processes the RCM data for use in creating a health-related emissions marginal social cost lookup dictionary.
# Marginal Social Costs for Health-Related Emissions
# UPDATE TO USE THE 2023USD INPUT CUSTOM VSL VALUES INSTEAD OF THE 2006USD VALUES INFLATED TO 2023USD!!!
# ======================================================================================================================
import os
import pandas as pd

# import from cmu_tare-model package
from config import PROJECT_ROOT
from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2023_2006

# UPDATE TO USE THE 2023USD INPUT CUSTOM VSL VALUES INSTEAD OF THE 2006USD VALUES INFLATED TO 2023USD!!!
def process_rcm_data(filename: str, PROJECT_ROOT: str) -> pd.DataFrame:
    """
    Processes the RCM data for county-level health-related emissions marginal social costs.

    This function reads a CSV file containing RCM data, retains only annual data, 
    converts county_fips codes to a 5-digit string format, and inflates 2006-dollar cost 
    values to 2023 dollars using a CPI ratio.
# UPDATE TO USE THE 2023USD INPUT CUSTOM VSL VALUES INSTEAD OF THE 2006USD VALUES INFLATED TO 2023USD!!!

    Args:
        filename (str): 
            Name of the CSV file (e.g. "rcm_msc_county_vsl857_usd2006_ground_acs.csv").
        PROJECT_ROOT (str): 
            Path to the root directory of the project.

    Returns:
        pd.DataFrame: 
            A DataFrame containing only annual data, with columns:
            - 'county_fips' (5-digit string)
            - 'damage_usd2023' (inflated cost in USD2023)
            plus all other original columns.

    Raises:
        FileNotFoundError: 
            If the CSV file is not found at the specified path.
        KeyError: 
            If expected columns (like 'season' or 'damage') are missing.
    """
    # Construct the absolute path to the CSV file
    relative_path = os.path.join("cmu_tare_model", "data", "marginal_social_costs", filename)
    file_path = os.path.join(PROJECT_ROOT, relative_path)
    df_rcm_msc_data = pd.read_csv(file_path)

    # Filter to retain only 'annual' season rows
    df_rcm_msc_data = df_rcm_msc_data[df_rcm_msc_data['season'] == 'annual']

    # Original RCM data uses 'fips' as the county identifier, EUSS dataframe uses 'county_fips'
    # Convert 'county_fips' to zero-padded string
    df_rcm_msc_data['county_fips'] = df_rcm_msc_data['fips'].astype(str).str.zfill(5)

    # create a new column 'state' from the 'state_abbr' column
    df_rcm_msc_data['state'] = df_rcm_msc_data['state_abbr']

    # Inflate from USD2006 to USD2023
    df_rcm_msc_data['damage_usd2023'] = df_rcm_msc_data['damage'] * cpi_ratio_2023_2006

    return df_rcm_msc_data


def create_lookup_nested(df_rcm_msc_data: pd.DataFrame) -> dict:
    """
    Creates a nested lookup dictionary keyed by (county_fips, state), then model, then pollutant.

    The structure allows quick retrieval of marginal social cost (in USD2023) for a specific 
    county (county_fips), model, and pollutant.

    Args:
        df_rcm_msc_data (pd.DataFrame):
            DataFrame containing columns:
            - 'county_fips' (zero-padded string)
            - 'state'
            - 'model'
            - 'pollutant'
            - 'damage_usd2023'

    Returns:
        dict: 
            A nested dictionary of the form:
            {
                (county_fips, state): {
                    model: {
                        pollutant: damage_usd2023,
                        ...
                    },
                    ...
                },
                ...
            }
    """
    lookup_health_rcm_msc = {}
    for _, row in df_rcm_msc_data.iterrows():
        county_key = (row['county_fips'], row['state'])
        model = row['model']
        pollutant = row['pollutant']
        damage = row['damage_usd2023']
        
        # Ensure a dict exists for the given county
        if county_key not in lookup_health_rcm_msc:
            lookup_health_rcm_msc[county_key] = {}
        # Ensure a dict exists for the given model
        if model not in lookup_health_rcm_msc[county_key]:
            lookup_health_rcm_msc[county_key][model] = {}
        # Map pollutant to damage
        lookup_health_rcm_msc[county_key][model][pollutant] = damage
        
    return lookup_health_rcm_msc

# ======================================================================================================================
# MARGINAL SOCIAL COSTS FOR HEALTH-RELATED EMISSIONS: FOSSIL FUEL COMBUSTION
# ======================================================================================================================
# Load data for Fossil Fuel MSC (Ground-level stack)
# ======================================================================================================================

df_health_rcm_ground_acs = process_rcm_data("rcm_msc_county_vsl857_usd2006_ground_acs.csv", PROJECT_ROOT)
df_health_rcm_ground_h6c = process_rcm_data("rcm_msc_county_vsl857_usd2006_ground_h6c.csv", PROJECT_ROOT)

print(f"""
======================================================================================================================
HEALTH IMPACTS (MARGINAL SOCIAL COSTS): FOSSIL FUELS
======================================================================================================================
Model: RCM
Output Area: county_fips
Geography: Counties
Pollutant: NOX,SO2,PM25
VSL: 8.57M (Inflated from 11.3M USD2021 to 8.57M USD2006)
Stack Height: ground level

C-R Function: ACS or H6C
======================================================================================================================
- Clean the data so that only annual values are retained. 
- Convert 'county_fips' to string and pad with leading zeros (to ensure a 5-digit format)

DATAFRAME: Ground Level, ACS C-R Function

{df_health_rcm_ground_acs}
      
DATAFRAME: Ground Level, H6C C-R Function

{df_health_rcm_ground_h6c}
      
""")

# Create lookup dictionaries for fossil fuel (ground-level stack)
lookup_health_fossil_fuel_acs = create_lookup_nested(df_health_rcm_ground_acs)
lookup_health_fossil_fuel_h6c = create_lookup_nested(df_health_rcm_ground_h6c)

print(f"""
======================================================================================================================
LOOKUP DICTIONARY: HEALTH IMPACTS (MSC) FROM FOSSIL FUELS
======================================================================================================================

LOOKUP: Ground Level, ACS C-R Function
      
{lookup_health_fossil_fuel_acs}
      
LOOKUP: Ground Level, H6C C-R Function

{lookup_health_fossil_fuel_h6c}

""")

# ======================================================================================================================
# HEALTH IMPACTS (MARGINAL SOCIAL COSTS): ELECTRICITY GENERATION
# ====================================================================================================================
# Load data for Electricity MSC (Elevated/High-stack)
# ====================================================================================================================

df_health_rcm_elevated_acs = process_rcm_data("rcm_msc_county_vsl857_usd2006_elevated_acs.csv", PROJECT_ROOT)
df_health_rcm_elevated_h6c = process_rcm_data("rcm_msc_county_vsl857_usd2006_elevated_h6c.csv", PROJECT_ROOT)

print(f"""
======================================================================================================================
HEALTH IMPACTS (MARGINAL SOCIAL COSTS): ELECTRICITY GENERATION
======================================================================================================================
Model: RCM
Output Area: county_fips
Geography: Counties
Pollutant: NOX,SO2,PM25
VSL: 8.57M (Inflated from 11.3M USD2021 to 8.57M USD2006)
Stack Height: high stack

C-R Function: ACS or H6C
======================================================================================================================
- Clean the data so that only annual values are retained. 
- Convert 'county_fips' to string and pad with leading zeros (to ensure a 5-digit format)

DATAFRAME: Elevated (High Stack), ACS C-R Function

{df_health_rcm_elevated_acs}
      
DATAFRAME: Elevated (High Stack), H6C C-R Function

{df_health_rcm_elevated_h6c}
      
""")

# Create lookup dictionaries for electricity generation (elevated/high-stack)
lookup_health_electricity_acs = create_lookup_nested(df_health_rcm_elevated_acs)
lookup_health_electricity_h6c = create_lookup_nested(df_health_rcm_elevated_h6c)

print(f"""
======================================================================================================================
LOOKUP DICTIONARY: HEALTH IMPACTS (MSC) FROM ELECTRICITY GENERATION
======================================================================================================================

LOOKUP: Elevated (High Stack), ACS C-R Function
      
{lookup_health_electricity_acs}
      
LOOKUP: Elevated (High Stack), H6C C-R Function

{lookup_health_electricity_h6c}

""")
