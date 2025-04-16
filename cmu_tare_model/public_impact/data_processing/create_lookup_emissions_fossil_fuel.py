import pandas as pd
from cmu_tare_model.constants import POLLUTANTS

# =======================================================================================================================
# Set print_verbose to True for detailed output, or False for minimal output
# By default, print_verbose is set to False because define_scenario_params is imported multiple times in the codebase
# and we don't want to print the same information multiple times.
print_verbose = False
# =======================================================================================================================

if print_verbose:
    print("""
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    FUNCTIONS: FOSSIL FUEL EMISSIONS FACTORS
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """)

# LAST UPDATED MARCH 26, 2025 @ 6:45 PM
def calculate_fossil_fuel_emission_factor(
    fuel_type: str,
    so2_factor: float,
    nox_factor: float,
    pm25_factor: float,
    conversion_factor1: int,
    conversion_factor2: int
) -> dict:
    """
    Calculates emission factors for a specified fossil fuel.

    This function computes the SO2, NOx, PM2.5, and CO2e emissions for the given 
    fossil fuel (natural gas, fuel oil, or propane) per kWh of energy, converting 
    from various initial units.

    Args:
        fuel_type (str): 
            Type of fuel (e.g., "naturalGas", "fuelOil", "propane").
        so2_factor (float): 
            SO2 emission factor in lb/Mbtu.
        nox_factor (float): 
            NOx emission factor in lb/Mbtu.
        pm25_factor (float): 
            PM2.5 emission factor in lb per volume unit (varies by fuel).
        conversion_factor1 (int): 
            Conversion factor for volume units to gallons/thousand gallons.
        conversion_factor2 (int): 
            Conversion factor for energy content (e.g., BTU per gallon/cf).

    Returns:
        dict: 
            A dictionary containing emission factors for the given fuel type. 
            The keys follow the pattern "<fuel_type>_so2", "<fuel_type>_nox", 
            "<fuel_type>_pm25", and optionally "<fuel_type>_co2e".

    References/Notes:
        Fossil Fuels (Natural Gas, Fuel Oil, Propane):
        - NOx, SO2, CO2: 
            - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
            - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
            - All factors are in units of lb/Mbtu; energy consumption in kWh needs conversion.
        - PM2.5: 
            - A National Methodology and Emission Inventory for Residential Fuel Combustion
            - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf
            - PM2.5 emission factors are given in lb per volume unit (varies by fuel).
                PM2.5 - FUEL OIL: 0.83 lb/thousand gallons * (1 thousand gallons / 1000 gallons) * (1 gallon heating oil/138,500 BTU) * (3412 BTU/1 kWh)
                PM2.5 - NATURAL GAS: 1.9 lb/million cf * (million cf/1000000 cf) * (1 cf natural gas/1039 BTU) * (3412 BTU/1 kWh)
                PM2.5 - PROPANE: 0.17 lb/thousand gallons * (1 thousand gallons / 1000 gallons) * (1 gallon propane/91,452 BTU) * (3412 BTU/1 kWh)

        - ResStock Carbon Emissions Associated With On-Site Fossil Fuel Combustion (No need for leakage factor):
            For calculating carbon emissions related to the on-site consumption of natural gas, propane, and
            fuel oil, this dataset uses emission factor values from Table 7.1.2(1) of draft PDS-01 of
            BSR/RESNET/ICCC 301 Addendum B, CO2 Index. 
            
            These values include both combustion and pre-combustion (e.g., methane leakage from natural gas) CO2e emissions:
            • 147.3 lb/MMBtu (228.5 kg/MWh) for natural gas
            • 177.8 lb/MMBtu (275.8 kg/MWh) for propane
            • 195.9 lb/MMBtu (303.9 kg/MWh) for fuel oil.

            Source: https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2024/resstock_tmy3_release_2/resstock_documentation_2024_release_2.pdf
            
    Raises:
        None: This function does not explicitly raise any exceptions.
    """
    # Correct conversion from Mbtu to kWh
    # 1 Mbtu = 1,000,000 Btu
    # 1 kWh = 3,412 Btu
    # So, 1 Mbtu = 1,000,000 / 3,412 kWh
    mbtu_to_kwh = 1_000_000 / 3412  # ~293.07 kWh per Mbtu

    # Emission factors in lb/kWh
    emission_factors = {
        f"{fuel_type}_so2": so2_factor * (1 / mbtu_to_kwh),
        f"{fuel_type}_nox": nox_factor * (1 / mbtu_to_kwh),
        # Convert from lb per volume to lb/kWh using the provided conversion factors
        f"{fuel_type}_pm25": pm25_factor * (1 / conversion_factor1) * (1 / conversion_factor2) * 3412,
    }

    # All 
    if fuel_type == "naturalGas":
        # Convert from kg/MWh to metric tons/kWh, then add the leakage
        emission_factors[f"{fuel_type}_co2e"] = (228.5 * (1 / 1000) * (1 / 1000))
    elif fuel_type == "propane":
        emission_factors[f"{fuel_type}_co2e"] = 275.8 * (1 / 1000) * (1 / 1000)
    elif fuel_type == "fuelOil":
        emission_factors[f"{fuel_type}_co2e"] = 303.9 * (1 / 1000) * (1 / 1000)

    return emission_factors

# Calculate emission factors for each fuel type
# Fuel Oil
fuel_oil_factors = calculate_fossil_fuel_emission_factor(
    fuel_type="fuelOil",
    so2_factor=0.0015,
    nox_factor=0.1300,
    pm25_factor=0.83,
    conversion_factor1=1000,
    conversion_factor2=138_500
)
# Natural Gas
natural_gas_factors = calculate_fossil_fuel_emission_factor(
    fuel_type="naturalGas",
    so2_factor=0.0006,
    nox_factor=0.0922,
    pm25_factor=1.9,
    conversion_factor1=1_000_000,
    conversion_factor2=1039
)
# Propane
propane_factors = calculate_fossil_fuel_emission_factor(
    fuel_type="propane",
    so2_factor=0.0002,
    nox_factor=0.1421,
    pm25_factor=0.17,
    conversion_factor1=1000,
    conversion_factor2=91_452
)

# Combine all factors
all_factors = {**fuel_oil_factors, **natural_gas_factors, **propane_factors}

# Create DataFrame from the combined factors
df_marg_emis_factors = pd.DataFrame.from_dict(all_factors, orient="index", columns=["value"])
df_marg_emis_factors.reset_index(inplace=True)
df_marg_emis_factors.columns = ["pollutant", "value"]

# Split pollutant column into fuel type and pollutant name
df_marg_emis_factors[["fuel_type", "pollutant"]] = df_marg_emis_factors["pollutant"].str.split("_", expand=True)

# Update the units to metric tons per kWh
df_marg_emis_factors["unit"] = "[mt/kWh]"

# Convert from lb/kWh to metric tons/kWh where applicable
lb_to_mt = 0.00045359237
pollutants_in_lb = ['so2', 'nox', 'pm25']
df_marg_emis_factors['value'] = df_marg_emis_factors.apply(
    lambda row: row['value'] * lb_to_mt if row['pollutant'] in pollutants_in_lb else row['value'], axis=1
)

# Add 'state' column with default value
df_marg_emis_factors = df_marg_emis_factors.assign(state="National")

# Reorder columns for clarity
df_marg_emis_factors = df_marg_emis_factors[["state", "fuel_type", "pollutant", "value", "unit"]]

# Create lookup dictionary
lookup_emissions_fossil_fuel = {}
fuel_types = df_marg_emis_factors["fuel_type"].unique()

for fuel in fuel_types:
    lookup_emissions_fossil_fuel[fuel] = {
        pollutant: df_marg_emis_factors[
            (df_marg_emis_factors["fuel_type"] == fuel) & 
            (df_marg_emis_factors["pollutant"] == pollutant)
        ]["value"].values[0]
        for pollutant in POLLUTANTS
    }

if print_verbose:
    # print the DataFrame and dictionary (commented out, left as-is)
    print(df_marg_emis_factors)
    print(lookup_emissions_fossil_fuel)