import pandas as pd
import os
# from cmu_tare_model.functions.tare_setup import project_root

# print("""
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS: FOSSIL FUEL EMISSIONS FACTORS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# """)

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

# # Print header
# print(""" 
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CALCULATE EMISSIONS FACTORS: FOSSIL FUELS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Fossil Fuels (Natural Gas, Fuel Oil, Propane):
# - NOx, SO2, CO2: 
#     - RESNET Table 7.1.2 Emissions Factors for Household Combustion Fuels
#     - Source: https://www.resnet.us/wp-content/uploads/ANSIRESNETICC301-2022_resnetpblshd.pdf
#     - All factors are in units of lb/Mbtu; energy consumption in kWh needs conversion.
# - PM2.5: 
#     - A National Methodology and Emission Inventory for Residential Fuel Combustion
#     - Source: https://www3.epa.gov/ttnchie1/conference/ei12/area/haneke.pdf
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# """)

# Calculate emission factors for each fuel type
fuel_oil_factors = calculate_fossil_fuel_emission_factor(
    fuel_type="fuelOil", so2_factor=0.0015, nox_factor=0.1300, pm25_factor=0.83,
    conversion_factor1=1000, conversion_factor2=138500
)
natural_gas_factors = calculate_fossil_fuel_emission_factor(
    fuel_type="naturalGas", so2_factor=0.0006, nox_factor=0.0922, pm25_factor=1.9,
    conversion_factor1=1_000_000, conversion_factor2=1039
)
propane_factors = calculate_fossil_fuel_emission_factor(
    fuel_type="propane", so2_factor=0.0002, nox_factor=0.1421, pm25_factor=0.17,
    conversion_factor1=1000, conversion_factor2=91452
)

# Combine all factors
all_factors = {**fuel_oil_factors, **natural_gas_factors, **propane_factors}

# Create DataFrame
df_marg_emis_factors = pd.DataFrame.from_dict(all_factors, orient="index", columns=["value"])
df_marg_emis_factors.reset_index(inplace=True)
df_marg_emis_factors.columns = ["pollutant", "value"]

# Split pollutant into fuel type and pollutant name
df_marg_emis_factors[["fuel_type", "pollutant"]] = df_marg_emis_factors["pollutant"].str.split("_", expand=True)

# Update the units to metric tons per kWh
df_marg_emis_factors["unit"] = "[mt/kWh]"

# Convert units from lb/kWh to metric tons/kWh where applicable
lb_to_mt = 0.00045359237
pollutants_in_lb = ['so2', 'nox', 'pm25']
df_marg_emis_factors['value'] = df_marg_emis_factors.apply(
    lambda row: row['value'] * lb_to_mt if row['pollutant'] in pollutants_in_lb else row['value'], axis=1
)

# Add 'state' column with default value
df_marg_emis_factors = df_marg_emis_factors.assign(state="National")

# Reorder columns for clarity
df_marg_emis_factors = df_marg_emis_factors[["state", "fuel_type", "pollutant", "value", "unit"]]

# Create lookup dictionary for fossil fuel emissions factors
lookup_emis_fossil_fuel = {}
fuel_types = df_marg_emis_factors["fuel_type"].unique()

for fuel in fuel_types:
    lookup_emis_fossil_fuel[fuel] = {
        pollutant: df_marg_emis_factors[
            (df_marg_emis_factors["fuel_type"] == fuel) & 
            (df_marg_emis_factors["pollutant"] == pollutant)
        ]["value"].values[0]
        for pollutant in ["co2e", "so2", "nox", "pm25"]
    }

# print(f"""
# --------------------------------------------------------------------------------------------------------------------------------------
# Fossil Fuels: Climate and Health-Related Pollutants
# --------------------------------------------------------------------------------------------------------------------------------------
# DATAFRAME: Marginal Emission Factors for Fossil Fuels
      
# {df_marg_emis_factors}  

# LOOKUP DICTIONARY: Fossil Fuel Emissions

# {lookup_emis_fossil_fuel}

# """)