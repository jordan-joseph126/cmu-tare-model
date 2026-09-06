import os
import pandas as pd
import numpy as np
import re
from typing import Any, Optional

from config import PROJECT_ROOT
from cmu_tare_model.constants import EQUIPMENT_SPECS, VALID_CATEGORIES, VERBOSE

from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask
from cmu_tare_model.utils.calculation_utils import (
    get_all_possible_fuel_columns,
    identify_valid_homes
    )

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
LOAD EUSS/RESSTOCK DATA AND APPLY FILTERS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# ---------------------------------------------------------------------------
# County --> GEA region crosswalk (Cambium 2023+ geography)
# ---------------------------------------------------------------------------
# NREL redefined GEA regions in Cambium 2023+, so a home's GEA can no longer be
# taken from the ResStock "in.generation_and_emissions_assessment_region" column
# (those are the retired "*c" codes). This crosswalk maps each county to its new
# Cambium GEA, and process_euss_data() uses it to set gea_region.
#
# Two read guards, both of which fail silently if skipped:
#   - The file carries a UTF-8 BOM, so it is read with encoding="utf-8-sig";
#     otherwise the first column name is corrupted and lookups raise KeyError.
#   - Leading zeros are already dropped in the file, so the 5-digit FIPS key is
#     rebuilt with zfill and kept as a string; read as an integer the keys
#     collide and mis-assign counties.
_COUNTY_GEA_CROSSWALK_PATH = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "projections",
    "county_to_gea_mapping_cambium23.csv")

_df_county_gea = pd.read_csv(_COUNTY_GEA_CROSSWALK_PATH, encoding="utf-8-sig")
_df_county_gea["county_fips"] = (
    _df_county_gea["State FIPS"].astype(str).str.zfill(2)
    + _df_county_gea["County FIPS"].astype(str).str.zfill(3)
)

# Map a 5-digit county FIPS string to its new Cambium GEA region.
COUNTY_TO_GEA = dict(
    zip(_df_county_gea["county_fips"], _df_county_gea["Cambium GEA"])
)


def extract_city_name(row: str) -> str:
    """
    Extracts the city name from a string in the format 'ST, CityName'.

    If the input does not match the pattern of two uppercase letters,
    followed by a comma and a space, then the original string is returned.

    Args:
        row: A string in the format 'ST, CityName'.

    Returns:
        The extracted city name if the format matches; otherwise, the original string.
    """
    if not isinstance(row, str):
        return row
        
    # Regex to match exactly two uppercase letters, then a comma and a space, capturing the remainder
    match = re.match(r'^[A-Z]{2}, (.+)$', row)
    return match.group(1) if match else row
 

def map_metro_status(metro_status: Optional[str]) -> Optional[str]:
    """
    Maps raw metro status values to Urbanicity labels.

    Args:
        metro_status: String from 'in.puma_metro_status' column.

    Returns:
        'Urban', 'Suburban', or 'Rural' if recognized; otherwise the original input.
    """
    if not isinstance(metro_status, str):
        return metro_status
        
    mapping = {
        'In metro area, principal city': 'Urban',
        'In metro area, not/partially in principal city': 'Suburban',
        'Not/partially in metro area': 'Rural'
    }

    return mapping.get(metro_status.strip(), metro_status)


def standardize_fuel_name(fuel_desc: Any) -> Optional[str]:
    """Standardizes a fuel description into a recognized category or None.

    This function inspects an input fuel description (e.g., "Electric Heater",
    "Gas Furnace", "Propane Heater") and maps it to one of the following strings:
    "Electricity", "Natural Gas", "Propane", or "Fuel Oil". If the input is NaN,
    not a string, or does not contain any recognizable fuel keyword, the function
    returns None.

    Args:
        fuel_desc: A value representing the fuel description. It can be a string
            containing words like "Electric," "Gas," "Propane," or "Oil." It may
            also be NaN (pandas missing value) or another data type.

    Returns:
        One of the strings {"Electricity", "Natural Gas", "Propane", "Fuel Oil"}
        if a match is found, or None otherwise.
    """
    # Check if fuel_desc is NaN or not a string; return None if so
    if pd.isna(fuel_desc) or not isinstance(fuel_desc, str):
        return None
    
    # Convert the string to uppercase for case-insensitive matching
    fuel_desc_upper = fuel_desc.upper()
    
    # Match substrings for known fuel types
    if 'ELECTRIC' in fuel_desc_upper:
        return 'Electricity'
    elif 'GAS' in fuel_desc_upper:
        return 'Natural Gas'
    elif 'PROPANE' in fuel_desc_upper:
        return 'Propane'
    elif 'OIL' in fuel_desc_upper:
        return 'Fuel Oil'
    else:
        # If no match is found, return None
        return None


def preprocess_fuel_data(df: pd.DataFrame,
                         column_name: str
) -> pd.DataFrame:
    """Applies a standardization process to the specified fuel column in the DataFrame.

    This function applies 'standardize_fuel_name' to every value in the specified column
    and updates the DataFrame in-place.

    Args:
        df: The input pandas DataFrame containing fuel data.
        column_name: The name of the column to standardize.

    Returns:
        The updated DataFrame with standardized fuel names in the specified column.

    Raises:
        KeyError: If the specified column does not exist in the DataFrame.
        TypeError: If the DataFrame is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")
    
    print(f"Processing column: {column_name}")
    print(f"Initial data types: {df[column_name].dtype}")

    # Use .loc to avoid SettingWithCopyWarning when applying the function
    df.loc[:, column_name] = df[column_name].apply(standardize_fuel_name)

    print(f"Data types after processing: {df[column_name].dtype}")
    return df


def df_enduse_refactored(
    df_baseline: pd.DataFrame,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """Creates a standardized energy usage DataFrame and applies data quality filters.

    This function creates a new DataFrame with standardized column names and structure,
    calculates total consumption by fuel type, creates data quality flags for analysis,
    and sets invalid consumption values to NaN.

    Args:
        df_baseline: The baseline DataFrame containing raw EUSS/ResStock data.
        verbose: Whether to print detailed processing information.

    Returns:
        A standardized DataFrame with processed consumption data and data quality flags.

    Raises:
        ValueError: If required columns are missing from the input DataFrame.
    """
    # Updated to handle different enduses based on EQUIPMENT_SPECS and VALID_CATEGORIES.
    # - Rest of codebase updated so only initial columns created for cooling and replacement cost calculations performed
    # - This allows for a scenario where only heating is replaced AND one where heating and cooling systems are both replace with HP
    # - Resolves the excessive data columns and double counting with $8000 rebate. No longer need CDD projections.
    # valid_categories = list(EQUIPMENT_SPECS.keys())
    # valid_categories.append('cooling')

    # Initial check
    if df_baseline.empty:
        print("Warning: Input DataFrame is empty")
        return df_baseline

    # Standardize fuel names in the base columns
    df_baseline = preprocess_fuel_data(df_baseline, 'in.clothes_dryer')
    df_baseline = preprocess_fuel_data(df_baseline, 'in.cooking_range')

    # ===== STEP 1: Initialize with common columns (always present) =====
    df_enduse = pd.DataFrame({
        'weight': df_baseline['weight'],
        'square_footage': df_baseline['in.sqft'],
        'census_region': df_baseline['in.census_region'],
        'census_division': df_baseline['in.census_division'],
        'census_division_recs': df_baseline['in.census_division_recs'],
        'building_america_climate_zone': df_baseline['in.building_america_climate_zone'],
        'reeds_balancing_area': df_baseline['in.reeds_balancing_area'],
        'state': df_baseline['in.state'],
        'city': df_baseline['in.city'].apply(extract_city_name),
        'urbanicity': df_baseline['in.puma_metro_status'].apply(map_metro_status),
        'county': df_baseline['in.county'],
        'county_fips': df_baseline['in.county'].apply(lambda x: x[1:3] + x[4:7]),
        'puma': df_baseline['in.puma'],
        'county_and_puma': df_baseline['in.county_and_puma'],
        'weather_file_city': df_baseline['in.weather_file_city'],
        'Longitude': df_baseline['in.weather_file_longitude'],
        'Latitude': df_baseline['in.weather_file_latitude'],
        'building_type': df_baseline['in.geometry_building_type_recs'],
        'income': df_baseline['in.income'],
        'federal_poverty_level': df_baseline['in.federal_poverty_level'],
        'occupancy': df_baseline['in.occupants'],
        'tenure': df_baseline['in.tenure'],
        'vacancy_status': df_baseline['in.vacancy_status'],
        'vintage': df_baseline['in.vintage']
    })

    # ===== STEP 1b: Assign the new Cambium GEA region from the county crosswalk =====
    # ResStock's emissions-assessment region uses the retired "*c" codes, so set
    # gea_region from the county-to-GEA crosswalk instead. county_fips is the
    # 5-digit string key. Most counties map directly; a few may not if their FIPS
    # code changed between the crosswalk vintage and ResStock 2022.1.1.
    df_enduse['gea_region'] = df_enduse['county_fips'].map(COUNTY_TO_GEA)

    # Flag any homes whose county is not in the crosswalk. We leave the data as
    # is (no remap) for reproducibility; these homes get NaN gea_region and are
    # excluded from climate damages via NaN masking downstream.
    unmapped_gea = df_enduse['gea_region'].isna()
    if unmapped_gea.any():
        unmapped_fips = sorted(
            df_enduse.loc[unmapped_gea, 'county_fips'].dropna().unique())
        print(
            f"WARNING: {int(unmapped_gea.sum())} home(s) have no Cambium GEA "
            f"region (county FIPS not in the crosswalk): {unmapped_fips}. "
            f"Climate damages for these homes will be NaN."
        )

    # ===== STEP 2: Conditionally add category-specific columns =====
    
    # HEATING - only if in scope
    if 'heating' in VALID_CATEGORIES:
        df_enduse['base_heating_fuel'] = df_baseline['in.heating_fuel']
        df_enduse['heating_type'] = df_baseline['in.hvac_heating_type_and_fuel']
        df_enduse['base_heating_efficiency'] = df_baseline['in.hvac_heating_efficiency']
        # The home's existing heating system's own size, straight from the
        # ResStock baseline run -- not the retrofit heat pump's size. This is
        # what the avoided-replacement cost should be priced from (see
        # add_remdb_metrics in remdb_v4_installed_cost_utils.py).
        df_enduse['base_size_heating_system_primary_k_btu_h'] = (
            df_baseline['out.params.size_heating_system_primary_k_btu_h'])
        df_enduse['base_electricity_heating_consumption'] = df_baseline['out.electricity.heating.energy_consumption.kwh']
        df_enduse['base_fuelOil_heating_consumption'] = df_baseline['out.fuel_oil.heating.energy_consumption.kwh']
        df_enduse['base_naturalGas_heating_consumption'] = df_baseline['out.natural_gas.heating.energy_consumption.kwh']
        df_enduse['base_propane_heating_consumption'] = df_baseline['out.propane.heating.energy_consumption.kwh']

    # COOLING - only if in scope
    if 'cooling' in VALID_CATEGORIES:
        df_enduse['base_cooling_fuel'] = 'Electricity'  # Cooling is always electric
        df_enduse['cooling_type'] = df_baseline['in.hvac_cooling_type']
        df_enduse['base_cooling_efficiency'] = df_baseline['in.hvac_cooling_efficiency']
        # The home's existing cooling system's own size, straight from the
        # ResStock baseline run -- not the retrofit heat pump's size. Same
        # reasoning as base_size_heating_system_primary_k_btu_h above.
        df_enduse['base_size_cooling_system_primary_k_btu_h'] = (
            df_baseline['out.params.size_cooling_system_primary_k_btu_h'])
        df_enduse['base_electricity_cooling_consumption'] = df_baseline['out.electricity.cooling.energy_consumption.kwh']

    # WATER HEATING - only if in scope
    if 'waterHeating' in VALID_CATEGORIES:
        df_enduse['base_waterHeating_fuel'] = df_baseline['in.water_heater_fuel']
        df_enduse['waterHeating_type'] = df_baseline['in.water_heater_efficiency']
        df_enduse['base_electricity_waterHeating_consumption'] = df_baseline['out.electricity.hot_water.energy_consumption.kwh']
        df_enduse['base_fuelOil_waterHeating_consumption'] = df_baseline['out.fuel_oil.hot_water.energy_consumption.kwh']
        df_enduse['base_naturalGas_waterHeating_consumption'] = df_baseline['out.natural_gas.hot_water.energy_consumption.kwh']
        df_enduse['base_propane_waterHeating_consumption'] = df_baseline['out.propane.hot_water.energy_consumption.kwh']
    
    # CLOTHES DRYING - only if in scope
    if 'clothesDrying' in VALID_CATEGORIES:
        df_enduse['base_clothesDrying_fuel'] = df_baseline['in.clothes_dryer']
        df_enduse['base_electricity_clothesDrying_consumption'] = df_baseline['out.electricity.clothes_dryer.energy_consumption.kwh']
        df_enduse['base_naturalGas_clothesDrying_consumption'] = df_baseline['out.natural_gas.clothes_dryer.energy_consumption.kwh']
        df_enduse['base_propane_clothesDrying_consumption'] = df_baseline['out.propane.clothes_dryer.energy_consumption.kwh']
    
    # COOKING - only if in scope
    if 'cooking' in VALID_CATEGORIES:
        df_enduse['base_cooking_fuel'] = df_baseline['in.cooking_range']
        df_enduse['base_electricity_cooking_consumption'] = df_baseline['out.electricity.range_oven.energy_consumption.kwh']
        df_enduse['base_naturalGas_cooking_consumption'] = df_baseline['out.natural_gas.range_oven.energy_consumption.kwh']
        df_enduse['base_propane_cooking_consumption'] = df_baseline['out.propane.range_oven.energy_consumption.kwh']

    # ===== Whole-home baseline site energy (HOMES savings-fraction denominator) =====
    # The June 2026 HOMES rebate tiers key on the modeled whole-home percent
    # savings. TARE only changes heating and cooling, but the savings fraction
    # must be expressed against the WHOLE home, so carry ResStock's total site
    # energy through as the denominator. Home-level total (not category-specific),
    # so it is not masked by heating/cooling validity.
    #
    # WATCH THE SOURCE: 'out.site_energy.total.energy_consumption.kwh' is the
    # whole-home total across ALL fuels (natural gas, fuel oil, and propane are
    # reported in kWh-equivalent), NOT electricity. It is deliberately a
    # different column from the electricity total set below. This all-fuel value
    # is correct ONLY as the savings-fraction denominator; do NOT feed it into
    # any electricity, demand, or peak metric -- those use the electricity total.
    df_enduse['baseline_total_site_consumption'] = (
        df_baseline['out.site_energy.total.energy_consumption.kwh']
    )

    # ===== Retain per-home peak demand + whole-home electricity (metadata) =====
    # Pass-through columns for a short-term peak-load approximation done per
    # building ID outside this model (a simple annual max per home, not aligned
    # in time across homes). These are raw ResStock annual results, carried
    # unchanged:
    #   - peak electric demand during the peak cooling / heating hour (kW).
    #   - peak delivered HVAC thermal load (kBtu/hr) -- a separate, fuel-agnostic
    #     quantity from the electric demand above. Kept as its own pair; do not
    #     merge or combine it with the kW demand.
    #   - whole-home annual electricity use (kWh), the baseline side of the
    #     baseline-vs-retrofit electricity change.
    # Home-level values, so they are left unmasked by heating/cooling validity,
    # the same treatment as baseline_total_site_consumption above.
    df_enduse['base_peak_electricity_cooling_kw'] = (
        df_baseline['out.electricity.peak_when_cooling.kw']
    )
    df_enduse['base_peak_electricity_heating_kw'] = (
        df_baseline['out.electricity.peak_when_heating.kw']
    )
    df_enduse['base_peak_load_cooling_kbtu_hr'] = (
        df_baseline['out.load.cooling.peak.kbtu_hr']
    )
    df_enduse['base_peak_load_heating_kbtu_hr'] = (
        df_baseline['out.load.heating.peak.kbtu_hr']
    )
    # 'out.electricity.total.energy_consumption.kwh' is the whole-home
    # ELECTRICITY total (all electric end uses), NOT the all-fuel site energy
    # above. This is the baseline side of the baseline-vs-retrofit electricity
    # change, and the value every demand and peak metric must use.
    #
    # Prefix convention for the two whole-home totals: the 'base_' vs 'baseline_'
    # prefix does NOT tell you electricity from site energy -- read the token
    # after it ('...electricity...' vs '...site...'). Generally 'base_' marks an
    # equipment/fuel/metadata-level baseline reading that pairs with a retrofit
    # column (base_total_electricity_consumption pairs with
    # mp{mp}_total_electricity_consumption), while 'baseline_' marks a whole-home
    # or category aggregate used in the cost/rebate pipeline
    # (baseline_total_site_consumption, baseline_{category}_consumption).
    df_enduse['base_total_electricity_consumption'] = (
        df_baseline['out.electricity.total.energy_consumption.kwh']
    )

    # ===== STEP 3: Calculate total consumption for each category in scope =====
    for category in VALID_CATEGORIES:
        # Get consumption columns for this category
        consumption_columns = get_all_possible_fuel_columns(category)
        
        # Calculate total consumption by summing fuel-specific columns
        total_consumption = sum(
            df_enduse.get(col, pd.Series([], dtype=float)).fillna(0)
            for col in consumption_columns
        )
        df_enduse[f'baseline_{category}_consumption'] = total_consumption.replace(0, np.nan)
        print(f"Calculated total {category} consumption")
    
    # ===== STEP 4: Create data quality flags =====
    df_enduse = identify_valid_homes(df_enduse)
    
    # ===== STEP 5: Apply validation =====
    print("\nApplying data validation (baseline only):")
    for category in VALID_CATEGORIES:
        # Get validation mask (baseline, so menu_mp = 0)
        valid_mask = get_valid_calculation_mask(df_enduse, category, menu_mp=0, verbose=verbose)
        
        # Apply masking to consumption columns
        columns_to_mask = get_all_possible_fuel_columns(category)
        columns_to_mask.append(f'baseline_{category}_consumption')
        
        # Apply masking
        for col in columns_to_mask:
            if col in df_enduse.columns:
                non_nan_before = df_enduse[col].notna().sum()
                df_enduse.loc[~valid_mask, col] = np.nan
                non_nan_after = df_enduse[col].notna().sum()
                
                masked_count = non_nan_before - non_nan_after
                if masked_count > 0:
                    print(f"  {col}: Masked {masked_count} values")

    return df_enduse


def df_enduse_compare(
    df_mp: pd.DataFrame, 
    input_mp: str, 
    menu_mp: int, 
    df_baseline: pd.DataFrame, 
    df_cooking_range: pd.DataFrame,
    verbose: bool = VERBOSE
) -> pd.DataFrame:
    """Creates a comparison DataFrame by merging multiple DataFrames based on measure packages.

    This function constructs a new DataFrame (df_compare) that includes columns
    from df_mp, df_cooking_range, and merges them with df_baseline to compare
    baseline vs. measure package outputs.
    
    Only includes columns for equipment categories present in EQUIPMENT_SPECS.

    Args:
        df_mp: The main DataFrame containing modeling parameters and outputs.
        input_mp: The input measure package ID (e.g., 'upgrade09', 'upgrade10').
        menu_mp: The menu measure package number.
        df_baseline: The baseline DataFrame to merge with df_compare.
        df_cooking_range: Additional DataFrame for cooking range parameters/outputs.
        verbose: Whether to print detailed processing information.

    Returns:
        A merged DataFrame (df_compare) that includes relevant columns for
        baseline and measure packages comparison.
    """
    # Updated to handle different enduses based on EQUIPMENT_SPECS.
    # - Rest of codebase updated so only initial columns created for cooling and replacement cost calculations performed
    # - This allows for a scenario where only heating is replaced AND one where heating and cooling systems are both replace with HP
    # - Resolves the excessive data columns and double counting with $8000 rebate. No longer need CDD projections.
    VALID_CATEGORIES = list(EQUIPMENT_SPECS.keys())

    # ===== STEP 1: Initialize with common columns (always present) =====
    df_compare = pd.DataFrame({
        'hvac_has_ducts': df_mp['in.hvac_has_ducts']
    })
    
    # ===== STEP 2: Conditionally add category-specific metadata columns =====
    
    # HEATING - only if in scope
    if 'heating' in VALID_CATEGORIES:
        df_compare['hvac_heating_type_and_fuel'] = df_mp['in.hvac_heating_type_and_fuel']
        df_compare['hvac_heating_efficiency'] = df_mp['in.hvac_heating_efficiency']
        # df_compare['size_heat_pump_backup_k_btu_h'] = df_mp['out.params.size_heat_pump_backup_primary_k_btu_h']
        # This is the retrofit heat pump's capacity for THIS measure package,
        # not the baseline furnace's nameplate size. ResStock autosizes
        # equipment separately for every upgrade run (out.params.* comes from
        # df_mp, the MP3/MP4 upgrade output), so the value varies by measure
        # package and by whether the home is ducted. One heat pump serves both
        # heating and cooling, so this equals the cooling capacity column
        # below for every home. Only the heat pump's own upgrade cost is
        # priced off this column -- the heating replacement cost (the avoided
        # cost of replacing the OLD furnace/boiler) is priced off
        # base_size_heating_system_primary_k_btu_h instead, added in
        # df_enduse_refactored. See docs/SESSION_CHANGELOG_2026-08-20.md.
        df_compare['size_heating_system_primary_k_btu_h'] = df_mp['out.params.size_heating_system_primary_k_btu_h']
        # df_compare['size_heating_secondary_k_btu_h'] = df_mp['out.params.size_heating_system_secondary_k_btu_h']
        df_compare['upgrade_hvac_heating_efficiency'] = df_mp['upgrade.hvac_heating_efficiency']

        # ENERGY STAR override (MP3 only). MP3's modeled heat pump is
        # SEER 15 / 9.0 HSPF -- just below the ENERGY STAR minimum
        # (>= 16.0 SEER1 / >= 9.5 HSPF1) required for the federal heat-pump
        # rebate. To model MP3 as a rebate-eligible ENERGY STAR install, rewrite
        # its upgrade spec to that floor. Only the heating SEER value feeds the
        # REMDB v4 upgrade cost (pm2 = SEER1), so this raises MP3 capital cost
        # modestly; HSPF is bumped for spec accuracy but has no cost lever in
        # this model. Energy use is unchanged (it comes from the ResStock
        # simulation, not from this string).
        if menu_mp == 3:
            df_compare['upgrade_hvac_heating_efficiency'] = (
                df_compare['upgrade_hvac_heating_efficiency']
                .str.replace('SEER 15', 'SEER 16', regex=False)
                .str.replace('9.0 HSPF', '9.5 HSPF', regex=False)
            )
    
    # COOLING - only if in scope
    if 'cooling' in VALID_CATEGORIES:
        df_compare['hvac_cooling_type'] = df_mp['in.hvac_cooling_type']
        df_compare['hvac_cooling_efficiency'] = df_mp['in.hvac_cooling_efficiency']
        # Same retrofit heat-pump capacity as size_heating_system_primary_k_btu_h
        # above -- one heat pump serves both loads, so heating and cooling
        # capacity are identical for every home. Not the baseline air
        # conditioner's size. Only the heat pump's own upgrade cost is priced
        # off this column -- the cooling replacement cost (the avoided cost
        # of replacing the OLD air conditioner) is priced off
        # base_size_cooling_system_primary_k_btu_h instead, added in
        # df_enduse_refactored. See docs/SESSION_CHANGELOG_2026-08-20.md.
        df_compare['size_cooling_system_primary_k_btu_h'] = df_mp['out.params.size_cooling_system_primary_k_btu_h']
        df_compare['upgrade_hvac_cooling_efficiency'] = df_mp['upgrade.hvac_cooling_efficiency']

        # ENERGY STAR override (MP3 only), parallel to the heating override above
        # so the two upgrade-spec columns stay consistent. ResStock records the
        # MP3 cooling upgrade as the bare "Heat Pump" label (no SEER encoded), so
        # this replace is a no-op today; it keeps the columns in sync if a future
        # data vintage carries a numeric cooling spec.
        if menu_mp == 3:
            df_compare['upgrade_hvac_cooling_efficiency'] = (
                df_compare['upgrade_hvac_cooling_efficiency']
                .str.replace('SEER 15', 'SEER 16', regex=False)
                .str.replace('9.0 HSPF', '9.5 HSPF', regex=False)
            )

    # WATER HEATING - only if in scope
    if 'waterHeating' in VALID_CATEGORIES:
        df_compare['water_heater_efficiency'] = df_mp['in.water_heater_efficiency']
        df_compare['water_heater_fuel'] = df_mp['in.water_heater_fuel']
        df_compare['water_heater_in_unit'] = df_mp['in.water_heater_in_unit']
        df_compare['size_water_heater_gal'] = df_mp['out.params.size_water_heater_gal']
        df_compare['upgrade_water_heater_efficiency'] = df_mp['upgrade.water_heater_efficiency']
    
    # CLOTHES DRYING - only if in scope
    if 'clothesDrying' in VALID_CATEGORIES:
        df_compare['clothes_dryer_in_unit'] = df_mp['in.clothes_dryer']
        df_compare['upgrade_clothes_dryer'] = df_mp['upgrade.clothes_dryer']
    
    # COOKING - only if in scope
    if 'cooking' in VALID_CATEGORIES:
        df_compare['cooking_range_in_unit'] = df_cooking_range['in.cooking_range']
        df_compare['upgrade_cooking_range'] = df_cooking_range['upgrade.cooking_range']

    # ===== STEP 3: Add consumption columns for each category in scope =====
    for category in VALID_CATEGORIES:
        if category == 'heating':
            # Special handling for measure packages 9 and 10 (MP9, MP10) with enclosure upgrades
            if input_mp == 'upgrade09':
                menu_mp = 9
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

                # Basic Enclosure Package
                df_compare['base_insulation_atticFloor'] = df_mp['in.insulation_ceiling']
                df_compare['upgrade_insulation_atticFloor'] = df_mp['upgrade.insulation_ceiling']
                df_compare['floor_area_attic_ft2'] = df_mp['out.params.floor_area_attic_ft_2']

                df_compare['upgrade_infiltration_reduction'] = df_mp['upgrade.infiltration_reduction']

                df_compare['base_ducts'] = df_mp['in.ducts']
                df_compare['upgrade_duct_sealing'] = df_mp['upgrade.ducts']
                df_compare['duct_unconditioned_area_ft2'] = df_mp['out.params.duct_unconditioned_surface_area_ft_2']

                df_compare['base_insulation_wall'] = df_mp['in.insulation_wall']
                df_compare['upgrade_insulation_wall'] = df_mp['upgrade.insulation_wall']
                df_compare['wall_area_above_grade_ft2'] = df_mp['out.params.wall_area_above_grade_exterior_ft_2']

            elif input_mp == 'upgrade10':
                menu_mp = 10
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

                # Basic Enclosure Package (same as MP9)
                df_compare['base_insulation_atticFloor'] = df_mp['in.insulation_ceiling']
                df_compare['upgrade_insulation_atticFloor'] = df_mp['upgrade.insulation_ceiling']
                df_compare['floor_area_attic_ft2'] = df_mp['out.params.floor_area_attic_ft_2']

                df_compare['upgrade_infiltration_reduction'] = df_mp['upgrade.infiltration_reduction']

                df_compare['base_ducts'] = df_mp['in.ducts']
                df_compare['upgrade_duct_sealing'] = df_mp['upgrade.ducts']
                df_compare['duct_unconditioned_area_ft2'] = df_mp['out.params.duct_unconditioned_surface_area_ft_2']

                df_compare['base_insulation_wall'] = df_mp['in.insulation_wall']
                df_compare['upgrade_insulation_wall'] = df_mp['upgrade.insulation_wall']
                df_compare['wall_area_above_grade_ft2'] = df_mp['out.params.wall_area_above_grade_exterior_ft_2']

                # Enhanced Enclosure Package (MP10 only)
                df_compare['base_foundation_type'] = df_mp['in.geometry_foundation_type']
                df_compare['base_insulation_foundation_wall'] = df_mp['in.insulation_foundation_wall']
                df_compare['base_insulation_rim_joist'] = df_mp['in.insulation_rim_joist']
                df_compare['upgrade_insulation_foundation_wall'] = df_mp['upgrade.insulation_foundation_wall']
                df_compare['floor_area_foundation_ft2'] = df_mp['out.params.floor_area_foundation_ft_2']
                df_compare['rim_joist_area_above_grade_ft2'] = df_mp['out.params.rim_joist_area_above_grade_exterior_ft_2']

                df_compare['upgrade_seal_crawlspace'] = df_mp['upgrade.geometry_foundation_type']
                df_compare['base_insulation_roof'] = df_mp['in.insulation_roof']
                df_compare['upgrade_insulation_roof'] = df_mp['upgrade.insulation_roof']
                df_compare['roof_area_ft2'] = df_mp['out.params.roof_area_ft_2']

            else:
                # Standard heating consumption (no enclosure upgrades)
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp['out.electricity.heating.energy_consumption.kwh'].round(2)

        elif category == 'cooling':
            df_compare[f'mp{menu_mp}_cooling_consumption'] = df_mp['out.electricity.cooling.energy_consumption.kwh'].round(2)

        elif category == 'waterHeating':
            df_compare[f'mp{menu_mp}_waterHeating_consumption'] = df_mp['out.electricity.hot_water.energy_consumption.kwh'].round(2)

        elif category == 'clothesDrying':
            df_compare[f'mp{menu_mp}_clothesDrying_consumption'] = df_mp['out.electricity.clothes_dryer.energy_consumption.kwh'].round(2)

        elif category == 'cooking':
            df_compare[f'mp{menu_mp}_cooking_consumption'] = df_cooking_range['out.electricity.range_oven.energy_consumption.kwh'].round(2)

    # ===== STEP 3b: Retain per-home peak demand + whole-home electricity =====
    # Post-retrofit counterparts of the baseline pass-through columns added in
    # df_enduse_refactored, for the same per-building-ID peak-load approximation
    # done outside this model. Raw ResStock annual results from the upgrade file,
    # carried unchanged. The upgrade files also publish ResStock's own
    # baseline-minus-upgrade delta as the ".savings" columns, kept here so the
    # peak change is available without re-differencing:
    #   - peak electric demand during the peak cooling / heating hour (kW),
    #     plus its savings.
    #   - peak delivered HVAC thermal load (kBtu/hr) plus its savings -- a
    #     separate, fuel-agnostic quantity; kept independent of the kW demand.
    #   - whole-home annual electricity use (kWh), the retrofit side of the
    #     baseline-vs-retrofit electricity change.
    # Home-level values; they are not added to any columns_to_mask list below, so
    # STEP 6 category validation leaves them intact.
    df_compare[f'mp{menu_mp}_peak_electricity_cooling_kw'] = (
        df_mp['out.electricity.peak_when_cooling.kw']
    )
    df_compare[f'mp{menu_mp}_peak_electricity_heating_kw'] = (
        df_mp['out.electricity.peak_when_heating.kw']
    )
    df_compare[f'mp{menu_mp}_peak_electricity_cooling_kw_savings'] = (
        df_mp['out.electricity.peak_when_cooling.kw.savings']
    )
    df_compare[f'mp{menu_mp}_peak_electricity_heating_kw_savings'] = (
        df_mp['out.electricity.peak_when_heating.kw.savings']
    )
    df_compare[f'mp{menu_mp}_peak_load_cooling_kbtu_hr'] = (
        df_mp['out.load.cooling.peak.kbtu_hr']
    )
    df_compare[f'mp{menu_mp}_peak_load_heating_kbtu_hr'] = (
        df_mp['out.load.heating.peak.kbtu_hr']
    )
    df_compare[f'mp{menu_mp}_peak_load_cooling_kbtu_hr_savings'] = (
        df_mp['out.load.cooling.peak.kbtu_hr.savings']
    )
    df_compare[f'mp{menu_mp}_peak_load_heating_kbtu_hr_savings'] = (
        df_mp['out.load.heating.peak.kbtu_hr.savings']
    )
    df_compare[f'mp{menu_mp}_total_electricity_consumption'] = (
        df_mp['out.electricity.total.energy_consumption.kwh']
    )

    # ===== STEP 4: Merge with baseline DataFrame =====
    df_compare = pd.merge(df_baseline, df_compare, how='inner', left_index=True, right_index=True)
    
    # ===== STEP 5: Ensure validation flags are preserved =====
    validation_flags = [col for col in df_baseline.columns 
                       if col.startswith('valid_') or col.startswith('include_')]
    
    for flag in validation_flags:
        if flag in df_baseline.columns and flag not in df_compare.columns:
            df_compare[flag] = df_baseline[flag]
    
    # ===== STEP 6: Apply combined validation (data quality + retrofit status) =====
    print("\nApplying combined validation (data quality + retrofit status):")
    for category in VALID_CATEGORIES:
        # Get combined validation mask
        valid_mask = get_valid_calculation_mask(df_compare, category, menu_mp, verbose=verbose)
        
        # Determine which columns to mask for this category
        category_cols = []
        
        # Add basic consumption columns
        fuel_columns = get_all_possible_fuel_columns(category)
        category_cols.extend([col for col in fuel_columns if col in df_compare.columns])
        
        # Add total baseline column
        baseline_col = f'baseline_{category}_consumption'
        if baseline_col in df_compare.columns:
            category_cols.append(baseline_col)
        
        # Add measure package column
        mp_col = f'mp{menu_mp}_{category}_consumption'
        if mp_col in df_compare.columns:
            category_cols.append(mp_col)
        
        # Apply masking
        for col in category_cols:
            non_nan_before = df_compare[col].notna().sum()
            df_compare.loc[~valid_mask, col] = np.nan
            non_nan_after = df_compare[col].notna().sum()
            
            masked_count = non_nan_before - non_nan_after
            if masked_count > 0:
                print(f"  {col}: Masked {masked_count} values")

    # ===== STEP 7: Whole-home modeled savings fraction (HOMES rebate tiers) =====
    # Whole-home percent savings = (HVAC energy delta) / whole-home baseline site
    # energy. Only heating and cooling change under a retrofit, so the numerator
    # is the heating+cooling consumption drop; the denominator is the unchanged
    # whole-home baseline total. Consumed only for electric-resistance homes in
    # the June 2026 HOMES rebate, where every quantity is electric kWh.
    #
    # NOTE: the heating/cooling terms are TARE's degree-day-adjusted consumption
    # while the denominator is raw ResStock site energy; this small
    # adjusted-vs-raw mismatch is an accepted approximation.
    heating_col = f'mp{menu_mp}_heating_consumption'
    cooling_col = f'mp{menu_mp}_cooling_consumption'
    savings_frac_col = f'mp{menu_mp}_modeled_savings_frac'

    # Heating delta propagates NaN for homes without valid heating (which are
    # excluded from any rebate anyway). Cooling delta is 0 when a home has no
    # cooling, so a heating-only home still gets a heating-based fraction.
    heating_delta = (
        df_compare['baseline_heating_consumption'] - df_compare[heating_col]
    )
    if cooling_col in df_compare.columns:
        cooling_delta = (
            df_compare['baseline_cooling_consumption'].fillna(0.0)
            - df_compare[cooling_col].fillna(0.0)
        )
    else:
        cooling_delta = 0.0

    df_compare[savings_frac_col] = (
        (heating_delta + cooling_delta)
        / df_compare['baseline_total_site_consumption']
    ).astype('float64')

    return df_compare
