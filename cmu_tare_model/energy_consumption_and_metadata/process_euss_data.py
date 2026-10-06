import os
import pandas as pd
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    ANCHOR_YEAR,
    EQUIPMENT_SPECS,
    VALID_CATEGORIES,
    VERBOSE,
    RESSTOCK_RELEASE_THIS_RUN,
    RESSTOCK_RELEASE_AND_MP,
    ALLOWED_HOUSING_TYPES,
    EXCLUDED_STATES,
    )

from cmu_tare_model.utils.validation_framework import get_valid_calculation_mask
from cmu_tare_model.utils.calculation_utils import (
    RESSTOCK_FUEL_NAMES,
    find_enduse_columns,
    get_all_possible_fuel_columns,
    get_consumption_component_columns,
    get_resstock_savings_column,
    identify_valid_homes,
    compute_funnel_stage_row,
    print_masking_funnel_stage,
    )
from cmu_tare_model.utils.resstock_schema import RESSTOCK_COLUMN_MAP, resstock_col
from cmu_tare_model.utils.degree_day_consumption_utils import (
    get_degree_day_adjusted_consumption_by_fuel,
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


def read_resstock_2025_1_parquet(mp: int) -> pd.DataFrame:
    """Reads one ResStock 2025.1 national AMY2018 parquet file.

    2025.1 file names do not follow the 2022.1.1 'upgradeNN' zero-padded
    convention (mp_to_upgrade), so this builds the path directly: the
    baseline is upgrade0.parquet and Upgrade 05 is upgrade5.parquet, with no
    padding. The parquet file also does not carry bldg_id as an index the
    way the 2022.1.1 CSV read does (index_col="bldg_id"), so it is set here
    after the read.

    Args:
        mp: Measure package number under the '2025.1' release (0 for the
            baseline, 5 for the dual-fuel Upgrade 05).

    Returns:
        The raw ResStock 2025.1 frame, indexed by bldg_id.

    Raises:
        ValueError: If mp is not in RESSTOCK_RELEASE_AND_MP['2025.1'].
    """
    if mp not in RESSTOCK_RELEASE_AND_MP['2025.1']:
        raise ValueError(
            f"mp={mp} is not a loadable 2025.1 package; expected one of "
            f"{sorted(RESSTOCK_RELEASE_AND_MP['2025.1'])}")
    filename = f"upgrade{mp}.parquet"
    file_path = os.path.join(
        PROJECT_ROOT, "data", "resstock_2025_1", filename)
    df_raw = pd.read_parquet(file_path)
    return df_raw.set_index("bldg_id")


def read_resstock_2022_1_1_csv(mp: int) -> pd.DataFrame:
    """Reads one ResStock 2022.1.1 (EUSS) national AMY2018 CSV file.

    2022.1.1 names the baseline 'baseline_metadata_and_annual_results.csv' and
    each package 'upgradeNN_metadata_and_annual_results.csv', zero-padded.

    Args:
        mp: Measure package number under the '2022.1.1' release (0 for the
            baseline, 3 or 4 for the heat-pump packages).

    Returns:
        The raw ResStock 2022.1.1 frame, indexed by bldg_id.

    Raises:
        ValueError: If mp is not in RESSTOCK_RELEASE_AND_MP['2022.1.1'].
    """
    if mp not in RESSTOCK_RELEASE_AND_MP['2022.1.1']:
        raise ValueError(
            f"mp={mp} is not a loadable 2022.1.1 package; expected one of "
            f"{sorted(RESSTOCK_RELEASE_AND_MP['2022.1.1'])}")
    file_prefix = 'baseline' if mp == 0 else f'upgrade{mp:02d}'
    filename = f"{file_prefix}_metadata_and_annual_results.csv"
    file_path = os.path.join(
        PROJECT_ROOT, "cmu_tare_model", "data", "euss_data",
        "resstock_amy2018_release_1.1", "national", "csv", filename)
    # low_memory=False reads the whole file before choosing column types, so
    # mixed-type columns come back as text instead of raising a DtypeWarning.
    return pd.read_csv(file_path, low_memory=False, index_col="bldg_id")


def load_and_filter_upgrade(
    menu_mp: int,
    state: Optional[str] = None,
    city: Optional[str] = None,
    verbose: bool = VERBOSE,
    release: str = RESSTOCK_RELEASE_THIS_RUN,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads one ResStock upgrade file (either release) and applies TARE's scope filters.

    Reorganizes the inline filtering that both the baseline and MP5 notebook
    cells duplicated into one reusable function, and adds two new filter
    stages ahead of the existing ones (D3, Phase 3): ResStock's own
    applicability flag and the Alaska/Hawaii exclusion. Every existing
    filter's logic and order is unchanged -- occupancy, housing type, and
    the state/city filter all run exactly as they did inline in the
    notebook cells, just routed through resstock_col() instead of hardcoded
    'in.*' literals.

    Filter order:
        1. Load -- no filter; the starting population for the funnel.
        2. Applicability -- ResStock's own upgrade_applicable flag (D3, the
           new leading filter; a no-op for menu_mp=0, where every row is
           True). Applied to both releases so the two samples use the same
           scope filters; for 2022.1.1 MP3/MP4 it removes no in-scope homes.
        3. Occupancy -- unchanged from the existing notebook cells.
        4. Housing type -- unchanged from the existing notebook cells.
        5. Alaska/Hawaii exclusion -- new; see EXCLUDED_STATES. A no-op for
           2022.1.1, which covers only the lower 48 states and DC.
        6. Geographic filter -- unchanged from the existing notebook cells,
           now driven by the state/city arguments instead of the notebook's
           own input_state/input_cityFilter globals. Only recorded as a
           funnel stage when state is given, since a National analysis
           (state=None) leaves the population unchanged from stage 5.

    Every stage is recorded as a funnel row via
    calculation_utils.compute_funnel_stage_row and returned as a DataFrame,
    instead of being reported only as loose print statements.

    Args:
        menu_mp: The measure package to load (0 for baseline, or one of
            RESSTOCK_RELEASE_AND_MP[release]).
        state: Two-letter state abbreviation to filter to, or None for the
            full (49-state-plus-DC, after the AK/HI exclusion) national
            footprint.
        city: City name to filter to within state, matching the 'in.city'
            column's 'ST, CityName' format, or None. Ignored if state is
            None.
        verbose: Whether to also print each filter stage's funnel row as it
            runs, via print_masking_funnel_stage.
        release: ResStock release to load ('2022.1.1' or '2025.1'). Defaults
            to RESSTOCK_RELEASE_THIS_RUN.

    Returns:
        A tuple of (df_filtered, df_funnel):
            df_filtered: The frame after every filter stage above, still
                carrying raw ResStock column names -- this function runs
                before df_enduse_refactored/df_enduse_compare rename
                anything.
            df_funnel: One row per filter stage (stage label, rdu count,
                weighted homes, and a percent-share column per baseline
                heating fuel bucket), in stage order.

    Raises:
        ValueError: If city is given but state is None, or release is unknown.
    """
    if city is not None and state is None:
        raise ValueError(
            "city was given without state; a city filter requires a state.")
    if release not in RESSTOCK_RELEASE_AND_MP:
        raise ValueError(
            f"Unknown release '{release}'; expected one of "
            f"{sorted(RESSTOCK_RELEASE_AND_MP)}")

    weight_col = 'weight'
    heating_fuel_col = resstock_col(release, 'heating_fuel')
    heating_type_col = resstock_col(release, 'heating_type_and_fuel')

    funnel_rows = []

    def record_stage(df_stage: pd.DataFrame, label: str) -> pd.DataFrame:
        # Computes the row once and both accumulates it into df_funnel and
        # (optionally) prints it, so the funnel math is never duplicated --
        # see compute_funnel_stage_row in calculation_utils.py.
        funnel_rows.append(compute_funnel_stage_row(
            df_stage, label,
            weight_col=weight_col, heating_fuel_col=heating_fuel_col,
            heating_type_col=heating_type_col))
        if verbose:
            print_masking_funnel_stage(
                df_stage, label,
                weight_col=weight_col, heating_fuel_col=heating_fuel_col,
                heating_type_col=heating_type_col)
        return df_stage

    # ===== Stage 1: Load =====
    if release == '2025.1':
        df_filtered = read_resstock_2025_1_parquet(menu_mp)
    else:
        df_filtered = read_resstock_2022_1_1_csv(menu_mp)
    record_stage(df_filtered, 'load')

    # ===== Stage 2: Applicability (new, D3) =====
    # Applied to both releases so the two samples use the same scope filters.
    # For 2022.1.1 MP3/MP4 it removes no in-scope homes.
    applicable_col = resstock_col(release, 'upgrade_applicable')
    is_applicable = df_filtered[applicable_col].astype(bool)
    df_filtered = df_filtered.loc[is_applicable]
    record_stage(df_filtered, 'applicability')

    # ===== Stage 3: Occupancy (unchanged) =====
    vacancy_col = resstock_col(release, 'vacancy_status')
    is_occupied = df_filtered[vacancy_col] == 'Occupied'
    df_filtered = df_filtered.loc[is_occupied]
    record_stage(df_filtered, 'occupancy')

    # ===== Stage 4: Housing type (unchanged) =====
    building_type_col = resstock_col(release, 'building_type')
    is_allowed_housing = df_filtered[building_type_col].isin(ALLOWED_HOUSING_TYPES)
    df_filtered = df_filtered.loc[is_allowed_housing]
    record_stage(df_filtered, 'housing_type')

    # ===== Stage 5: Alaska/Hawaii exclusion (new) =====
    state_col = resstock_col(release, 'state')
    is_included_state = ~df_filtered[state_col].isin(EXCLUDED_STATES)
    df_filtered = df_filtered.loc[is_included_state]
    record_stage(df_filtered, 'exclude_AK_HI')

    # ===== Stage 6: Geographic filter (unchanged) =====
    if state is not None:
        df_filtered = df_filtered.loc[df_filtered[state_col].eq(state)]
        if city is not None:
            city_col = resstock_col(release, 'city')
            df_filtered = df_filtered.loc[
                df_filtered[city_col].eq(f"{state}, {city}")]
        record_stage(df_filtered, 'geographic_filter')

    df_funnel = pd.DataFrame(funnel_rows)
    return df_filtered, df_funnel


# Backup fuel abbreviations published in the dual-fuel upgrade string, mapped
# to the same fuel labels FUEL_MAPPING and this module use elsewhere
# ('Natural Gas', not the abbreviation 'NG'). Only 'NG' has been observed in
# the published data (Phase 1 audit, Section B.7); the other entries are
# here so a future dual-fuel package with a different backup fuel fails
# loudly instead of silently mapping to the wrong fuel.
_DUAL_FUEL_BACKUP_FUEL_LABELS = {
    'NG': 'Natural Gas',
    'FO': 'Fuel Oil',
    'LPG': 'Propane',
    'Electric': 'Electricity',
}

# Matches strings like:
#   "Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5% AFUE NG, 35F switchover"
# Both published AFUE tiers (92.5 and 95.0 percent) fit this same pattern;
# only the numeric fields and the backup fuel abbreviation vary.
_DUAL_FUEL_HEATING_EFFICIENCY_PATTERN = re.compile(
    r'^Dual-Fuel ASHP, SEER ([\d.]+), ([\d.]+) HSPF2, Integrated Backup, '
    r'([\d.]+)% AFUE (\w+), (\d+)F switchover$'
)


def parse_dual_fuel_heating_efficiency(spec: str) -> Dict[str, Any]:
    """Parses a ResStock 2025.1 dual-fuel upgrade.hvac_heating_efficiency string.

    The dual-fuel package (Upgrade 05) publishes one option string per home
    that packs the heat pump's SEER2/HSPF2 rating, the backup furnace's fuel
    and AFUE, and the outdoor temperature at which the system switches from
    the heat pump to the backup furnace. This is a dedicated parser for that
    format -- it must not be run through the MP3 ENERGY STAR override's
    plain substring replace, which would match 'SEER 15' inside 'SEER 15.2'
    and corrupt the value (see the release guard on that override, below).

    Args:
        spec: The raw upgrade.hvac_heating_efficiency string, for example
            "Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5%
            AFUE NG, 35F switchover".

    Returns:
        A dict with:
            hp_seer2 (float): the heat pump's SEER2 rating.
            hp_hspf2 (float): the heat pump's HSPF2 rating.
            backup_fuel (str): the backup furnace's fuel, using the same
                labels as FUEL_MAPPING ('Natural Gas', not 'NG').
            backup_afue (float): the backup furnace's AFUE as a fraction
                (0.925, not 92.5), matching the REMDB v4 AFUE convention.
            switchover_f (float): the outdoor temperature, in degrees F, at
                which the system switches from the heat pump to the backup
                furnace.

    Raises:
        ValueError: If spec is not a string, does not match the published
            dual-fuel format, or names a backup fuel abbreviation this
            function does not recognize.
    """
    if not isinstance(spec, str):
        raise ValueError(
            f"Expected a dual-fuel heating efficiency string, got "
            f"{type(spec).__name__}: {spec!r}")

    match = _DUAL_FUEL_HEATING_EFFICIENCY_PATTERN.match(spec.strip())
    if match is None:
        raise ValueError(
            f"'{spec}' does not match the published dual-fuel "
            f"upgrade.hvac_heating_efficiency format")

    seer2_str, hspf2_str, afue_pct_str, fuel_abbrev, switchover_str = match.groups()

    if fuel_abbrev not in _DUAL_FUEL_BACKUP_FUEL_LABELS:
        raise ValueError(
            f"Unrecognized dual-fuel backup fuel abbreviation "
            f"'{fuel_abbrev}' in '{spec}'; expected one of "
            f"{sorted(_DUAL_FUEL_BACKUP_FUEL_LABELS)}")

    return {
        'hp_seer2': float(seer2_str),
        'hp_hspf2': float(hspf2_str),
        'backup_fuel': _DUAL_FUEL_BACKUP_FUEL_LABELS[fuel_abbrev],
        'backup_afue': float(afue_pct_str) / 100,
        'switchover_f': float(switchover_str),
    }


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
                         column_name: str,
                         verbose: bool = VERBOSE
) -> pd.DataFrame:
    """Applies a standardization process to the specified fuel column in the DataFrame.

    This function applies 'standardize_fuel_name' to every value in the specified column
    and updates the DataFrame in-place.

    Args:
        df: The input pandas DataFrame containing fuel data.
        column_name: The name of the column to standardize.
        verbose: Whether to print the column's data type before and after.

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
    
    if verbose:
        print(f"Processing column: {column_name}")
        print(f"Initial data types: {df[column_name].dtype}")

    # Use .loc to avoid SettingWithCopyWarning when applying the function
    df.loc[:, column_name] = df[column_name].apply(standardize_fuel_name)

    if verbose:
        print(f"Data types after processing: {df[column_name].dtype}")
    return df


def df_enduse_refactored(
    df_baseline: pd.DataFrame,
    applicable_bldg_ids: List[pd.Index],
    verbose: bool = VERBOSE,
    release: str = RESSTOCK_RELEASE_THIS_RUN
) -> pd.DataFrame:
    """Creates a standardized energy usage DataFrame and applies data quality filters.

    This function creates a new DataFrame with standardized column names and structure,
    calculates total consumption by fuel type, creates data quality flags for analysis,
    and sets invalid consumption values to NaN.

    Args:
        df_baseline: The baseline DataFrame containing raw EUSS/ResStock data.
        applicable_bldg_ids: One index per measure package in the run: the
            bldg_ids ResStock applied that package to. A home is in the study
            sample (include_sample) only if it is in every one, so all packages
            compare the same homes.
        verbose: Whether to print detailed processing information.
        release: ResStock release the frame was loaded from ('2022.1.1' or
            '2025.1'). Selects which physical column names to read via
            RESSTOCK_COLUMN_MAP. Defaults to RESSTOCK_RELEASE_THIS_RUN, so
            existing 2022.1.1 callers that don't pass this argument are
            unaffected.

    Returns:
        A standardized DataFrame with processed consumption data and data quality flags.

    Raises:
        ValueError: If df_baseline is empty, required columns are missing
            from it, or applicable_bldg_ids is empty.
        TypeError: If an entry of applicable_bldg_ids is not a pd.Index.
    """
    # Updated to handle different enduses based on EQUIPMENT_SPECS and VALID_CATEGORIES.
    # - Rest of codebase updated so only initial columns created for cooling and replacement cost calculations performed
    # - This allows for a scenario where only heating is replaced AND one where heating and cooling systems are both replace with HP
    # - Resolves the excessive data columns and double counting with $8000 rebate. No longer need CDD projections.
    # valid_categories = list(EQUIPMENT_SPECS.keys())
    # valid_categories.append('cooling')

    if not applicable_bldg_ids:
        raise ValueError(
            "applicable_bldg_ids needs one index per measure package in the run")
    for package_ids in applicable_bldg_ids:
        if not isinstance(package_ids, pd.Index):
            raise TypeError(
                "applicable_bldg_ids entries must be pd.Index, "
                f"got {type(package_ids)}")

    # An empty table cannot produce a run, so stop here and not further down.
    if df_baseline.empty:
        raise ValueError(
            "df_baseline has no rows; check the scope and geographic filters "
            "that built it")

    # Standardize fuel names in the base columns
    df_baseline = preprocess_fuel_data(
        df_baseline, resstock_col(release, 'clothes_dryer_type'),
        verbose=verbose)
    df_baseline = preprocess_fuel_data(
        df_baseline, resstock_col(release, 'cooking_range_type'),
        verbose=verbose)

    # ===== STEP 1: Initialize with common columns (always present) =====
    # Every physical column name below is looked up by its logical name so
    # this function works unchanged on both ResStock releases; weight is not
    # looked up because its name is identical in both releases.
    df_enduse = pd.DataFrame({
        'weight': df_baseline['weight'],
        'square_footage': df_baseline[resstock_col(release, 'floor_area_sqft')],
        'census_region': df_baseline[resstock_col(release, 'census_region')],
        'census_division': df_baseline[resstock_col(release, 'census_division')],
        'census_division_recs': df_baseline[resstock_col(release, 'census_division_recs')],
        'building_america_climate_zone': df_baseline[resstock_col(release, 'climate_zone_ba')],
        'climate_zone_iecc': df_baseline[resstock_col(release, 'climate_zone_iecc')],
        'reeds_balancing_area': df_baseline[resstock_col(release, 'reeds_balancing_area')],
        'state': df_baseline[resstock_col(release, 'state')],
        'city': df_baseline[resstock_col(release, 'city')].apply(extract_city_name),
        'urbanicity': df_baseline[resstock_col(release, 'metro_status')].apply(map_metro_status),
        'county': df_baseline[resstock_col(release, 'county')],
        'county_fips': df_baseline[resstock_col(release, 'county')].apply(
            lambda x: x[1:3] + x[4:7]),
        'puma': df_baseline[resstock_col(release, 'puma')],
        'county_and_puma': df_baseline[resstock_col(release, 'county_and_puma')],
        'weather_file_city': df_baseline[resstock_col(release, 'weather_city')],
        'Longitude': df_baseline[resstock_col(release, 'longitude')],
        'Latitude': df_baseline[resstock_col(release, 'latitude')],
        'building_type': df_baseline[resstock_col(release, 'building_type')],
        'income': df_baseline[resstock_col(release, 'income')],
        'federal_poverty_level': df_baseline[resstock_col(release, 'federal_poverty_level')],
        'occupancy': df_baseline[resstock_col(release, 'occupants')],
        'tenure': df_baseline[resstock_col(release, 'tenure')],
        'vacancy_status': df_baseline[resstock_col(release, 'vacancy_status')],
        'vintage': df_baseline[resstock_col(release, 'vintage')]
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
        # Printed even when quiet, so a newly unmapped county is seen.
        print(
            f"WARNING: {int(unmapped_gea.sum())} rdu have no Cambium GEA "
            f"region (county FIPS not in the crosswalk): {unmapped_fips}. "
            f"Climate damages for these rdu will be NaN."
        )

    # ===== STEP 1c: Electric panel service rating (2025.1 only) =====
    # A home's main panel amperage, published starting ResStock 2025.1.
    # There is no 2022.1.1 counterpart, so this column is simply absent from
    # a 2022.1.1 call's output rather than filled with a placeholder.
    if release == '2025.1':
        df_enduse['panel_service_rating_amps'] = (
            df_baseline[resstock_col(release, 'panel_service_rating')])

    # ===== STEP 2: Conditionally add category-specific columns =====
    
    # HEATING - only if in scope
    if 'heating' in VALID_CATEGORIES:
        df_enduse['base_heating_fuel'] = df_baseline[resstock_col(release, 'heating_fuel')]
        df_enduse['heating_type'] = df_baseline[resstock_col(release, 'heating_type_and_fuel')]
        df_enduse['base_heating_efficiency'] = df_baseline[resstock_col(release, 'heating_efficiency')]
        # The home's existing heating system's own size, straight from the
        # ResStock baseline run -- not the retrofit heat pump's size. This is
        # what the avoided-replacement cost should be priced from (see
        # add_remdb_metrics in remdb_v4_installed_cost_utils.py).
        df_enduse['base_size_heating_system_primary_k_btu_h'] = (
            df_baseline[resstock_col(release, 'size_heating_primary')])
        df_enduse['base_electricity_heating_consumption'] = df_baseline[resstock_col(release, 'heating_electricity')]
        df_enduse['base_fuelOil_heating_consumption'] = df_baseline[resstock_col(release, 'heating_fuel_oil')]
        df_enduse['base_naturalGas_heating_consumption'] = df_baseline[resstock_col(release, 'heating_natural_gas')]
        df_enduse['base_propane_heating_consumption'] = df_baseline[resstock_col(release, 'heating_propane')]

    # COOLING - only if in scope
    if 'cooling' in VALID_CATEGORIES:
        df_enduse['base_cooling_fuel'] = 'Electricity'  # Cooling is always electric
        df_enduse['cooling_type'] = df_baseline[resstock_col(release, 'cooling_type')]
        df_enduse['base_cooling_efficiency'] = df_baseline[resstock_col(release, 'cooling_efficiency')]
        # The home's existing cooling system's own size, straight from the
        # ResStock baseline run -- not the retrofit heat pump's size. Same
        # reasoning as base_size_heating_system_primary_k_btu_h above.
        df_enduse['base_size_cooling_system_primary_k_btu_h'] = (
            df_baseline[resstock_col(release, 'size_cooling_primary')])
        df_enduse['base_electricity_cooling_consumption'] = df_baseline[resstock_col(release, 'cooling_electricity')]

    # WATER HEATING - only if in scope
    if 'waterHeating' in VALID_CATEGORIES:
        df_enduse['base_waterHeating_fuel'] = df_baseline[resstock_col(release, 'water_heater_fuel')]
        df_enduse['waterHeating_type'] = df_baseline[resstock_col(release, 'water_heater_efficiency')]
        df_enduse['base_electricity_waterHeating_consumption'] = df_baseline[resstock_col(release, 'hot_water_electricity')]
        df_enduse['base_fuelOil_waterHeating_consumption'] = df_baseline[resstock_col(release, 'hot_water_fuel_oil')]
        df_enduse['base_naturalGas_waterHeating_consumption'] = df_baseline[resstock_col(release, 'hot_water_natural_gas')]
        df_enduse['base_propane_waterHeating_consumption'] = df_baseline[resstock_col(release, 'hot_water_propane')]

    # CLOTHES DRYING - only if in scope
    if 'clothesDrying' in VALID_CATEGORIES:
        df_enduse['base_clothesDrying_fuel'] = df_baseline[resstock_col(release, 'clothes_dryer_type')]
        df_enduse['base_electricity_clothesDrying_consumption'] = df_baseline[resstock_col(release, 'clothes_dryer_electricity')]
        df_enduse['base_naturalGas_clothesDrying_consumption'] = df_baseline[resstock_col(release, 'clothes_dryer_natural_gas')]
        df_enduse['base_propane_clothesDrying_consumption'] = df_baseline[resstock_col(release, 'clothes_dryer_propane')]

    # COOKING - only if in scope
    if 'cooking' in VALID_CATEGORIES:
        df_enduse['base_cooking_fuel'] = df_baseline[resstock_col(release, 'cooking_range_type')]
        df_enduse['base_electricity_cooking_consumption'] = df_baseline[resstock_col(release, 'cooking_electricity')]
        df_enduse['base_naturalGas_cooking_consumption'] = df_baseline[resstock_col(release, 'cooking_natural_gas')]
        df_enduse['base_propane_cooking_consumption'] = df_baseline[resstock_col(release, 'cooking_propane')]

    # ===== Separately reported heating/cooling components (fans, backup) =====
    # Every part the ResStock file publishes for heating and cooling (fans and
    # pumps, heat-pump backup), one column per fuel and part, read from the file
    # itself. Primary-system energy is added above; parts do not overlap, so a
    # fuel's full use is its primary-system column plus its parts.
    for category in ('heating', 'cooling'):
        if category not in VALID_CATEGORIES:
            continue
        for fuel, part, resstock_column in find_enduse_columns(
                df_baseline.columns, category):
            if part == 'primary_system':
                continue
            col = f'base_{fuel}_{category}_{part}_consumption'
            df_enduse[col] = df_baseline[resstock_column]

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
        df_baseline[resstock_col(release, 'site_energy_total')]
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
        df_baseline[resstock_col(release, 'peak_electricity_cooling')]
    )
    df_enduse['base_peak_electricity_heating_kw'] = (
        df_baseline[resstock_col(release, 'peak_electricity_heating')]
    )
    df_enduse['base_peak_load_cooling_kbtu_hr'] = (
        df_baseline[resstock_col(release, 'peak_load_cooling')]
    )
    df_enduse['base_peak_load_heating_kbtu_hr'] = (
        df_baseline[resstock_col(release, 'peak_load_heating')]
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
        df_baseline[resstock_col(release, 'electricity_total')]
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
        # A true zero stays a zero; STEP 5 blanks homes outside the sample.
        df_enduse[f'baseline_{category}_consumption'] = total_consumption
        if verbose:
            print(f"Calculated total {category} consumption")
    
    # ===== STEP 4: Create data quality flags =====
    df_enduse = identify_valid_homes(df_enduse)

    # ===== STEP 4b: Study sample flag =====
    # The one definition of the study sample: a heating system the study can
    # replace and cost, a central or room AC of the home's own
    # (include_cooling), and applied in every package in the run. Set here once;
    # get_valid_calculation_mask requires it, so every result is limited to it.
    # Homes without AC are left out: the heat pump would add cooling they never
    # had (CLAUDE.md Limitation 11). Shared cooling is left out: it has no
    # replacement cost (SHARED_COOLING_EFFICIENCY in constants.py).
    # TODO (no-AC homes): to bring homes with no existing cooling into the
    # sample, this is the rule to change (see ALLOWED_TECHNOLOGIES in
    # constants.py). Their cooling energy must then stay as published (0 kWh
    # before the retrofit), not be blanked in STEP 5 below.
    is_applicable = pd.Series(True, index=df_enduse.index)
    for package_ids in applicable_bldg_ids:
        is_applicable &= df_enduse.index.isin(package_ids)
    df_enduse['include_sample'] = (
        df_enduse['include_heating'] & df_enduse['include_cooling'] & is_applicable)
    n_sample = int(df_enduse['include_sample'].sum())
    print(f"Study sample: {n_sample:,} rdu "
          f"({df_enduse.loc[df_enduse['include_sample'], 'weight'].sum():,.0f} homes)")
    # The GEA warning above counts the whole table; this counts the sample.
    n_sample_no_gea = int(
        (df_enduse['include_sample'] & df_enduse['gea_region'].isna()).sum())
    if n_sample_no_gea > 0:
        print(f"  {n_sample_no_gea:,} sample rdu have no GEA region; their "
              "climate values stay blank (CLAUDE.md Limitation 12)")

    # ===== STEP 5: Apply validation =====
    if verbose:
        print("\nApplying data validation (baseline only):")
    for category in VALID_CATEGORIES:
        # Get validation mask (baseline, so menu_mp = 0)
        valid_mask = get_valid_calculation_mask(df_enduse, category, menu_mp=0, verbose=verbose)
        
        # Apply masking to consumption columns
        columns_to_mask = get_all_possible_fuel_columns(category)
        columns_to_mask.append(f'baseline_{category}_consumption')
        # Component columns too (fans, heat-pump backup), so an invalid home's
        # components are masked the same as its primary energy.
        columns_to_mask.extend(
            col for _, col in get_consumption_component_columns(
                category, 0, columns=df_enduse.columns)
            if col not in columns_to_mask)

        # Apply masking
        for col in columns_to_mask:
            if col in df_enduse.columns:
                non_nan_before = df_enduse[col].notna().sum()
                df_enduse.loc[~valid_mask, col] = np.nan
                non_nan_after = df_enduse[col].notna().sum()
                
                masked_count = non_nan_before - non_nan_after
                if masked_count > 0 and verbose:
                    print(f"  {col}: Masked {masked_count} values")

    return df_enduse


# Test (a) limit. Both sides add up the same ResStock numbers, so 1 kWh is far
# above what number storage can explain and far below a part left out.
SAVINGS_MATCH_TOLERANCE_KWH = 1.0

# Share of a home's baseline use of a fuel by which ResStock's whole-home
# savings, less the end uses below, may differ from TARE's heating + cooling
# savings. Every occupied home in both releases is under 0.3%.
WHOLE_HOME_SAVINGS_TOLERANCE = 0.01

# End uses outside heating and cooling that ResStock itself changes when the
# equipment changes (2025.1 refrigerators by up to 1,100 kWh). Test (b) takes
# their savings out of the whole-home savings first.
INTERACTING_END_USES = ('hot_water', 'hot_water_solar_th', 'refrigerator')


def check_savings_against_resstock(
    tare_savings_by_fuel: Dict[str, pd.Series],
    df_mp: pd.DataFrame,
    sample_bldg_ids: pd.Index,
    menu_mp: int,
    release: str = RESSTOCK_RELEASE_THIS_RUN,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """Checks TARE's heating + cooling savings against ResStock's savings columns.

    Two tests for every study-sample home and fuel:
    (a) TARE's savings equal the sum of ResStock's savings columns for the
        heating and cooling parts of that fuel.
    (b) ResStock's whole-home savings, less the savings of the end uses in
        INTERACTING_END_USES, differ from TARE's savings by at most 1% of
        the home's baseline use of the fuel.
        A home that does not use the fuel must therefore show exactly zero.

    Args:
        tare_savings_by_fuel: TARE's annual heating + cooling savings (kWh,
            baseline minus retrofit), one Series per fuel spelled as in
            FUEL_MAPPING's values, indexed by bldg_id.
        df_mp: Raw ResStock upgrade frame, indexed by bldg_id.
        sample_bldg_ids: The study sample.
        menu_mp: Measure package number, for the printed title.
        release: ResStock release df_mp was loaded from.
        verbose: Whether to print the full table when every home passes. A
            failure always prints it.

    Returns:
        One row per fuel: rdu checked, the largest gap in each test, how many
        rdu fail, and '[OK]' or '[WARNING]'.

    Raises:
        ValueError: If any sample home fails either test or has a blank.
        KeyError: If df_mp lacks a savings column or a sample home, or
            tare_savings_by_fuel lacks a fuel.
    """
    # Step 1 -- sample rows, and every heating and cooling part ResStock reports
    df_sample_mp = df_mp.loc[sample_bldg_ids]
    part_columns = [
        (fuel, resstock_column)
        for category in ('heating', 'cooling')
        for fuel, _, resstock_column in find_enduse_columns(df_mp.columns, category)]

    # Step 2 -- both tests, one fuel at a time
    rows = []
    for resstock_fuel, fuel in RESSTOCK_FUEL_NAMES.items():
        tare_savings = tare_savings_by_fuel[fuel].loc[sample_bldg_ids]

        # Test (a): ResStock's savings for this fuel's heating and cooling parts
        part_savings_columns = [
            get_resstock_savings_column(resstock_column)
            for part_fuel, resstock_column in part_columns if part_fuel == fuel]
        part_gap = (
            tare_savings - df_sample_mp[part_savings_columns].sum(axis=1, skipna=False)
        ).abs()

        # Test (b): whole home less the end uses ResStock changes on its own.
        # Baseline use is the upgrade total plus its savings, so the upgrade
        # file alone is enough. Not every fuel has every end use (there is no
        # gas refrigerator), so only the columns in the file are read.
        total_column = resstock_col(release, f'{resstock_fuel}_total')
        total_savings_column = get_resstock_savings_column(total_column)
        total_savings = df_sample_mp[total_savings_column]
        interaction_columns = [
            total_savings_column.replace('.total.', f'.{end_use}.')
            for end_use in INTERACTING_END_USES]
        interaction_savings = df_sample_mp[[
            column for column in interaction_columns if column in df_mp.columns
        ]].sum(axis=1, skipna=False)
        baseline_use = df_sample_mp[total_column] + total_savings
        left_over = (total_savings - interaction_savings - tare_savings).abs()

        # Written as "not <=" so a blank counts as a failure.
        fails_parts = ~(part_gap <= SAVINGS_MATCH_TOLERANCE_KWH)
        fails_whole_home = ~(left_over <= WHOLE_HOME_SAVINGS_TOLERANCE * baseline_use)
        n_failed = int((fails_parts | fails_whole_home).sum())
        rows.append({
            'fuel': fuel,
            'rdu': len(sample_bldg_ids),
            'largest_part_gap_kwh': part_gap.max(),
            'largest_left_over_kwh': left_over.max(),
            'largest_left_over_pct': (100 * left_over / baseline_use).max(),
            'rdu_failed': n_failed,
            'status': '[OK]' if n_failed == 0 else '[WARNING]',
        })

    # Step 3 -- a failure always prints the table and stops the run; a pass
    # prints one line unless verbose
    df_check = pd.DataFrame(rows)
    df_failed = df_check[df_check['rdu_failed'] > 0]
    if verbose or not df_failed.empty:
        print(f"\nMP{menu_mp} heating + cooling savings vs ResStock's savings columns:")
        print(df_check.to_string(index=False))
    else:
        print(
            f"[OK] MP{menu_mp} heating + cooling savings match ResStock's "
            f"savings columns ({len(sample_bldg_ids):,} rdu, "
            f"{len(df_check)} fuels)")
    if not df_failed.empty:
        raise ValueError(
            f"MP{menu_mp} heating + cooling savings do not match ResStock's "
            "savings columns for "
            f"{dict(zip(df_failed['fuel'], df_failed['rdu_failed']))} (fuel: rdu)")
    return df_check


def df_enduse_compare(
    df_mp: pd.DataFrame,
    input_mp: str,
    menu_mp: int,
    df_baseline: pd.DataFrame,
    df_cooking_range: Optional[pd.DataFrame] = None,
    verbose: bool = VERBOSE,
    release: str = RESSTOCK_RELEASE_THIS_RUN
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
        df_cooking_range: Additional DataFrame for cooking range parameters and
            outputs. Only required when 'cooking' is in EQUIPMENT_SPECS (it is
            not, today); left as None otherwise.
        verbose: Whether to print detailed processing information.
        release: ResStock release df_mp and df_baseline were loaded from
            ('2022.1.1' or '2025.1'). Gates the MP3 ENERGY STAR override
            below, since 2025.1 Upgrade 03 will later also load as mp=3.
            Defaults to RESSTOCK_RELEASE_THIS_RUN, so existing 2022.1.1
            callers that don't pass this argument are unaffected.

    Returns:
        A merged DataFrame (df_compare) that includes relevant columns for
        baseline and measure packages comparison.

    Raises:
        ValueError: If 'cooking' is in EQUIPMENT_SPECS but df_cooking_range
            is None.
    """
    # Updated to handle different enduses based on EQUIPMENT_SPECS.
    # - Rest of codebase updated so only initial columns created for cooling and replacement cost calculations performed
    # - This allows for a scenario where only heating is replaced AND one where heating and cooling systems are both replace with HP
    # - Resolves the excessive data columns and double counting with $8000 rebate. No longer need CDD projections.
    VALID_CATEGORIES = list(EQUIPMENT_SPECS.keys())

    # Fail fast: cooking is inactive today (EQUIPMENT_SPECS has only heating
    # and cooling), but if it is ever turned on, a caller that forgot
    # df_cooking_range should get a clear error here, not a KeyError deep in
    # STEP 2 or STEP 3 below.
    if 'cooking' in VALID_CATEGORIES and df_cooking_range is None:
        raise ValueError(
            "df_cooking_range is required when 'cooking' is in "
            "EQUIPMENT_SPECS, but None was passed."
        )

    # ===== STEP 1: Initialize with common columns (always present) =====
    df_compare = pd.DataFrame({
        'hvac_has_ducts': df_mp['in.hvac_has_ducts']
    })
    
    # ===== STEP 2: Conditionally add category-specific metadata columns =====
    
    # HEATING - only if in scope
    if 'heating' in VALID_CATEGORIES:
        df_compare['hvac_heating_type_and_fuel'] = df_mp[resstock_col(release, 'heating_type_and_fuel')]
        df_compare['hvac_heating_efficiency'] = df_mp[resstock_col(release, 'heating_efficiency')]
        # The heat pump's own backup-coil (or, for mp=5, backup furnace)
        # capacity for THIS measure package. Carried as a plain pass-through;
        # no cost is attached to it here (a later phase will price the mp=5
        # backup furnace from it). This is the retrofit heat pump's capacity,
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
        df_compare['size_heat_pump_backup_primary_k_btu_h'] = df_mp[resstock_col(release, 'size_heat_pump_backup')]
        df_compare['size_heating_system_primary_k_btu_h'] = df_mp[resstock_col(release, 'size_heating_primary')]
        # df_compare['size_heating_secondary_k_btu_h'] = df_mp['out.params.size_heating_system_secondary_k_btu_h']
        df_compare['upgrade_hvac_heating_efficiency'] = df_mp[resstock_col(release, 'upgrade_heating_efficiency')]

        # ENERGY STAR override (MP3 only). MP3's modeled heat pump is
        # SEER 15 / 9.0 HSPF -- just below the ENERGY STAR minimum
        # (>= 16.0 SEER1 / >= 9.5 HSPF1) required for the federal heat-pump
        # rebate. To model MP3 as a rebate-eligible ENERGY STAR install, rewrite
        # its upgrade spec to that floor. Only the heating SEER value feeds the
        # REMDB v4 upgrade cost (pm2 = SEER1), so this raises MP3 capital cost
        # modestly; HSPF is bumped for spec accuracy but has no cost lever in
        # this model. Energy use is unchanged (it comes from the ResStock
        # simulation, not from this string).
        # Release guard: this override is tuned to the exact 2022.1.1 MP3
        # string ("...SEER 15, 9.0 HSPF..."). 2025.1 Upgrade 03 will later
        # load under the same mp=3 number, but with a different string
        # format; without this guard the override would fire on that
        # package too and silently corrupt it (the same way it would
        # corrupt the dual-fuel string -- see parse_dual_fuel_heating_efficiency).
        if release == '2022.1.1' and menu_mp == 3:
            df_compare['upgrade_hvac_heating_efficiency'] = (
                df_compare['upgrade_hvac_heating_efficiency']
                .str.replace('SEER 15', 'SEER 16', regex=False)
                .str.replace('9.0 HSPF', '9.5 HSPF', regex=False)
            )
    
    # COOLING - only if in scope
    if 'cooling' in VALID_CATEGORIES:
        df_compare['hvac_cooling_type'] = df_mp[resstock_col(release, 'cooling_type')]
        df_compare['hvac_cooling_efficiency'] = df_mp[resstock_col(release, 'cooling_efficiency')]
        # Same retrofit heat-pump capacity as size_heating_system_primary_k_btu_h
        # above -- one heat pump serves both loads, so heating and cooling
        # capacity are identical for every home. Not the baseline air
        # conditioner's size. Only the heat pump's own upgrade cost is priced
        # off this column -- the cooling replacement cost (the avoided cost
        # of replacing the OLD air conditioner) is priced off
        # base_size_cooling_system_primary_k_btu_h instead, added in
        # df_enduse_refactored. See docs/SESSION_CHANGELOG_2026-08-20.md.
        df_compare['size_cooling_system_primary_k_btu_h'] = df_mp[resstock_col(release, 'size_cooling_primary')]
        df_compare['upgrade_hvac_cooling_efficiency'] = df_mp[resstock_col(release, 'upgrade_cooling_efficiency')]

        # ENERGY STAR override (MP3 only), parallel to the heating override above
        # so the two upgrade-spec columns stay consistent. ResStock records the
        # MP3 cooling upgrade as the bare "Heat Pump" label (no SEER encoded), so
        # this replace is a no-op today; it keeps the columns in sync if a future
        # data vintage carries a numeric cooling spec.
        # Same release guard as the heating override above.
        if release == '2022.1.1' and menu_mp == 3:
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
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp[resstock_col(release, 'heating_electricity')]

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
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp[resstock_col(release, 'heating_electricity')]

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
                df_compare[f'mp{menu_mp}_heating_consumption'] = df_mp[resstock_col(release, 'heating_electricity')]

        elif category == 'cooling':
            df_compare[f'mp{menu_mp}_cooling_consumption'] = df_mp[resstock_col(release, 'cooling_electricity')]

        elif category == 'waterHeating':
            df_compare[f'mp{menu_mp}_waterHeating_consumption'] = df_mp['out.electricity.hot_water.energy_consumption.kwh']

        elif category == 'clothesDrying':
            df_compare[f'mp{menu_mp}_clothesDrying_consumption'] = df_mp['out.electricity.clothes_dryer.energy_consumption.kwh']

        elif category == 'cooking':
            df_compare[f'mp{menu_mp}_cooking_consumption'] = df_cooking_range['out.electricity.range_oven.energy_consumption.kwh']

    # ===== STEP 3a: Separately reported heating/cooling components =====
    # Retrofit side of the component columns df_enduse_refactored adds for the
    # baseline, read from the upgrade file. For a dual-fuel retrofit (mp=5) the
    # backup furnace's natural gas lands here -- most of that package's heating
    # energy. In 2022.1.1 the fuel-oil, propane, and gas backup columns exist
    # but are always zero (no 2022.1.1 package has a fossil backup).
    for category in ('heating', 'cooling'):
        if category not in VALID_CATEGORIES:
            continue
        for fuel, part, resstock_column in find_enduse_columns(
                df_mp.columns, category):
            if part == 'primary_system':
                continue
            col = f'mp{menu_mp}_{fuel}_{category}_{part}_consumption'
            df_compare[col] = df_mp[resstock_column]

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
    # 2022.1.1's peak columns are conditioned on the end use running that hour;
    # 2025.1's are conditioned on the calendar season instead, so the two
    # releases' zero counts differ even though both represent "the peak."
    df_compare[f'mp{menu_mp}_peak_electricity_cooling_kw'] = (
        df_mp[resstock_col(release, 'peak_electricity_cooling')]
    )
    df_compare[f'mp{menu_mp}_peak_electricity_heating_kw'] = (
        df_mp[resstock_col(release, 'peak_electricity_heating')]
    )
    df_compare[f'mp{menu_mp}_peak_electricity_cooling_kw_savings'] = (
        df_mp[resstock_col(release, 'peak_electricity_cooling_savings')]
    )
    df_compare[f'mp{menu_mp}_peak_electricity_heating_kw_savings'] = (
        df_mp[resstock_col(release, 'peak_electricity_heating_savings')]
    )
    df_compare[f'mp{menu_mp}_peak_load_cooling_kbtu_hr'] = (
        df_mp[resstock_col(release, 'peak_load_cooling')]
    )
    df_compare[f'mp{menu_mp}_peak_load_heating_kbtu_hr'] = (
        df_mp[resstock_col(release, 'peak_load_heating')]
    )
    df_compare[f'mp{menu_mp}_peak_load_cooling_kbtu_hr_savings'] = (
        df_mp[resstock_col(release, 'peak_load_cooling_savings')]
    )
    df_compare[f'mp{menu_mp}_peak_load_heating_kbtu_hr_savings'] = (
        df_mp[resstock_col(release, 'peak_load_heating_savings')]
    )
    df_compare[f'mp{menu_mp}_total_electricity_consumption'] = (
        df_mp[resstock_col(release, 'electricity_total')]
    )
    # Whole-home site energy after the retrofit, all fuels (the retrofit side
    # of baseline_total_site_consumption). Used only for the savings check
    # column in STEP 7.
    df_compare[f'mp{menu_mp}_total_site_consumption'] = (
        df_mp[resstock_col(release, 'site_energy_total')]
    )

    # ===== STEP 3c: Post-upgrade electrical panel constraint flags (2025.1) =====
    # Reporting-only pass-through: whether the new equipment runs into an
    # existing panel's capacity or breaker-space limit under the 2023 NEC
    # existing-dwelling load calculation. No panel upgrade cost is modeled
    # from these flags (see CLAUDE.md's capital-cost scope limitation); they
    # are carried so a later phase can decide whether to price one. Only
    # published starting ResStock 2025.1, so this block is release-gated.
    if release == '2025.1':
        df_compare[f'mp{menu_mp}_panel_constraint_overall'] = (
            df_mp[resstock_col(release, 'panel_constraint_overall')])
        df_compare[f'mp{menu_mp}_panel_constraint_capacity'] = (
            df_mp[resstock_col(release, 'panel_constraint_capacity')])
        df_compare[f'mp{menu_mp}_panel_constraint_breaker_space'] = (
            df_mp[resstock_col(release, 'panel_constraint_breaker_space')])

    # ===== STEP 3d: ResStock's own applicability flag (2025.1) =====
    # Whether ResStock actually ran this upgrade on this home -- distinct from
    # any of TARE's own fuel/technology masks. Coerced to bool defensively on
    # read: the guide notes this column's dtype was a string in some upgrade
    # parquets before a 2025-07-08 ResStock fix, even though it reads as a
    # real bool in the published 2025.1 files TARE loads today. Carried as a
    # plain pass-through column here; Phase 3's masking funnel is the first
    # place that filters on it. Only published starting ResStock 2025.1, so
    # this block is release-gated like the panel columns above.
    if release == '2025.1':
        df_compare[f'mp{menu_mp}_resstock_applicable'] = (
            df_mp[resstock_col(release, 'upgrade_applicable')].astype(bool))

    # ===== STEP 4: Merge with baseline DataFrame =====
    df_compare = pd.merge(df_baseline, df_compare, how='inner', left_index=True, right_index=True)
    
    # ===== STEP 5: Ensure validation flags are preserved =====
    validation_flags = [col for col in df_baseline.columns 
                       if col.startswith('valid_') or col.startswith('include_')]
    
    for flag in validation_flags:
        if flag in df_baseline.columns and flag not in df_compare.columns:
            df_compare[flag] = df_baseline[flag]
    
    # ===== STEP 6: Apply combined validation (data quality + retrofit status) =====
    if verbose:
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

        # Component columns (fans, heat-pump backup), baseline and retrofit
        for mp_value in (0, menu_mp):
            category_cols.extend(
                col for _, col in get_consumption_component_columns(
                    category, mp_value, columns=df_compare.columns)
                if col in df_compare.columns and col not in category_cols)

        # Apply masking
        for col in category_cols:
            non_nan_before = df_compare[col].notna().sum()
            df_compare.loc[~valid_mask, col] = np.nan
            non_nan_after = df_compare[col].notna().sum()
            
            masked_count = non_nan_before - non_nan_after
            if masked_count > 0 and verbose:
                print(f"  {col}: Masked {masked_count} values")

    # ===== STEP 7: Whole-home modeled savings fraction (HOMES rebate tiers) =====
    # Numerator: annual drop in heating + cooling energy, counting every
    # component on both sides (primary energy, fans and pumps, heat-pump
    # backup) and every fuel. Earlier versions counted primary energy only,
    # which left out backup heat and fans and overstated savings. Read at
    # ANCHOR_YEAR, where every degree-day factor is 1.0, so these are
    # ResStock's own annual values. Denominator: whole-home baseline site
    # energy, all fuels.
    # Annual use by fuel, before (mp 0) and after, read once: the fraction
    # below adds the fuels together and the check in STEP 8 keeps them apart.
    use_by_fuel = {
        (category, mp): get_degree_day_adjusted_consumption_by_fuel(
            df_compare, category, ANCHOR_YEAR, mp)
        for category in ('heating', 'cooling') if category in VALID_CATEGORIES
        for mp in (0, menu_mp)}

    def _annual_use(category: str, mp: int) -> pd.Series:
        """All-fuel annual use for one category; NaN for homes left out."""
        return pd.concat(
            use_by_fuel[(category, mp)].values(), axis=1).sum(axis=1, min_count=1)

    # Savings stay NaN for homes outside the study sample.
    # Annual heating use before and after, every fuel and component (a gas
    # furnace's gas plus its blower; a heat pump's electricity plus backup and
    # fans). Kept as columns for the furnace-vs-heat-pump comparison figure.
    df_compare['baseline_heating_annual_consumption_kwh'] = _annual_use('heating', 0)
    df_compare[f'mp{menu_mp}_heating_annual_consumption_kwh'] = (
        _annual_use('heating', menu_mp))
    heating_savings = (df_compare['baseline_heating_annual_consumption_kwh']
                       - df_compare[f'mp{menu_mp}_heating_annual_consumption_kwh'])
    if 'cooling' in VALID_CATEGORIES:
        cooling_savings = (_annual_use('cooling', 0)
                           - _annual_use('cooling', menu_mp))
    else:
        cooling_savings = 0.0

    hvac_savings_col = f'mp{menu_mp}_hvac_energy_savings_kwh'
    savings_frac_col = f'mp{menu_mp}_modeled_savings_frac'
    df_compare[hvac_savings_col] = (
        heating_savings + cooling_savings).astype('float64')
    df_compare[savings_frac_col] = (
        df_compare[hvac_savings_col]
        / df_compare['baseline_total_site_consumption']
    ).astype('float64')

    # Check only, used in no calculation: ResStock's own whole-home change in
    # site energy. It differs from the HVAC savings above mostly because
    # ResStock's hot-water use also changes with the heating and cooling
    # equipment. STEP 8 tests what is left, per home and fuel.
    whole_home_col = f'mp{menu_mp}_whole_home_energy_savings_kwh'
    df_compare[whole_home_col] = (
        df_compare['baseline_total_site_consumption']
        - df_compare[f'mp{menu_mp}_total_site_consumption']
    ).where(heating_savings.notna()).astype('float64')
    df_compare[f'{savings_frac_col}_whole_home'] = (
        df_compare[whole_home_col]
        / df_compare['baseline_total_site_consumption']
    ).astype('float64')

    # ===== STEP 8: Check the savings against ResStock's own savings columns =====
    # The same baseline-minus-retrofit savings as STEP 7, kept apart by fuel.
    tare_savings_by_fuel = {}
    for (category, mp), by_fuel in use_by_fuel.items():
        sign = 1.0 if mp == 0 else -1.0
        for fuel, annual_use in by_fuel.items():
            tare_savings_by_fuel[fuel] = (
                tare_savings_by_fuel.get(fuel, 0.0) + sign * annual_use)
    check_savings_against_resstock(
        tare_savings_by_fuel, df_mp,
        df_compare.index[df_compare['include_sample'].astype(bool)],
        menu_mp, release, verbose=verbose)

    return df_compare
