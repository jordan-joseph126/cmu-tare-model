"""
Shared data loading functions and constants for the adoption KPI modules.

Provides EUSS baseline/upgrade loading, unit conversion constants,
column name constants, file path constants, and state name lookups
used across spark_gap.py and thermal_cop.py.

Every ResStock column name here is looked up for this run's release
(RESSTOCK_RELEASE_THIS_RUN) in the column map (resstock_col), and the two
loaders read the same release's files with the same scope filters as the
model run, so the county tables describe the same homes as the TARE results.
No ResStock column name is typed into this module: 2025.1 renamed most of its
energy columns, and a typed 2022.1.1 name would fail on a 2025.1 frame.

Location: cmu_tare_model/adoption_kpis/data_loading.py
"""

import os
from typing import List

import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    RESSTOCK_RELEASE_AND_MP,
    RESSTOCK_RELEASE_THIS_RUN,
    VERBOSE,
)
from cmu_tare_model.utils.calculation_utils import get_resstock_savings_column
from cmu_tare_model.utils.resstock_schema import RESSTOCK_COLUMN_MAP, resstock_col


# ============================================================================
# UNIT CONVERSION CONSTANTS (Source: EIA)
# ============================================================================

BTU_PER_CF_NATURAL_GAS: int = 1036
"""
BTU per cubic foot of natural gas (EIA US 2025 average).
https://www.eia.gov/dnav/ng/ng_cons_heat_a_epg0_vgth_btucf_a.htm
"""

BTU_PER_KWH: int = 3412
"""BTU per kilowatt-hour (by definition)."""

KWH_PER_MMBTU: float = 293.07107
"""1 MMBTU = 293.07107 kWh."""

KBTU_PER_KWH: float = 3.412
"""1 kWh = 3.412 kBtu."""

# Derived: natural gas $/1000cf -> $/kWh
NG_CONVERSION_FACTOR: float = BTU_PER_KWH / (1000 * BTU_PER_CF_NATURAL_GAS)
"""Multiply ng_price_per_1000cf by this to get $/kWh."""


# ============================================================================
# FILE PATHS
# ============================================================================

# ResStock 2022.1.1 CSV folder. Kept for reference only: both loaders below
# read through load_and_filter_upgrade, which knows each release's own files.
EUSS_DATA_DIR: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "euss_data",
    "resstock_amy2018_release_1.1", "national", "csv"
)

FUEL_PRICES_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "fuel_prices",
    "fuel_prices_nominal_2015_2024.csv"
)

# --- State geometry vintage and product -------------------------------------
# Kept equal to the county geometry (below) on purpose. When the county layer
# was TIGER/Line, the state overlay could also be TIGER without visible harm.
# Now that counties are a shoreline-clipped, generalized cartographic boundary
# file, a full-resolution TIGER state layer would draw state outlines out past
# the county fill into open water, and interior state borders that no longer
# coincide with the generalized county edges. Matching product, vintage, and
# scale keeps the two layers aligned.
STATE_GEOMETRY_PRODUCT: str = "cb"
"""Census product for the state overlay: 'tl' or 'cb'. Match the county layer."""

STATE_GEOMETRY_VINTAGE: int = 2021
"""State geometry vintage year. Kept equal to the county geometry vintage."""

STATE_GEOMETRY_SCALE: str = "500k"
"""Cartographic boundary scale ('500k', '5m', '20m'). Ignored when product='tl'."""


def _state_shapefile_stem(product: str, vintage: int, scale: str) -> str:
    """Build the Census state shapefile stem for a product and vintage.

    Mirrors :func:`_county_shapefile_stem` for the state boundary layer. The
    two products name the folder and .shp differently: 'tl_2025_us_state' vs
    'cb_2021_us_state_500k'.

    Args:
        product: 'tl' (TIGER/Line) or 'cb' (cartographic boundary).
        vintage: Geometry vintage year.
        scale: Cartographic boundary scale; ignored when product is 'tl'.

    Returns:
        Shapefile stem without extension.

    Raises:
        ValueError: If product is not 'tl' or 'cb'.
    """
    if product == "cb":
        return f"cb_{vintage}_us_state_{scale}"
    if product == "tl":
        return f"tl_{vintage}_us_state"
    raise ValueError(f"Unknown state geometry product: {product!r}. Use 'tl' or 'cb'.")


_STATE_STEM: str = _state_shapefile_stem(
    STATE_GEOMETRY_PRODUCT, STATE_GEOMETRY_VINTAGE, STATE_GEOMETRY_SCALE
)

SHAPEFILE_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "shapefiles",
    _STATE_STEM, f"{_STATE_STEM}.shp"
)
# State borders for the county-map overlay. Versioned alongside the county
# geometry (same product, vintage, and scale) so the two layers' generalized
# shorelines and interior borders coincide -- this is for generalization
# consistency, not the CT change (the CT county-to-planning-region switch does
# not alter CT's state polygon). Site 3 reads STUSPS, which cb state files carry.

# --- County geometry vintage and product ------------------------------------
# Change the three constants below to repoint the county layer; nothing else
# encodes the vintage. Current: product "cb", vintage 2021, scale "500k"
# (cb_2021_us_county_500k) -- the newest cartographic boundary vintage that
# still carries Connecticut's eight pre-2023 counties.
#
# Vintage must match ResStock 2022.1.1, which uses 2010-vintage geography.
# Connecticut replaced its eight counties (FIPS 09001-09015) with nine planning
# regions (09110-09190); a sweep of the 2020-2025 vintages confirmed the new
# codes first appear in 2022 and that this is the ONLY change to the national
# county universe across that span. Any vintage from 2022 on drops all of
# Connecticut from the county join. Do not advance it without re-running the
# sweep.
#
# Product cb_* over tl_*: cartographic boundary polygons are clipped to the
# shoreline (TIGER includes territorial water, giving coastal states spurious
# offshore lobes) and are ~1/10 the size; 500k is the most detailed cb scale.
# cb_* polygons are generalized, which is safe ONLY because no numeric quantity
# derives from county geometry here -- polygons are fill shapes keyed on GEOID
# (no area, density-per-area, or centroid-based spatial join).
#
# Dropped by the tl_* -> cb_* swap; none are used in this codebase:
#   INTPTLAT/INTPTLON       internal point, for label placement and bubble
#                           overlays. Unused: national maps carry no labels.
#   CBSAFP/CSAFP/METDIVFP   metro-area codes, for CBSA aggregation only.
#   CLASSFP/FUNCSTAT/MTFCC  Census classification bookkeeping.
#   GEOIDFQ                 not a loss -- cb_* carries it as AFFGEOID.
# Gained: AFFGEOID, STUSPS, STATE_NAME. Both products are EPSG:4269, so no
# reprojection is needed and the SHAPEFILE_PATH state overlay stays aligned.
#
# Download: https://www.census.gov/cgi-bin/geo/shapefiles/index.php
# Required columns: GEOID (5-digit FIPS), STATEFP (2-digit state FIPS).

COUNTY_GEOMETRY_PRODUCT: str = "cb"
"""Census product: 'tl' (TIGER/Line) or 'cb' (cartographic boundary)."""

COUNTY_GEOMETRY_VINTAGE: int = 2021
"""Census geometry vintage year. Must predate 2022 to retain CT counties."""

COUNTY_GEOMETRY_SCALE: str = "500k"
"""Cartographic boundary scale ('500k', '5m', '20m'). Ignored when product='tl'."""


def _county_shapefile_stem(product: str, vintage: int, scale: str) -> str:
    """Build the Census county shapefile stem for a product and vintage.

    Census names the folder and the .shp identically, and the two products use
    different conventions: 'tl_2025_us_county' vs 'cb_2021_us_county_500k'.

    Args:
        product: 'tl' (TIGER/Line) or 'cb' (cartographic boundary).
        vintage: Geometry vintage year.
        scale: Cartographic boundary scale; ignored when product is 'tl'.

    Returns:
        Shapefile stem without extension.

    Raises:
        ValueError: If product is not 'tl' or 'cb'.
    """
    if product == "cb":
        return f"cb_{vintage}_us_county_{scale}"
    if product == "tl":
        return f"tl_{vintage}_us_county"
    raise ValueError(f"Unknown county geometry product: {product!r}. Use 'tl' or 'cb'.")


_COUNTY_STEM: str = _county_shapefile_stem(
    COUNTY_GEOMETRY_PRODUCT, COUNTY_GEOMETRY_VINTAGE, COUNTY_GEOMETRY_SCALE
)

COUNTY_SHAPEFILE_PATH: str = os.path.join(
    PROJECT_ROOT, "cmu_tare_model", "data", "shapefiles",
    _COUNTY_STEM, f"{_COUNTY_STEM}.shp"
)


# ============================================================================
# COLUMN NAME CONSTANTS
# ============================================================================
# One release per run, so these are fixed when the module is imported. The
# release itself is fixed before any import (see RESSTOCK_RELEASE_THIS_RUN).

_RELEASE: str = RESSTOCK_RELEASE_THIS_RUN


def _release_columns(logical_names: List[str]) -> List[str]:
    """Physical names, in order, of the logical names this release publishes.

    A part one release does not publish (2025.1 adds the heat-pump backup's
    own fans) is left out rather than raising, so one list serves both
    releases.

    Args:
        logical_names: Logical names from the column map.

    Returns:
        The physical column names for _RELEASE.
    """
    return [
        resstock_col(_RELEASE, logical_name)
        for logical_name in logical_names
        if logical_name in RESSTOCK_COLUMN_MAP[_RELEASE]]


DWELLING_UNIT_WEIGHT: str = resstock_col(_RELEASE, "dwelling_unit_weight")
"""EUSS survey weight column (applies to all dwelling-unit counts)."""

GAS_FUEL_COL: str = resstock_col(_RELEASE, "heating_natural_gas")
"""EUSS column for natural gas heating energy consumption."""

HEATING_LOAD_COL: str = resstock_col(_RELEASE, "heating_load_delivered")
"""EUSS column for heating load delivered to the space (kBtu)."""

HEATING_ELEC_COL: str = resstock_col(_RELEASE, "heating_electricity")
"""EUSS column for heating electricity (heat pump or resistance heat), not
counting backup heat or fans and pumps."""

HP_BACKUP_ELEC_COL: str = resstock_col(_RELEASE, "heating_hp_backup_electricity")
"""EUSS column for heat-pump backup (resistance) electricity."""

HP_FANS_PUMPS_COL: str = resstock_col(_RELEASE, "heating_fans_pumps")
"""EUSS column for fan and pump electricity. Always included in COP denominator."""

COOLING_ELEC_COL: str = resstock_col(_RELEASE, "cooling_electricity")
"""EUSS column for cooling electricity, not counting its fans and pumps."""

COOLING_FANS_PUMPS_COL: str = resstock_col(_RELEASE, "cooling_fans_pumps")
"""EUSS column for cooling fan and pump electricity."""

HOT_WATER_ELEC_COL: str = resstock_col(_RELEASE, "hot_water_electricity")
"""EUSS column for hot-water electricity."""

ELEC_TOTAL_COL: str = resstock_col(_RELEASE, "electricity_total")
"""EUSS column for total residential ELECTRICITY (kWh). Includes all electric
end uses. Use this for demand change calculations -- do NOT use the heating-only
column, and do NOT substitute the site energy total (SITE_ENERGY_TOTAL_COL),
which is the all-fuel site energy (gas/oil/propane in kWh-equivalent), not
electricity."""

SITE_ENERGY_TOTAL_COL: str = resstock_col(_RELEASE, "site_energy_total")
"""EUSS column for whole-home site energy, ALL fuels (kWh; gas/oil/propane in
kWh-equivalent). Use this for the site-energy change only -- do NOT use it for
electricity demand, which is ELEC_TOTAL_COL."""

# ResStock's own savings columns (baseline minus upgrade, positive = use fell).
# Only the upgrade files carry them. An end use's change is minus its savings.
ELEC_TOTAL_SAVINGS_COL: str = get_resstock_savings_column(ELEC_TOTAL_COL)
"""Whole-home electricity savings."""

SITE_ENERGY_TOTAL_SAVINGS_COL: str = get_resstock_savings_column(SITE_ENERGY_TOTAL_COL)
"""Whole-home site energy savings, all fuels."""

HEATING_ELEC_SAVINGS_COLS: List[str] = [
    get_resstock_savings_column(energy_col)
    for energy_col in _release_columns([
        "heating_electricity", "heating_hp_backup_electricity",
        "heating_hp_backup_fans", "heating_fans_pumps"])]
"""Heating electricity savings: heating, backup heat, the backup's own fans
(2025.1 only), and fans and pumps -- every electric part of heating, the same
parts TARE counts in a home's heating energy."""

COOLING_ELEC_SAVINGS_COLS: List[str] = [
    get_resstock_savings_column(energy_col)
    for energy_col in (COOLING_ELEC_COL, COOLING_FANS_PUMPS_COL)]
"""Cooling electricity savings: cooling, and its fans and pumps."""

HOT_WATER_ELEC_SAVINGS_COL: str = get_resstock_savings_column(HOT_WATER_ELEC_COL)
"""Hot-water electricity savings."""

CLIMATE_ZONE_COL: str = resstock_col(_RELEASE, "climate_zone_iecc")
"""EUSS column for ASHRAE/IECC 2004 climate zone."""

COUNTY_COL: str = resstock_col(_RELEASE, "county")
"""EUSS column for county GISJOIN code (e.g., 'G4200030')."""

STATE_COL: str = resstock_col(_RELEASE, "state")
"""EUSS column for the two-letter state code."""

HEATING_FUEL_COL: str = resstock_col(_RELEASE, "heating_fuel")
"""EUSS column for the home's heating fuel before the retrofit."""

HEATING_FUEL_COLS: List[str] = _release_columns([
    "heating_electricity", "heating_natural_gas",
    "heating_fuel_oil", "heating_propane"])
"""All EUSS heating energy consumption columns (kWh)."""

FUEL_PRICE_MAP: dict[str, str] = {
    "electricity": "elec_price_kwh",
    "naturalGas": "gas_price_kwh",
}
"""Mapping from EIA fuel-type string to price column name."""

# Column subsets for CSV loading
BASELINE_USECOLS: List[str] = _release_columns([
    "building_id", "state", "vacancy_status", "building_type",
    "heating_fuel", "heating_type_and_fuel", "heating_efficiency",
    "climate_zone_iecc", "county", "dwelling_unit_weight",
]) + HEATING_FUEL_COLS + [
    HEATING_LOAD_COL, HP_BACKUP_ELEC_COL, HP_FANS_PUMPS_COL, ELEC_TOTAL_COL]

UPGRADE_USECOLS: List[str] = BASELINE_USECOLS + [
    resstock_col(_RELEASE, "upgrade_applicable")]


# ============================================================================
# STATE NAME LOOKUP
# ============================================================================

STATE_NAMES: dict[str, str] = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska",
    "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas",
    "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}
"""Mapping from 2-letter state abbreviation to full state name."""


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

# The baseline file name load_euss_baseline has always taken. Kept so existing
# calls still work; the file read is now chosen by the release.
DEFAULT_BASELINE_FILENAME: str = "baseline_metadata_and_annual_results.csv"


def mp_to_upgrade(mp_num: int) -> str:
    """Convert a measure package number to its EUSS upgrade identifier string.

    ResStock 2022.1.1 zero-pads the upgrade number ('upgrade04'); 2025.1
    does not ('upgrade5'), matching each release's own file names.

    Args:
        mp_num: Measure package number (e.g., 3 or 4).

    Returns:
        EUSS upgrade identifier string (e.g., ``'upgrade03'``, ``'upgrade04'``
        for 2022.1.1; ``'upgrade5'`` for 2025.1).
    """
    if _RELEASE == "2022.1.1":
        return f"upgrade{mp_num:02d}"
    return f"upgrade{mp_num}"


def _load_scope_filtered(menu_mp: int, verbose: bool) -> pd.DataFrame:
    """Loads one package's file with the model run's scope filters.

    Uses load_and_filter_upgrade, the loader the model run itself uses, so
    the KPI tables count exactly the homes the run counts: ResStock's
    applicability flag first, then occupied, single-family, and Alaska and
    Hawaii left out. On 2025.1 only the columns the pipeline uses are read.

    Args:
        menu_mp: Measure package number (0 for the baseline).
        verbose: Whether to print the filter funnel.

    Returns:
        The filtered frame, indexed by bldg_id.
    """
    # Imported here rather than at the top so modules that need only this
    # file's constants (the map modules, for example) do not also load the
    # ResStock processing module and the data files it reads on import.
    from cmu_tare_model.energy_consumption_and_metadata.process_euss_data import (
        load_and_filter_upgrade,
    )
    df_filtered, df_funnel = load_and_filter_upgrade(
        menu_mp=menu_mp, verbose=False, release=_RELEASE)
    if verbose:
        print(df_funnel.to_string(index=False))
    return df_filtered


def load_euss_baseline(
    filename: str = DEFAULT_BASELINE_FILENAME,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """Load this run's ResStock baseline with the model run's scope filters.

    Reads the baseline of RESSTOCK_RELEASE_THIS_RUN (the 2022.1.1 CSV or the
    2025.1 parquet file) and applies, in order: ResStock's applicability
    flag (every baseline row is applicable), occupied homes, housing type in
    ``ALLOWED_HOUSING_TYPES``, and Alaska and Hawaii left out. On 2022.1.1
    the rows kept are the same as before; the applicability and Alaska and
    Hawaii filters remove nothing there.

    Args:
        filename: Kept so existing calls still work. Only the default is
            accepted: the file is chosen by the release.
        verbose: Whether to print the filter funnel.

    Returns:
        DataFrame indexed by ``bldg_id``, restricted to occupied SF homes.
        On 2022.1.1 every EUSS column is kept; on 2025.1 the columns the
        pipeline uses (select_resstock_2025_1_columns in process_euss_data.py).

    Raises:
        ValueError: If filename is not the default.
        FileNotFoundError: If the release's baseline file is not on disk.
    """
    if filename != DEFAULT_BASELINE_FILENAME:
        raise ValueError(
            f"load_euss_baseline reads the baseline file of ResStock "
            f"{_RELEASE}; filename={filename!r} cannot choose another file")
    return _load_scope_filtered(0, verbose)


def load_euss_upgrade(
    upgrade_name: str,
    verbose: bool = VERBOSE,
) -> pd.DataFrame:
    """Load one of this run's ResStock upgrade files with the run's scope filters.

    Applies the same filters as the model run, in the same order: ResStock's
    applicability flag first, then occupied homes, housing type in
    ``ALLOWED_HOUSING_TYPES``, and Alaska and Hawaii left out.

    Args:
        upgrade_name: EUSS upgrade identifier (e.g., ``'upgrade04'`` on
            2022.1.1, ``'upgrade5'`` on 2025.1). Use :func:`mp_to_upgrade` to
            convert a measure package number.
        verbose: Whether to print the filter funnel.

    Returns:
        DataFrame indexed by ``bldg_id``, restricted to applicable occupied
        SF homes. On 2025.1, the columns the pipeline uses.

    Raises:
        ValueError: If upgrade_name is not a package of this run's release.
        FileNotFoundError: If the upgrade file is not on disk.
    """
    upgrade_names = {
        mp_to_upgrade(menu_mp): menu_mp
        for menu_mp in RESSTOCK_RELEASE_AND_MP[_RELEASE] if menu_mp != 0}
    if upgrade_name not in upgrade_names:
        raise ValueError(
            f"{upgrade_name!r} is not a measure package of ResStock "
            f"{_RELEASE}; expected one of {sorted(upgrade_names)}")
    return _load_scope_filtered(upgrade_names[upgrade_name], verbose)
