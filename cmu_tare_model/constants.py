# Constants for the TARE Model (Updated for TRANE Analysis)
# Adjust or remove them as needed, or move them to a separate config file.
# UPDATE: 
# - Updated to handle different enduses based on EQUIPMENT_SPECS.
# - Rest of codebase updated so only initial columns created for cooling and replacement cost calculations performed
# - This allows for a scenario where only heating is replaced AND one where heating and cooling systems are both replace with HP
# - Resolves the excessive data columns and double counting with $8000 rebate. No longer need CDD projections.

# =============================================================
# TARE MODEL RUN CONFIGURATION
# =============================================================
# Configuration
VERBOSE = False
PRINT_DEBUG = False
PRINT_VERBOSE_DATAFRAMES = False
FIGURE_DPI = 600
MAP_TITLE_FONT_SIZE = 18
MAP_TITLE_PAD = 10
MAP_CBAR_LABEL_FONT_SIZE = 16
MAP_CBAR_TICK_LABEL_SIZE = 16
MAP_LEGEND_FONT_SIZE = 16

FIGURE_TITLE_FONT_SIZE = 18

# ALLOWED_HOUSING_TYPES = ['Single-Family Attached', 'Single-Family Detached', 'Mobile Home', 'Multi-Family with 2 - 4 Units']
ALLOWED_HOUSING_TYPES = ['Single-Family Attached', 'Single-Family Detached']

# Excludes HP Tech for Space/Water Heating and Clothes Drying. Also excludes electric resistance cooking and induction cooking.
# enumeration_dictionary.tsv provides additional details on the allowed technologies for each equipment category.
# Trane GC focuses on heating and cooling. Cooling category is used for initial replacement cost estimates.
ALLOWED_TECHNOLOGIES = {
    # in.hvac_heating_type_and_fuel exclude existing heat pump options
    'heating': [
        'Electricity Baseboard', 'Electricity Electric Boiler', 'Electricity Electric Furnace', 
        # 'Electricity ASHP',
        'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
        'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
        'Propane Fuel Boiler', 'Propane Fuel Furnace'
    ],
    # in.hvac_cooling_type exclude existing heat pump options
    'cooling': [
        'Central AC',
        'Room AC'
    ],
    # in.water_heater_efficiency exclude heat pump options, tankless, other fuel (e.g., solar), and indirect fuel oil
    # 'waterHeating': [
    #     'Electric Premium', 'Electric Standard',
    #     'Fuel Oil Premium', 'Fuel Oil Standard', 
    #     'Natural Gas Premium', 'Natural Gas Standard',
    #     'Propane Premium', 'Propane Standard'
    # ],
    # # in.clothes_dryer also includes information on usage (%) exclude homes with no dryer or heat pump dryers
    # 'clothesDrying': [
    #     'Electric', 'Gas', 'Propane'
    # ],
    # # in.cooking_range exclude electric cooking (since resistance ranges are an upgrade option) and homes with no cooking range
    # 'cooking': [
    #     'Gas', 'Propane'
    # ]
} 

# Trane GC focuses on heating and cooling. Cooling category is used for initial replacement cost estimates.
# Cooling is excluded from EQUIPMENT_SPECS and VALID_CATEGORIES because other calculations are not performed.
EQUIPMENT_SPECS = {
    'heating': 15,
    'cooling': 15,
    # 'waterHeating': 12,
    # 'clothesDrying': 13,
    # 'cooking': 15
    }
VALID_CATEGORIES = list(EQUIPMENT_SPECS.keys())

# Run the model for all measure packages (MPs) or a specific MP
# Enclosure upgrades (MP9 and MP10) are excluded for now since they are not yet included in the REMDB v4 code.
VALID_MENU_MPS = [
    0,
    3,
    4,
    # 8,
    # 9,
    # 10
    ]

# Short key identifiers for discount rates (used in dictionaries and user-facing code)
PRIVATE_DISCOUNT_RATE_SHORT_KEYS = [
    # 'fixed_low',
    'fixed_base',
    # 'fixed_high',
    # 'variable'
]

# Full column names for discount rates (used in DataFrames)
PRIVATE_DISCOUNT_RATE_COLS = [
    # 'private_discount_rate_fixed_low',
    'private_discount_rate_fixed_base',
    # 'private_discount_rate_fixed_high',
    # 'private_discount_rate_variable'
]

# Legacy - Method suffixes for file naming
PRIVATE_DISCOUNTING_METHOD_SUFFIXES = {
    # 'private_discount_rate_fixed_low': '_fixed_low',
    'private_discount_rate_fixed_base': '_fixed_base',
    # 'private_discount_rate_fixed_high': '_fixed_high',
    # 'private_discount_rate_variable': '_variable'
}

PUBLIC_DISCOUNTING_METHOD_SUFFIXES = {
    'public_discount_rate': ''
}

# =============================================================
# CONSTANTS: TARE MODEL - GENERAL
# =============================================================

# Updated to 5% based on the latest estimates from EIA, formerly 6%
TD_LOSSES = 0.05 
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)

# Fuel type mapping for column name conventions
FUEL_MAPPING = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}

# Color mapping (keeping original style)
COLOR_MAP_FUEL = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'gray',  # Changed to gray for accessibility
}


# =============================================================
# CONSTANTS: TARE MODEL - PUBLIC PERSPECTIVE CALCULATIONS
# =============================================================
POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
MER_TYPES = ['lrmer', 'srmer']

# Central SCC assumption is considered our base case
# However, we include the lower and upper bounds for sensitivity analysis and to capture the range of estimates in the literature.
# Running all three sensitivity cases is not very computationally intensive.
SCC_ASSUMPTIONS = [
    'lower',
    'central',
    'upper'
    ]

# =============================================================
# CONSTANTS: TARE MODEL - PRIVATE PERSPECTIVE CALCULATIONS
# =============================================================
# Reference dollar year for all private-perspective figures. Income, REMDB
# capital costs, and fuel prices are each inflated from their own source year
# to this year so every dollar value is directly comparable. Defined here so
# the reference year lives in exactly one place rather than as scattered
# literals.
ANCHOR_YEAR = 2025
# FUEL_PRICE_ASSUMPTIONS = ['lower', 'central', 'upper']

# Discount rate constants (centralized for easy modification)
PUBLIC_DISCOUNT_RATE = 0.02      # 2% social discount rate, converted to decimal

# Fixed Private Discount Rate Constants
# PRIVATE_FIXED_RATE_LOW = 0.02
PRIVATE_FIXED_RATE_BASE = 0.07
# PRIVATE_FIXED_RATE_HIGH = 0.12

# Variable Private Discount Rate Parameters
VARIABLE_RATE_MIN = 0.07         # Minimum rate for high-AMI households (>=150% AMI)
VARIABLE_RATE_MAX = 0.45         # Maximum rate for low-AMI households (0% AMI)
AMI_THRESHOLD = 150              # AMI percentage at which minimum rate applies

# Define equipment categories and their corresponding upgrade columns
# There is no separate cooling upgrade column since both receive the same tech upgrade
UPGRADE_COLUMNS = {
    'heating': 'upgrade_hvac_heating_efficiency',
    # 'waterHeating': 'upgrade_water_heater_efficiency',
    # 'clothesDrying': 'upgrade_clothes_dryer',
    # 'cooking': 'upgrade_cooking_range'
    }

# Mapping for categories and their corresponding rebate amounts
REBATE_MAPPING = {
    # There is no separate cooling rebate modeled since both receive the same tech upgrade
    'heating': ('upgrade_hvac_heating_efficiency', ['ASHP', 'MSHP'], 8000.00),
    # 'waterHeating': ('upgrade_water_heater_efficiency', ['Electric Heat Pump'], 1750.00),
    # 'clothesDrying': ('upgrade_clothes_dryer', ['Electric, Premium, Heat Pump, Ventless'], 840.00),
    # 'cooking': ('upgrade_cooking_range', ['Electric, '], 840.00)
}

# =============================================================
# CONSTANTS: IRA REBATE ELIGIBILITY BY MEASURE PACKAGE
# =============================================================
# Only high-efficiency equipment qualifies for IRA HOMES rebates.
# MP3 uses standard-efficiency ASHP (SEER 15) which does NOT meet
# Energy Star certification requirements for rebate eligibility.
# MP4/MP8/MP9/MP10 use high-efficiency ASHP (SEER 24+) and qualify.
REBATE_ELIGIBLE_HEATING_MPS = [4, 8, 9, 10]

# =============================================================
# CONSTANTS: CAPITAL COST SCENARIOS (REMDB v3 + v4)
# =============================================================

# REMDB v4 MID is considered our base case
# Removed v3
REMDB_COST_SCENARIO_KEYS = [
    # 'v4LOW',       # REMDB v4: 10th percentile
    'v4MID',       # REMDB v4: 50th percentile (median)
    # 'v4HIGH'       # REMDB v4: 90th percentile
]

# =============================================================
# CONSTANTS: HVAC REPLACEMENT SCENARIO (Case A / Case B)
# =============================================================
# Internal capital-cost parameter only. Used by the calculate_lifetime_private_impact
# module to decide which incumbent equipment costs are credited when computing net
# capital cost. Do NOT use this list to drive NPV or adoption loops in notebook code
# -- the three public NPV cases are defined in NPV_CASE_CATEGORIES (column_names.py).
#
# Case A ('heating'):
#   Net capital cost = ASHP upgrade - heating replacement cost
#   Assumes the household replaces only the heating system.
#
# Case B ('heating_and_cooling'):
#   Net capital cost = ASHP upgrade - (heating + cooling replacement cost)
#   Assumes the household replaces both heating AND cooling systems
#   with a single heat pump that serves both loads.
VALID_HVAC_REPLACEMENT_SCENARIOS = ['heating', 'heating_and_cooling']

# =============================================================
# CONSTANTS: EFFICIENCY FLOORS FOR REPLACEMENT COST ESTIMATION
# =============================================================
# Applied to pm2 (efficiency) values BEFORE the REMDB v4 regression,
# only for replacement costs.
#
# Rationale: The EUSS housing stock (~2018 vintage) contains systems
# with efficiencies far below what is available or legal today
# (e.g., SEER 8 central ACs, 60% AFUE furnaces). Since replacement
# costs represent what a homeowner would buy TODAY, we clamp ALL
# below-floor pm2 values up to the floor — the minimum efficiency
# equipment a homeowner can legally purchase.
#
# Values are in REMDB pm2 units (SEER1 for cooling/heat-pumps,
# decimal AFUE for furnaces).
#
# Sources:
#   - DOE 2023 final rule: SEER2 14.3 (South) / 13.4 (North) for CAC
#     ≈ SEER1 ~15 (South), ~14 (North). We use 15.0 nationally because
#     the majority of cooling load is in the South region and for
#     consistency with the ASHP floor.
#   - NAECA federal minimum for gas furnaces: 80% AFUE
# =============================================================
EFFICIENCY_FLOORS_PM2 = {
    'air_source_heat_pump_centrally_ducted':       15.0,   # SEER1
    'air_source_heat_pump_non_ducted_multi_zone':  15.0,   # SEER1
    'air_conditioner_centrally_ducted':            15.0,   # SEER1
    'furnaces_gas_furnace':                        0.80,   # AFUE (decimal)
    # 'electric_baseboard_default' has pm2_coef=0, no floor needed
}

# =============================================================
# CONSTANTS: CAPACITY BOUNDS FOR REPLACEMENT COSTS
# =============================================================
# Capacity (pm1) is fed to the REMDB v4 regression exactly as converted from
# the EUSS size fields -- it is never clamped to the training bounds. Values
# outside the bounds are reported for diagnostics only (see
# _report_bounds_comparison in remdb_v4_installed_cost_utils.py); genuine
# outliers are handled by the upstream capacity percentile filter and by NaN
# propagation. The former CAPACITY_BOUND_CLAMPING_TOLERANCE and the tolerance-
# based clamping step were removed on 07 July 2026 because clamping silently
# moved a small number of homes' capacities (and therefore their capital
# costs) with no methodological basis over the plain converted value.

# =============================================================
# CONSTANTS: BSQ / EUSS TIMESERIES COLUMN NAMES
# =============================================================
# Column names for BuildStockQuery timeseries queries against
# ResStock EUSS 2022.1.1 (AMY2018). Used by the peak load
# analysis notebook and peak_load_functions.py.
#
# Weight handling: BSQ reads per-row weights from the metadata
# table and applies them via SUM(enduse × baseline.weight) in
# generated SQL. No hardcoded weight constants are used.

# Timeseries table columns
BLDG_ID_COL: str = "bldg_id"
TIMESTAMP_COL: str = "timestamp"
ELEC_TOTAL_COL: str = "out.electricity.total.energy_consumption"

# BSQ returns enduse columns WITHOUT the 'out.' prefix
BSQ_ELEC_COL: str = "electricity.total.energy_consumption"

# Metadata table columns
METADATA_TABLE: str = "resstock_amy2018_release_1_1_metadata"
COUNTY_COL: str = "in.county"     # GISJOIN format
STATE_COL: str = "in.state"       # 2-char state code
WEIGHT_COL: str = "weight"        # BSQ reads per-row from metadata

# Minimum sample count per county/state for spatial aggregation.
# Set to 1 — all counties are included regardless of sample size.
# Sparsely populated counties naturally have fewer samples; excluding
# them introduces geographic bias.  Consistent with approaches in
# similar ResStock-based studies.
MIN_HOME_COUNT: int = 1

# Reference values for Allegheny County validation
TEST_FIPS: str = "42003"
TEST_GISJOIN: str = "G4200030"

# =============================================================
# CONSTANTS: PRE-TARE KPI VALIDATION
# =============================================================
# Jenkins et al. break-even COP at 90% AFUE reference values.
# Used in Task D cross-validation in the preTARE KPI notebook.
# ASSUMPTION: Jenkins assumes 1020 BTU/cf gas heat content;
# we use 1036 BTU/cf (current EIA average). This ~1.8% difference
# propagates into spark gap and break-even COP.
JENKINS_BREAKEVEN_REF_90: dict = {
    'FL': 1.50, 'PA': 3.51, 'MN': 3.90,
    'AK': 5.69, 'CA': 4.49, 'MA': 3.60,
}

# PA climate zone spot-check ranges for COP benchmark validation.
# PA is primarily CZ 4-5 (Pittsburgh, Philadelphia).
# Source: Literature estimates for ASHP performance in mixed-humid climate.
# TODO: follow-up (P0.2) — PA CZ 6-7 spot check currently fails.
PA_COP_RANGES: dict = {
    'mp3': (1.8, 2.4),
    'mp4': (2.5, 3.4),
}

