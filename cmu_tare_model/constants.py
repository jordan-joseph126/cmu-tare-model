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
MAP_TITLE_PAD = 5
MAP_CBAR_LABEL_FONT_SIZE = 18
MAP_CBAR_TICK_LABEL_SIZE = 18
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
#
# ANCHOR_YEAR is also the first year of every lifetime cost stream. Fuel-price
# and degree-day projection factors are measured relative to this year, so the
# factor for ANCHOR_YEAR itself is exactly 1.0 in every source file. The
# projection data begins in this year: there is no earlier year to fall back
# on, and nothing in the model may invent one.
ANCHOR_YEAR = 2025

# Last year covered by the fuel-price and degree-day projection files. Used to
# check at load time that a projection file covers ANCHOR_YEAR through this
# year with no gaps, so a truncated or re-cut source file is caught on import
# rather than silently shortening a cost stream.
PROJECTION_END_YEAR = 2050
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
# Only ENERGY STAR-certified heat pumps qualify for the federal rebates.
# MP3's modeled heat pump (SEER 15 / 9.0 HSPF) sits just below the ENERGY STAR
# minimum, but is re-specified to the ENERGY STAR floor (>= 16.0 SEER1 /
# >= 9.5 HSPF1) in process_euss_data.df_enduse_compare so it qualifies -- and
# its capital cost reflects that ENERGY STAR install. MP4/MP8/MP9/MP10 use
# high-efficiency ASHP (SEER 24+) and qualify as modeled.
REBATE_ELIGIBLE_HEATING_MPS = [3, 4, 8, 9, 10]

# =============================================================
# CONSTANTS: REBATE-POLICY-SCENARIO SENSITIVITY AXIS (2024 vs June 2026 DOE guidance)
# =============================================================
# The rebate program is a sensitivity axis, handled like the discount-rate axis:
# each rebate policy scenario produces its own parallel net-capital -> NPV ->
# adopter columns in one dataframe. Two rebate policy scenarios already exist
# implicitly in the six NPV cases:
#   - unsubsidized              -> the existing '_unsub' cases (no rebate)
#   - subsidized, 2024 guidance -> the existing '_sub' cases (current HEEHR)
# This constant names the subsidized guidances that carry a rebate column. The
# 2024 guidance keeps the original (guidance-less) rebate column names so those
# results stay byte-identical; June 2026 adds a distinct token.
REBATE_GUIDANCE_IRA2024 = "ira2024"
REBATE_GUIDANCE_JUNE2026 = "june2026"
REBATE_POLICY_SCENARIOS = [REBATE_GUIDANCE_IRA2024, REBATE_GUIDANCE_JUNE2026]

# --- June 2026 DOE guidance rule constants ---
# Income cutoffs are AMI ratios. The percent_AMI column is on a 0-150+ percent
# scale, so the call site compares percent_AMI against these values x 100
# (or divides percent_AMI by 100). Kept as ratios to match the ratified rule.
AMI_LOW_CUTOFF = 0.80        # <=80% AMI -> HEEHR full cost-share
AMI_MODERATE_CUTOFF = 1.50   # 80-150% AMI -> HEEHR half cost-share; >150% -> HOMES

# HEEHR: a fixed per-measure cap; income varies the SHARE OF PROJECT COST covered,
# not the cap. This matches the existing 100%/50% HEEHR behavior.
HEEHR_COVERAGE_LOW = 1.00    # <=80% AMI covers 100% of project cost, up to the cap
HEEHR_COVERAGE_MOD = 0.50    # 80-150% AMI covers 50% of project cost, up to the cap
HEEHR_CAP_HEAT_PUMP = 8_000  # dollars; per-heat-pump cap, fixed across income

# HOMES: savings-based, consulted only above 150% AMI. Because every HOMES home
# is >150% AMI, the <=80% AMI doubling is unreachable by construction, so only
# the non-LMI amounts are implemented.
HOMES_MIN_SAVINGS_FRAC = 0.20    # >=20% whole-home savings -> tier 1
HOMES_TIER2_SAVINGS_FRAC = 0.35  # >=35% whole-home savings -> tier 2
HOMES_CAP_TIER1 = 2_000          # dollars; 20-34% savings, non-LMI
HOMES_CAP_TIER2 = 4_000          # dollars; >=35% savings, non-LMI
HOMES_COVERAGE_NON_LMI = 0.50    # covers 50% of project cost, up to the tier cap

# June 2026 fuel gate: rebates may no longer fund removing a fossil heating
# system. TARE models only full electrification, so only homes whose existing
# heating is electric resistance qualify. Value matches the base_heating_fuel
# label for electric-resistance heating.
ELECTRIC_RESISTANCE_BASELINE = {"Electricity"}

# States that never participated in the federal rebate programs are ineligible
# under every rebate policy scenario (2024 HEEHR and June 2026 HEEHR + HOMES).
# South Dakota never participated. Two-letter USPS abbreviations, matching the
# `state` column.
NON_PARTICIPATING_REBATE_STATES = {"SD"}

# Program labels recorded in the rebate_eligibility output column (both vintages).
REBATE_NONE = "Not Eligible"
REBATE_HEEHR = "HEEHR"
REBATE_HOMES = "HOMES"

# =============================================================
# CONSTANTS: PER-VINTAGE REBATE RULE CONFIG (central rebate function)
# =============================================================
# One rebate function (calculate_rebate_program) reads this config to apply the
# vintage-specific rules. Each vintage models BOTH programs (HEEHR + HOMES),
# routed by income (HEEHR at/below 150% AMI, HOMES above). The four rule
# scenarios in docs/rebate_guidance_reference.md map onto two vintages here:
# 2024 = {heehr_2024, homes_2024}; June 2026 = {heehr_2026, homes_2026}.
#
# Config keys:
#   column_guidance -- token passed to create_rebate_col. None keeps the 2024
#       rebate column name guidance-less so the existing '_sub' NPV/adopter
#       columns stay byte-identical; 'june2026' gives the June 2026 column its
#       own name.
#   eligibility_col -- name template ('{mp}' substituted) for the program label
#       column ('HEEHR'/'HOMES'/'None').
#   heehr_fuel_gate -- True restricts HEEHR to existing electric-resistance
#       heating (June 2026: rebates may not fund removing a fossil system).
#       False allows fuel switching (2024).
#   homes_enabled -- whether the HOMES savings-tier pathway is modeled for homes
#       above 150% AMI. 2024 starts False (matches the historical HEEHR-only
#       behavior); it is flipped to True when 2024 HOMES is added (the one
#       intended value move). 2026 already models HOMES.
#   homes_fuel_gate -- True restricts HOMES to electric-resistance heating.
#       HOMES is fuel-neutral by design (the fossil-removal restriction is
#       HEEHR-only), so 2024 is False. 2026 is True ONLY to preserve the current
#       byte-identical June 2026 output; making 2026 HOMES fuel-neutral is a
#       separate value move deferred to the full-run re-derivation session.
REBATE_RULE_CONFIG = {
    REBATE_GUIDANCE_IRA2024: {
        "column_guidance": None,
        "eligibility_col": "mp{mp}_rebate_eligibility_ira2024",
        "heehr_fuel_gate": False,
        # THE ONE INTENDED VALUE MOVE (2026-07-14 session): 2024 HOMES enabled.
        # The 2024 program previously modeled HEEHR only, so homes above 150% AMI
        # received $0. Enabling the fuel-neutral HOMES pathway credits those homes
        # (fossil and electric) under 2024, which raises the '_sub' NPV/adopter
        # adoption rows. HOMES has no fuel gate (the fossil-removal restriction is
        # HEEHR-only), so homes_fuel_gate stays False.
        "homes_enabled": True,
        "homes_fuel_gate": False,
        "heehr_python_round": True,
    },
    REBATE_GUIDANCE_JUNE2026: {
        "column_guidance": REBATE_GUIDANCE_JUNE2026,
        "eligibility_col": "mp{mp}_rebate_eligibility_june2026",
        "heehr_fuel_gate": True,
        "homes_enabled": True,
        "homes_fuel_gate": True,
        "heehr_python_round": False,
    },
}
# heehr_python_round preserves a pre-consolidation rounding quirk so both
# vintages stay byte-identical. The original 2024 path rounded the covered
# project cost with Python's builtin round(); the original June 2026 path used
# numpy's array .round(). Those two disagree by one cent on exact half-cent
# products (e.g. a $9,777.77 cost at 50% coverage = $4,888.885 -> Python
# $4,888.89 vs numpy $4,888.88), which occurs whenever a two-decimal cost with
# an odd final cent is halved. Keeping the per-vintage rounding avoids a
# sub-penny value move on the moderate-income HEEHR homes.

# =============================================================
# CONSTANTS: CAPITAL COST SCENARIOS (REMDB v3 + v4)
# =============================================================

# REMDB v4 MID is considered our base case
# Removed v3 -- if v3 is ever turned back on, it still has the old-system-size
# bug fixed in v4 on 20 Aug 2026; see calculate_equipment_replacement_costs.py.
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
# Whole-home ELECTRICITY total (all electric end uses) -- NOT the all-fuel
# 'out.site_energy.total.energy_consumption', which sums gas/oil/propane in
# kWh-equivalent. Use this electricity total for demand/peak metrics; the site
# energy total is only a HOMES savings-fraction denominator elsewhere.
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

