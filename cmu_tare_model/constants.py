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

# ALLOWED_HOUSING_TYPES = ['Single-Family Attached', 'Single-Family Detached', 'Mobile Home', 'Multi-Family with 2 - 4 Units']
ALLOWED_HOUSING_TYPES = ['Single-Family Attached', 'Single-Family Detached']

# Excludes HP Tech for Space/Water Heating and Clothes Drying. Also excludes electric resistance cooking and induction cooking.
# enumeration_dictionary.tsv provides additional details on the allowed technologies for each equipment category.
# Trane GC focuses on heating and cooling. Cooling category is used for initial replacement cost estimates.
ALLOWED_TECHNOLOGIES = {
    # in.hvac_heating_type_and_fuel exclude existing heat pump options
    'heating': [
        'Electricity Baseboard', 'Electricity Electric Boiler', 'Electricity Electric Furnace', 'Electricity ASHP',
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
    8,
    # 9,
    # 10
    ]

# InMAP-ACS is considered our base case
CR_FUNCTIONS = [
    'acs',
    'h6c'
    ]

# InMAP-ACS sensitivity is considered our base case.
RCM_MODELS = [
    'ap2',
    'easiur',
    'inmap'
    ]

# Short key identifiers for discount rates (used in dictionaries and user-facing code)
PRIVATE_DISCOUNT_RATE_SHORT_KEYS = [
    'fixed_low',
    'fixed_base',
    'fixed_high',
    'variable'
]

# Full column names for discount rates (used in DataFrames)
PRIVATE_DISCOUNT_RATE_COLS = [
    'private_discount_rate_fixed_low',
    'private_discount_rate_fixed_base',
    'private_discount_rate_fixed_high',
    'private_discount_rate_variable'
]

# Legacy - Method suffixes for file naming
PRIVATE_DISCOUNTING_METHOD_SUFFIXES = {
    'private_discount_rate_fixed_low': '_fixed_low',
    'private_discount_rate_fixed_base': '_fixed_base',
    'private_discount_rate_fixed_high': '_fixed_high',
    'private_discount_rate_variable': '_variable'
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
# DOLLAR_YEAR = 2023
# INFLATION_ADJUSTED_USD = f'USD{DOLLAR_YEAR}'
# FUEL_PRICE_ASSUMPTIONS = ['lower', 'central', 'upper']

# Discount rate constants (centralized for easy modification)
PUBLIC_DISCOUNT_RATE = 0.02      # 2% social discount rate, converted to decimal

# Fixed Private Discount Rate Constants
PRIVATE_FIXED_RATE_LOW = 0.02
PRIVATE_FIXED_RATE_BASE = 0.07
PRIVATE_FIXED_RATE_HIGH = 0.12

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
# CONSTANTS: CAPITAL COST SCENARIOS (REMDB v3 + v4)
# =============================================================

# REMDB v4 MID is considered our base case
REMDB_COST_SCENARIO_KEYS = [
    'v3',          # Existing method (Excel dictionaries for REMDB v3)
    # 'v4LOW',       # REMDB v4: 10th percentile
    'v4MID',       # REMDB v4: 50th percentile (median)
    # 'v4HIGH'       # REMDB v4: 90th percentile
]
