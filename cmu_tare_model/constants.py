# Constants for the TARE Model (Updated for TRANE Analysis)
# Adjust or remove them as needed, or move them to a separate config file.
# UPDATE: 
# - Updated to handle different enduses based on EQUIPMENT_SPECS.
# - Rest of codebase updated so only initial columns created for cooling and replacement cost calculations performed
# - This allows for a scenario where only heating is replaced AND one where heating and cooling systems are both replace with HP
# - Resolves the excessive data columns and double counting with $8000 rebate. No longer need CDD projections.

# =============================================================
# CONSTANTS: TARE MODEL
# =============================================================
# Configuration
VERBOSE = False
PRINT_DEBUG = False
PRINT_VERBOSE_DATAFRAMES = False

# ALLOWED_HOUSING_TYPES = ['Single-Family Attached', 'Single-Family Detached', 'Mobile Home', 'Multi-Family with 2 - 4 Units']
ALLOWED_HOUSING_TYPES = ['Single-Family Attached', 'Single-Family Detached']

# Excludes HP Tech for Space/Water Heating and Clothes Drying. Also excludes electric resistance cooking and induction cooking.
# enumeration_dictionary.tsv provides additional details on the allowed technologies for each equipment category.
ALLOWED_TECHNOLOGIES = {
    # in.hvac_heating_type_and_fuel exclude existing heat pump options
    'heating': [
        'Electricity Baseboard', 'Electricity Electric Boiler', 
        'Electricity Electric Furnace', 'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
        'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
        'Propane Fuel Boiler', 'Propane Fuel Furnace'
    ],
    # in.hvac_cooling_type exclude existing heat pump options
    'cooling': [
        'Central AC',
        # 'Room AC'
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

# POSSIBLY UPDATE THESE VALUES BASED ON NEW DATA FROM REMBD 2024
# Cooling category is only used for initial columns and retrofit capital costs, not for emissions/health/fuel calculations
EQUIPMENT_SPECS = {
    'heating': 15,
    # 'waterHeating': 12,
    # 'clothesDrying': 13,
    # 'cooking': 15
    }

VALID_CATEGORIES = list(EQUIPMENT_SPECS.keys())

VALID_MENU_MPS = [0, 3, 4, 7, 8, 9, 10]

TD_LOSSES = 0.05 # Updated to 5% based on the latest estimates from EIA, formerly 6%
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
CR_FUNCTIONS = ['acs', 'h6c']
RCM_MODELS = ['ap2', 'easiur', 'inmap']
SCC_ASSUMPTIONS = ['lower', 'central', 'upper']


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

REMDB_COST_SCENARIO_KEYS = [
    'v3',          # Existing method (Excel dictionaries for REMDB v3)
    # 'v4LOW',       # REMDB v4: 10th percentile
    'v4MID',       # REMDB v4: 50th percentile (median)
    # 'v4HIGH'       # REMDB v4: 90th percentile
]
