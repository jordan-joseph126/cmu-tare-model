# Constants for the TARE Model (Updated for TRANE Analysis)
# Adjust or remove them as needed, or move them to a separate config file.

VALID_MENU_MPS = [3, 4, 7, 8, 9, 10]

ALLOWED_HOUSING_TYPES = ['Single-Family Attached', 'Single-Family Detached', 'Mobile Home', 'Multi-Family with 2 - 4 Units']

TD_LOSSES = 0.05 # Updated to 5% based on the latest estimates from EIA, formerly 6%
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)

FUEL_MAPPING = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}
FUEL_PRICE_ASSUMPTIONS = ['lower', 'central', 'upper']

POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
MER_TYPES = ['lrmer', 'srmer']
CR_FUNCTIONS = ['acs', 'h6c']
RCM_MODELS = ['ap2', 'easiur', 'inmap']
SCC_ASSUMPTIONS = ['lower', 'central', 'upper']

# Color mapping (keeping original style)
COLOR_MAP_FUEL = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'gray',  # Changed to gray for accessibility
}

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
        'Central AC', 'Room AC'
    ],
    # # in.water_heater_efficiency exclude heat pump options, tankless, other fuel (e.g., solar), and indirect fuel oil
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
EQUIPMENT_SPECS = {
    'heating': 15,
    'cooling': 15,
    # 'waterHeating': 12,
    # 'clothesDrying': 13,
    # 'cooking': 15
    }

# Define equipment categories and their corresponding upgrade columns
UPGRADE_COLUMNS = {
    'heating': 'upgrade_hvac_heating_efficiency',
    'cooling': 'upgrade_hvac_cooling_efficiency',
    # 'waterHeating': 'upgrade_water_heater_efficiency',
    # 'clothesDrying': 'upgrade_clothes_dryer',
    # 'cooking': 'upgrade_cooking_range'
    }

# Mapping for categories and their corresponding rebate amounts
REBATE_MAPPING = {
    # Be sure to update code logic so that the space conditioning rebate is only applied once (i.e., $8000 total, not $8000 per equipment if both heating and cooling are upgraded)
    'heating': ('upgrade_hvac_heating_efficiency', ['ASHP', 'MSHP'], 8000.00),
    'cooling': ('upgrade_hvac_cooling_efficiency', ['Heat Pump'], 0.00),
    # 'waterHeating': ('upgrade_water_heater_efficiency', ['Electric Heat Pump'], 1750.00),
    # 'clothesDrying': ('upgrade_clothes_dryer', ['Electric, Premium, Heat Pump, Ventless'], 840.00),
    # 'cooking': ('upgrade_cooking_range', ['Electric, '], 840.00)
}

# # For the TRANE Technologies analysis focusing only on space conditioning (heating and cooling), we modify the constants as follows:
# # Excludes HP Tech for Space/Water Heating and Clothes Drying. Also excludes electric resistance cooking and induction cooking.
# # enumeration_dictionary.tsv provides additional details on the allowed technologies for each equipment category.
# TRANE_ALLOWED_TECHNOLOGIES = {
#     # in.hvac_heating_type_and_fuel exclude existing heat pump options
#     'heating': [
#         'Electricity Baseboard', 'Electricity Electric Boiler', 
#         'Electricity Electric Furnace', 'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
#         'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
#         'Propane Fuel Boiler', 'Propane Fuel Furnace'
#     ],
#     # in.hvac_cooling_type exclude existing heat pump options
#     'cooling': [
#         'Central AC', 'Room AC'
#     ],
# } 

# # POSSIBLY UPDATE THESE VALUES BASED ON NEW DATA FROM REMBD 2024
# TRANE_EQUIPMENT_SPECS = {'heating': 15, 'cooling': 15}

# # Define equipment categories and their corresponding upgrade columns
# TRANE_UPGRADE_COLUMNS = {
#     'heating': 'upgrade_hvac_heating_efficiency',
#     'cooling': 'upgrade_hvac_cooling_efficiency',
#     }

# # Mapping for categories and their corresponding rebate amounts
# TRANE_REBATE_MAPPING = {
#     # Be sure to update code logic so that the space conditioning rebate is only applied once (i.e., $8000 total, not $8000 per equipment if both heating and cooling are upgraded)
#     'heating': ('upgrade_hvac_heating_efficiency', ['ASHP', 'MSHP'], 8000.00),
#     'cooling': ('upgrade_hvac_cooling_efficiency', ['ASHP', 'MSHP'], 8000.00),
# }
