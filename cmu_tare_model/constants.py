# Example constants (copied from the original code).
# Adjust or remove them as needed, or move them to a separate config file.
# Excludes HP Tech for Space/Water Heating and Clothes Drying. Also excludes electric resistance cooking and induction cooking.
ALLOWED_TECHNOLOGIES = {
    'heating': [
        'Electricity Baseboard', 'Electricity Electric Boiler', 
        'Electricity Electric Furnace', 'Fuel Oil Fuel Boiler', 'Fuel Oil Fuel Furnace', 
        'Natural Gas Fuel Boiler', 'Natural Gas Fuel Furnace',
        'Propane Fuel Boiler', 'Propane Fuel Furnace'
    ],
    'waterHeating': [
        'Electric Premium', 'Electric Standard',
        'Fuel Oil Premium', 'Fuel Oil Standard', 
        'Natural Gas Premium', 'Natural Gas Standard',
        'Propane Premium', 'Propane Standard'
    ],
    'clothesDrying': [
        'Electric', 'Gas', 'Propane'
    ],
    'cooking': [
        'Gas', 'Propane'
    ]
} 

TD_LOSSES = 0.05 # Updated to 5% based on the latest estimates from EIA, formerly 6%
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
FUEL_MAPPING = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}
FUEL_PRICE_ASSUMPTIONS = ['lower', 'central', 'upper']

POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
MER_TYPES = ['lrmer', 'srmer']
CR_FUNCTIONS = ['acs', 'h6c']
RCM_MODELS = ['ap2', 'easiur', 'inmap']
SCC_ASSUMPTIONS = ['lower', 'central', 'upper']

# DOLLAR_YEAR = 2023
# INFLATION_ADJUSTED_USD = f'USD{DOLLAR_YEAR}'

# Define equipment categories and their corresponding upgrade columns
UPGRADE_COLUMNS = {
    'heating': 'upgrade_hvac_heating_efficiency',
    'waterHeating': 'upgrade_water_heater_efficiency',
    'clothesDrying': 'upgrade_clothes_dryer',
    'cooking': 'upgrade_cooking_range'
    }

# Mapping for categories and their corresponding rebate amounts
REBATE_MAPPING = {
    'heating': ('upgrade_hvac_heating_efficiency', ['ASHP', 'MSHP'], 8000.00),
    'waterHeating': ('upgrade_water_heater_efficiency', ['Electric Heat Pump'], 1750.00),
    'clothesDrying': ('upgrade_clothes_dryer', ['Electric, Premium, Heat Pump, Ventless'], 840.00),
    'cooking': ('upgrade_cooking_range', ['Electric, '], 840.00)
}


# Color mapping (keeping original style)
COLOR_MAP_FUEL = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'gray',  # Changed to gray for accessibility
}
