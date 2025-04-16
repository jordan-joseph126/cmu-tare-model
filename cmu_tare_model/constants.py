# Example constants (copied from the original code).
# Adjust or remove them as needed, or move them to a separate config file.
TD_LOSSES = 0.06
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
FUEL_MAPPING = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}
FUEL_PRICE_ASSUMPTIONS = ['lower', 'central', 'upper']

POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
MER_TYPES = ['lrmer', 'srmer']
CR_FUNCTIONS = ['acs', 'h6c']
RCM_MODELS = ['AP2', 'EASIUR', 'InMAP']
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
