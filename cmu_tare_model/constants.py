# Example constants (copied from the original code).
# Adjust or remove them as needed, or move them to a separate config file.
TD_LOSSES = 0.06
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
FUEL_MAPPING = {'Electricity': 'electricity', 'Natural Gas': 'naturalGas', 'Fuel Oil': 'fuelOil', 'Propane': 'propane'}

POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
MER_TYPES = ['lrmer', 'srmer']
CR_FUNCTIONS = ['acs', 'h6c']
RCM_MODELS = ['AP2', 'EASIUR', 'InMAP']
SCC_ASSUMPTIONS = ['lower', 'central', 'upper']

DOLLAR_YEAR = 2023
INFLATION_ADJUSTED_USD = f'USD{DOLLAR_YEAR}'
