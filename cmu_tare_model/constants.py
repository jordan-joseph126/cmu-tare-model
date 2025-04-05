from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2023_2021, cpi_ratio_2023_2022

# Example constants (copied from the original code).
# Adjust or remove them as needed, or move them to a separate config file.
TD_LOSSES = 0.06
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
MER_TYPES = ['lrmer', 'srmer']
CR_FUNCTIONS = ['acs', 'h6c']
RCM_MODELS = ['AP2', 'EASIUR', 'InMAP']
