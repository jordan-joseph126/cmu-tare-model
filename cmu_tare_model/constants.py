from cmu_tare_model.utils.inflation_adjustment import cpi_ratio_2023_2020

# Example constants (copied from the original code).
# Adjust or remove them as needed, or move them to a separate config file.
TD_LOSSES = 0.06
TD_LOSSES_MULTIPLIER = 1 / (1 - TD_LOSSES)
EQUIPMENT_SPECS = {'heating': 15, 'waterHeating': 12, 'clothesDrying': 13, 'cooking': 15}
POLLUTANTS = ['so2', 'nox', 'pm25', 'co2e']
MER_TYPES = ['lrmer', 'srmer']
CR_FUNCTIONS = ['acs', 'h6c']
RCM_MODELS = ['AP2', 'EASIUR', 'InMAP']

EPA_SCC_USD2023_PER_MT_LOW = 14 * cpi_ratio_2023_2020       # 5% constant discount rate reported in IWG (slightly higher than Trump's 7% discount rate ~$1-$6)
EPA_SCC_USD2023_PER_MT_BASE = 51 * cpi_ratio_2023_2020      # Pre-2017 Obama Administration SCC (3% constant discount rate)
EPA_SCC_USD2023_PER_MT_HIGH = 190 * cpi_ratio_2023_2020     # Biden Administration SCC (2% near-term Ramsey discount rate)

