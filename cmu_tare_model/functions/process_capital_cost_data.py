# import os
# import pandas as pd

# from config import PROJECT_ROOT
# print(f"Project root directory: {PROJECT_ROOT}")

# """
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PROCESS CAPITAL COST DATA
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# """

# # Collect Capital Cost Data for different End-uses
# filename = "tare_retrofit_costs_cpi.xlsx"
# relative_path = os.path.join("cmu_tare_model", "data", "retrofit_costs", filename)
# file_path = os.path.join(PROJECT_ROOT, relative_path)

# print(f"Retrieved data for filename: {filename}")
# print(f"Located at filepath: {file_path}")
# print("\n")

# df_heating_retrofit_costs = pd.read_excel(io=file_path, sheet_name='heating_costs')
# df_waterHeating_retrofit_costs = pd.read_excel(io=file_path, sheet_name='waterHeating_costs')
# df_clothesDrying_retrofit_costs = pd.read_excel(io=file_path, sheet_name='clothesDrying_costs')
# df_cooking_retrofit_costs = pd.read_excel(io=file_path, sheet_name='cooking_costs')
# df_enclosure_retrofit_costs = pd.read_excel(io=file_path, sheet_name='enclosure_upgrade_costs')