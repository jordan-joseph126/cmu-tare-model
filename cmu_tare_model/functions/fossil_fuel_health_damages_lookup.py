# import pandas as pd
# import os
# from config import PROJECT_ROOT

# # import functions.tare_setup as tare_setup
# from cmu_tare_model.functions.rsMeans_adjustment import (
#     cpi_ratio_2023_2020,
#     cpi_ratio_2023_2010,
#     cpi_ratio_2023_2021
#     )

# """
# DAMAGES FROM CLIMATE RELATED EMISSIONS (CO2e):
#     Use the updated Social Cost of Carbon (190 USD-2020/ton CO2) and inflate to USD-2023
#         - EPA Median for 2% near term discount rate and most commonly mentioned value is 190 USD-2020 using the GIVE model.
#         - 190 USD-2020 has some inconsistency with the VSL being used. An old study and 2008 VSL is noted
#         - 190 USD value and inflate to USD 2023 because there is a clear source and ease of replicability.

#     Adjustment for VSL
#     - EASIUR uses a VSL of 8.8M USD-2010 
#     - New EPA VSL is 11.3M USD-2021
#     - INFLATE TO $USD-2023

#     ALL DOLLAR VALUES ARE NOW IN USD2023, PREVIOUSLY USED $USD-2021
# """

# # For CO2 adjust SCC
# # Create an adjustment factor for the new Social Cost of Carbon (SCC)
# epa_scc = 190 * cpi_ratio_2023_2020
# old_scc = 40 * cpi_ratio_2023_2010
# scc_adjustment_factor = epa_scc / old_scc

# # For Health-Related Emissions Adjust for different Value of a Statistical Life (VSL) values
# # Current VSL is $11.3 M USD2021
# # INFLATE TO USD2022, PREVIOUSLY USD2021
# current_VSL_USD2022 = 11.3 * cpi_ratio_2023_2021

# # Easiur uses a VSL of $8.8 M USD2010
# # INFLATE TO USD2022, PREVIOUSLY USD2021
# easiur_VSL_USD2022 = 8.8 * (cpi_ratio_2023_2010)

# # Calculate VSL adjustment factor
# vsl_adjustment_factor = current_VSL_USD2022 / easiur_VSL_USD2022



# print(
# """
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# FOSSIL FUEL DAMAGES LOOKUP: Quantify monitized HEALTH damages using EASIUR Marginal Social Cost Factors
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THE STEPS BELOW SUMMARIZE WHAT WAS DONE TO OBTAIN ALL NATIONAL EASIUR VALUES INCLUDED IN THE CSV FILE:
# - Obtain all of the dwelling unit latitude and longitude values from the metadata columns
# - Make a new dataframe of just the longitude and latitude values 
#     - Make sure that the order is (longitude, latitude)
#     - Do not include the index or column name when exporting 
# - Export the CSV
# - **Upload csv to EASIUR Website:**
#     - Website: https://barney.ce.cmu.edu/~jinhyok/easiur/online/
#     - See inputs in respective sections
# - Download the file and put it in the 'easiur_batchConversion_download' folder
# - Copy and paste the name of the file EASIUR generated when prompted
# - Copy and paste the name of the filepath for the 'easiur_batchConversion_download' folder when prompted
# - Match up the longitude and latitudes for each dwelling unit with the selected damages
# """)

# # Create a dataframe containing just the longitude and Latitude
# df_EASIUR_batchConversion = pd.DataFrame({
#     'Longitude':df_euss_am_baseline['in.weather_file_longitude'],
#     'Latitude':df_euss_am_baseline['in.weather_file_latitude'],
# })

# # Drop duplicate rows based on 'Longitude' and 'Latitude' columns
# df_EASIUR_batchConversion.drop_duplicates(subset=['Longitude', 'Latitude'], keep='first', inplace=True)

# # Create a location ID for the name of the batch conversion file
# while True:
#     if menu_state == 'N':
#         location_id = 'National'
#         print("You chose to analyze all of the United States.")
#         break
#     elif menu_state == 'Y':
#         if menu_city == 'N':
#             try:
#                 location_id = str(input_state)
#                 print(f"Location ID is: {location_id}")
#                 break
#             except ValueError:
#                 print("Invalid input for state!")
#         elif menu_city == 'Y':
#             try:
#                 location_id = input_cityFilter.replace(', ', '_').strip()
#                 print(f"Location ID is: {location_id}")
#                 break
#             except AttributeError:
#                 print("Invalid input for city filter!")
#         else:
#             print("Incorrect state or city filter assignment!")
#     else:
#         print("Invalid data location. Check your inputs at the beginning of this notebook!")

# # Updated GitHub code has EASIUR file with all unique latitude, longitude coordinates in the US
# filename = 'easiur_National2024-06-1421-22.csv'
# relative_path = os.path.join("" margDamages_EASIUR\easiur_batchConversion_download", filename)
# file_path = os.path.join(PROJECT_ROOT, relative_path)

# print(f"Retrieved data for filename: {filename}")
# print(f"Located at filepath: {file_path}")
# print("\n")

# df_margSocialCosts = pd.read_csv(file_path)

# # Convert from kg/MWh to lb/kWh
# # Obtain value from the CSV file and convert to lbs pollutant per kWh 

# # Define df_marg_social_costs_EASIUR DataFrame first
# df_marg_social_costs_EASIUR = pd.DataFrame({
#     'Longitude': df_margSocialCosts['Longitude'],
#     'Latitude': df_margSocialCosts['Latitude'],
# })

# # Use df_marg_social_costs_EASIUR in the calculation of other columns
# # Also adjust the VSL
# df_marg_social_costs_EASIUR['marg_social_costs_pm25'] = round((df_margSocialCosts['PM25 Annual Ground'] * (1/2204.6) * vsl_adjustment_factor), 2)
# df_marg_social_costs_EASIUR['marg_social_costs_so2'] = round((df_margSocialCosts['SO2 Annual Ground'] * (1/2204.6) * vsl_adjustment_factor), 2)
# df_marg_social_costs_EASIUR['marg_social_costs_nox'] = round((df_margSocialCosts['NOX Annual Ground'] * (1/2204.6) * vsl_adjustment_factor), 2)
# df_marg_social_costs_EASIUR['unit'] = '[$USD2023/lb]'

# # Dispalay the EASIUR marginal social costs df
# print(df_marg_social_costs_EASIUR)

# # FOSSIL FUELS DAMAGES LOOKUP
# # Create a damages_fossil_fuel_lookup dictionary from df_marg_social_costs_EASIUR
# damages_fossil_fuel_lookup = df_marg_social_costs_EASIUR.groupby(['Longitude', 'Latitude']).first().to_dict()
# print(damages_fossil_fuel_lookup)