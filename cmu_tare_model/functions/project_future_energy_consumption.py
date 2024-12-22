import os
import pandas as pd
import numpy as np
import re

from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
PROJECT FUTURE ENERGY CONSUMPTION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
- HDD factors for different census divisions and years
- Functions to project future energy consumption based on HDD projections
"""

# HDD factors for different census divisions and years
# Factors for 2022 to 2050
# Define the relative path to the target file
filename = 'aeo_projections_2022_2050.xlsx'
relative_path = os.path.join("cmu_tare_model", "data", "projections", filename)
file_path = os.path.join(PROJECT_ROOT, relative_path)

df_hdd_projection_factors = pd.read_excel(io=file_path, sheet_name='hdd_factors_2022_2050')

print(f"Retrieved data for filename: {filename}")
print(f"Located at filepath: {file_path}")

# Convert the factors dataframe into a lookup dictionary
lookup_hdd_factor = df_hdd_projection_factors.set_index(['census_division']).to_dict('index')
lookup_hdd_factor

# LAST UPDATED ON DECEMBER 5, 2024 @ 6:50 PM
# UPDATED TO RETURN BOTH DF_COPY AND DF_CONSUMPTION
# THIS FIXES THE ISSUE WITH MP_SCENARIO_DAMAGES AND PUBLIC NPV NOT BEING CALCULATED CORRECTLY
# df_consumption contains only the projected consumption data. df_copy contains all columns including the projected consumption data.
def project_future_consumption(df, lookup_hdd_factor, menu_mp):
    """
    Projects future energy consumption based on baseline or upgraded equipment specifications.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing baseline consumption data.
    lookup_hdd_factor (dict): A dictionary with Heating Degree Day (HDD) factors for different census divisions and years.
    menu_mp (int): Indicates the measure package to apply. 0 for baseline, 8/9/10 for retrofit scenarios.
    
    Returns:
    pd.DataFrame: A DataFrame with projected future energy consumption and reductions.
    """

    # Equipment lifetime specifications in years
    equipment_specs = {
        'heating': 15,
        'waterHeating': 12,
        'clothesDrying': 13,
        'cooking': 15
    }

    # Create a copy of the input DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Check if the 'census_division' column exists in the DataFrame
    if 'census_division' not in df_copy.columns:
        raise KeyError("'census_division' column is missing from the DataFrame")

    # Prepare a dictionary to hold new columns for projected consumption
    new_columns = {}

    # Baseline policy_scenario: Existing Equipment
    if menu_mp == 0:
        for category, lifetime in equipment_specs.items():
            print(f"Projecting Future Energy Consumption (Baseline Equipment): {category}")
            for year in range(1, lifetime + 1):
                year_label = 2023 + year

                # Adjust consumption based on HDD factors for heating and water heating
                if category in ['heating', 'waterHeating']:
                    hdd_factor = df_copy['census_division'].map(lambda x: lookup_hdd_factor.get(x, {}).get(year_label, lookup_hdd_factor['National'][year_label]))
                    new_columns[f'baseline_{year_label}_{category}_consumption'] = (df_copy[f'baseline_{category}_consumption'] * hdd_factor).round(2)

                else:
                    new_columns[f'baseline_{year_label}_{category}_consumption'] = df_copy[f'baseline_{category}_consumption'].round(2)

    # Retrofit policy_scenario: Upgraded Equipment (Measure Packages 8, 9, 10)
    else:
        for category, lifetime in equipment_specs.items():
            print(f"Projecting Future Energy Consumption (Upgraded Equipment): {category}")
            for year in range(1, lifetime + 1):
                year_label = 2023 + year

                # Adjust consumption based on HDD factors for heating and water heating
                if category in ['heating', 'waterHeating']:
                    hdd_factor = df_copy['census_division'].map(lambda x: lookup_hdd_factor.get(x, {}).get(year_label, lookup_hdd_factor['National'][year_label]))
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'] = (df_copy[f'mp{menu_mp}_{category}_consumption'] * hdd_factor).round(2)

                    # Calculate the reduction in annual energy consumption
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_reduction_consumption'] = df_copy[f'baseline_{year_label}_{category}_consumption'].sub(
                        new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'], axis=0, fill_value=0
                    ).round(2)
                else:
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'] = df_copy[f'mp{menu_mp}_{category}_consumption'].round(2)

                    # Calculate the reduction in annual energy consumption
                    new_columns[f'mp{menu_mp}_{year_label}_{category}_reduction_consumption'] = df_copy[f'baseline_{year_label}_{category}_consumption'].sub(
                        new_columns[f'mp{menu_mp}_{year_label}_{category}_consumption'], axis=0, fill_value=0
                    ).round(2)

    # Calculate the new columns based on policy scenario and create dataframe based on df_copy index
    df_new_columns = pd.DataFrame(new_columns, index=df_copy.index)

    # Identify overlapping columns between the new and existing DataFrame.
    overlapping_columns = df_new_columns.columns.intersection(df_copy.columns)

    # Drop overlapping columns from df_copy.
    if not overlapping_columns.empty:
        df_copy.drop(columns=overlapping_columns, inplace=True)

    # Merge new columns into df_copy, ensuring no duplicates or overwrites occur.
    df_copy = df_copy.join(df_new_columns, how='left')

    df_consumption = df_copy.copy()

    # Return the updated DataFrames. df_consumption contains only the projected consumption data. df_copy contains all columns including the projected consumption data.
    return df_copy, df_consumption

