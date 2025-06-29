import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
# Assuming utils imports are relevant for other parts of your project
# from utils import building_unit_types, adoption_tiers, original_adoption_tiers
import os
import pickle
import numpy as np
import seaborn as sns
from itertools import product
from collections import defaultdict
import math

# --- Configuration and Directory Setup ---
# Use fixed paths as per your PSC run configuration
cwd_filepath = os.getcwd()
TARE_dir = os.path.join("/ocean", "projects", "eng220005p", "agautam3", "cmu-tare-model")
resstock_dir = os.path.join("/ocean", "projects", "eng220005p", "agautam3", "resstock-3.4.0")
visuals_dir = os.path.join("/ocean","projects","eng220005p","agautam3","TPIA-MP", "visuals")

save_not_show = False # This variable isn't used in the provided snippet logic, kept for context

urbanization_level_dict = {
    "urban":"In metro area, principal city",
    "suburban":"In metro area, not/partially in principal city",
    "rural":"Not/partially in metro area",
}

level_of_urbanization_dict = {
    "In metro area, principal city":"urban",
    "In metro area, not/partially in principal city":"suburban",
    "Not/partially in metro area":"rural",
}

# Define adoption tiers that signify "adopting"
ADOPTION_TIERS = ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility', 'Feasible per MMPV']

# --- Main Execution Logic ---
# if __name__ == "__main__":
penetration_level_label = "COST_BASED" # Only one value used in the original snippet

region = "national_ASHP"
NUM_RESIDENCES = 8000
unit_type = "all"
prefix="alpha_beta"
POWER_FACTOR = 0.9
electricity_cols = ["Fuel Use: Electricity: Total"] # Columns to read from timeseries files

# 1. Load DataFrames once
tare_results_filepath = os.path.join(TARE_dir, "output_results",f"{region}_{NUM_RESIDENCES}_{unit_type}_unit_residence", f"{prefix}_NPV_tare_output.csv")
tare_results_df = pd.read_csv(tare_results_filepath)

buildstock_filepath = os.path.join(resstock_dir, "resources", "national", f"{'all_national' if region == 'national_ASHP' else region}_{NUM_RESIDENCES}_all_unit_buildstock.csv")
buildstock_df = pd.read_csv(buildstock_filepath, low_memory=True) # Use low_memory for potentially large files

# 2. Merge tare_results_df with buildstock_df for direct access to climate zone and urbanization
# Rename 'Building' in buildstock_df to match 'bldg_id' in tare_results_df for merge
# Or simply merge on 'bldg_id' and 'Building' if pandas can infer
merged_df = tare_results_df.merge(buildstock_df[['Building', 'ASHRAE IECC Climate Zone 2004', 'PUMA Metro Status']],
                                    left_on='bldg_id',
                                    right_on='Building',
                                    how='inner') # Use inner to only keep buildings present in both

# Add urbanization_level (urban, suburban, rural) as a new column
merged_df['urbanization_level'] = merged_df['PUMA Metro Status'].map(level_of_urbanization_dict)

# 3. Vectorize 'is_adopting' calculation
if penetration_level_label == "1pt0":
    merged_df["is_adopting"] = True
else: # For "COST_BASED"
    merged_df["is_adopting"] = merged_df["iraRef_mp8_heating_adoption"].isin(ADOPTION_TIERS)

# Sort by NPV if still needed for other purposes, though not directly used in the final plot
merged_df.sort_values(by="iraRef_mp8_heating_total_npv_moreWTP", axis=0, ascending=False, inplace=True)

# Filter to only include adopting buildings for timeseries processing
adopting_buildings_df = merged_df[merged_df['is_adopting']].copy()


# 4. Process Timeseries Data More Efficiently
# Prepare to store lists of DataFrames for concatenation later
per_climate_zone_and_urbanization_dfs_lists = defaultdict(lambda: defaultdict(list))

# Pre-compute the common part of the resstock path
base_resstock_path_segment = os.path.join(resstock_dir, f"{region}{'' if '_' in region else '_upgrades'}_{NUM_RESIDENCES}_{unit_type}_unit_residence")
upgrade_resstock_path_segment = os.path.join(resstock_dir, f"{region}{'' if '_' in region else '_upgrades'}_{NUM_RESIDENCES}_{unit_type}_unit_residence_ASHP")

# # Iterate only through the adopting buildings
# for idx, row in adopting_buildings_df.iterrows():

baseline_peak_value_list = None
upgrade_peak_value_list = None

# Iterate only through all buildings
for idx, row in merged_df.iterrows():
    bldg_id = int(row["bldg_id"]) # Use bldg_id directly from the merged_df

    baseline_load_file_path = os.path.join(base_resstock_path_segment, f"run{bldg_id}", "run", "results_timeseries.csv")
    upgrade_load_file_path = os.path.join(upgrade_resstock_path_segment, f"run{bldg_id}", "run", "results_timeseries.csv")

    if not os.path.exists(baseline_load_file_path):
        # print(f"Couldn't find baseline data for building {bldg_id}, not adding it")
        continue

    baseline_power_timeseries_df = pd.read_csv(baseline_load_file_path, low_memory=True, usecols=electricity_cols)
    # Ensure it's numeric and handle potential issues with header row/data types
    baseline_power_timeseries_df = baseline_power_timeseries_df.iloc[1:].astype(float) # Remove first row if it's string header, convert to float

    # Calculate baseline power in VA (if power factor is applied to kWh)
    baseline_power_timeseries_df.iloc[:, 0] = baseline_power_timeseries_df.iloc[:, 0] / POWER_FACTOR

    if baseline_peak_value_list is None:
        baseline_peak_value_list = baseline_power_timeseries_df
    else:
        baseline_peak_value_list += baseline_power_timeseries_df

    if not os.path.exists(upgrade_load_file_path):
        # print(f"Couldn't find upgrade data for building {bldg_id}, so adding a zero sequence for it")
        # Create a zero DataFrame matching baseline's shape and index
        power_timeseries_diff_df = pd.DataFrame(0.0, index=baseline_power_timeseries_df.index, columns=[str(bldg_id)])

        if upgrade_peak_value_list is None:
            upgrade_peak_value_list = baseline_power_timeseries_df
        else:
            upgrade_peak_value_list += baseline_power_timeseries_df

    else:
        upgrade_power_timeseries_df = pd.read_csv(upgrade_load_file_path, low_memory=True, usecols=electricity_cols)
        upgrade_power_timeseries_df = upgrade_power_timeseries_df.iloc[1:].astype(float) # Remove first row, convert to float
        upgrade_power_timeseries_df.iloc[:, 0] = upgrade_power_timeseries_df.iloc[:, 0] / POWER_FACTOR

        if upgrade_peak_value_list is None:
            upgrade_peak_value_list = upgrade_power_timeseries_df
        else:
            upgrade_peak_value_list += upgrade_power_timeseries_df
        # upgrade_peak_value_list.append(upgrade_power_timeseries_df.max())

            
            # # Ensure indices are aligned for subtraction
            # power_timeseries_diff_df = upgrade_power_timeseries_df - baseline_power_timeseries_df
            # power_timeseries_diff_df.columns = [str(bldg_id)] # Rename column to bldg_id for later concat/sum

    # Store the individual timeseries diff DFs in lists
    climate_zone = row['ASHRAE IECC Climate Zone 2004']
    level_of_urbanization = row['urbanization_level']
    per_climate_zone_and_urbanization_dfs_lists[climate_zone][level_of_urbanization].append(power_timeseries_diff_df)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_title("Peak electricity usage over the year per residence")
    ax.set_ylabel("MWh")
    plt.bar([0,1],[baseline_peak_value_list.max()*16.500, upgrade_peak_value_list.max()*16.500])
    plt.xticks([0,1], ["Baseline", "With Heat Pumps"])
    
    bar_chart_dir = os.path.join(visuals_dir, "bar_chart")
    os.makedirs(bar_chart_dir, exist_ok=True)
    bar_chart_path = os.path.join(bar_chart_dir, "bar_chart_for_national_load.svg")

    fig.savefig(bar_chart_path)
    plt.close(fig) # Close the figure to free up memory
    # plt.ylabel("")
    
    # # 5. Concatenate lists of DataFrames for each category outside the loop
    # per_climate_zone_and_urbanization_dfs = defaultdict(lambda: defaultdict(lambda: None))
    # for climate_zone, urbanization_levels_dict in per_climate_zone_and_urbanization_dfs_lists.items():
    #     for level_of_urbanization, list_of_dfs in urbanization_levels_dict.items():
    #         if list_of_dfs: # Only concatenate if there are DFs in the list
    #             # Concatenate all individual building time series for this specific zone/urbanization
    #             per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization] = pd.concat(list_of_dfs, axis=1)
    #         else:
    #             # If no data for this combo, keep it None or an empty DF as desired
    #             per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization] = None


    # # 6. Prepare data for the final box plot more efficiently
    # climate_zones = ["1A","2A","2B","3A","3B","3C","4A","4B","4C","5A","5B","6A","6B","7A","7B"]
    # urbanization_levels = urbanization_level_dict.keys() # 'urban', 'suburban', 'rural'

    # # 6a. Plot individual box plots per region
    # # fig, ax = plt.subplots(figsize=(16,5))

    # all_series_for_final_concat = []
    # for climate_zone, level_of_urbanization in product(climate_zones, urbanization_levels):
    #     df_for_combo = per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization]
    #     if df_for_combo is not None and df_for_combo.shape[1] > 0: # Ensure DF exists and has columns
    #         # Sum horizontally for each hour across all buildings in this combo
    #         # This gives a single Series of hourly sums for this specific (climate_zone, urbanization_level) combo
    #         all_series_for_final_concat.append(df_for_combo.sum(axis=1) / df_for_combo.shape[1])
    #         # (df_for_combo.sum(axis=1) / df_for_combo.shape[1]).plot(kind="box", label=f"{climate_zone}_{level_of_urbanization}", ax=ax)
    
    # # # Ensure directory exists before saving
    # # combined_box_plot_dir = os.path.join(visuals_dir, "combined_box_plot")
    # # os.makedirs(combined_box_plot_dir, exist_ok=True)
    # # combined_box_plot_path = os.path.join(combined_box_plot_dir, "box_plot_per_climate_zone_and_urbanization.svg")

    # # fig.savefig(combined_box_plot_path)
    # # plt.close(fig) # Close the figure to free up memory

    # # Concatenate all these hourly sum series into one large Series for the single box plot
    # fully_combined_series = None
    # if all_series_for_final_concat:
    #     fully_combined_series = pd.concat(all_series_for_final_concat)

    # # 7. Final Figure Creation and Save
    # if fully_combined_series is not None and not fully_combined_series.empty:
    #     fig, ax = plt.subplots(figsize=(8,6))
    #     ax.set_title("Change in hourly electricity usage over the year per residence")
    #     ax.set_ylabel("kWh")

    #     # Plot a single box plot of the combined series
    #     fully_combined_series.plot(kind="box", ax=ax, showfliers=True) # showfliers=False often makes box plots cleaner

    #     # Ensure directory exists before saving
    #     combined_box_plot_dir = os.path.join(visuals_dir, "combined_box_plot")
    #     os.makedirs(combined_box_plot_dir, exist_ok=True)
    #     combined_box_plot_path = os.path.join(combined_box_plot_dir, "box_plot_combining_all_climate_zone_and_urbanization.svg")

    #     fig.savefig(combined_box_plot_path)
    #     plt.close(fig) # Close the figure to free up memory
    # else:
    #     print("No data available to create the final box plot.")