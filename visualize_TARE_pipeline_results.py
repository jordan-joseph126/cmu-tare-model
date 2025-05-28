import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from utils import building_unit_types, adoption_tiers, original_adoption_tiers
import os
import pickle
import numpy as np
import seaborn as sns
from itertools import product
from collections import defaultdict
import math

cwd_filepath = os.getcwd()
# TARE_dir = cwd_filepath
# TARE_dir = os.path.join(os.path.dirname(cwd_filepath), "Trane_Technologies", "cmu-tare-model")
# TARE_dir = os.path.join(os.path.dirname(cwd_filepath), "cmu-tare-model") # For PSC runs
TARE_dir = os.path.join("/ocean", "projects", "eng220005p", "agautam3", "cmu-tare-model") # For PSC runs

# resstock_dir = os.path.join("/mnt", "wsl", "instances", "Ubuntu_24_04", "home", "arnavgautam", "resstock-3.4.0")
resstock_dir = os.path.join("/ocean", "projects", "eng220005p", "agautam3", "resstock-3.4.0") # For PSC runs

visuals_dir = os.path.join("/ocean","projects","eng220005p","agautam3","TPIA-MP", "visuals")

save_not_show = False



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

def get_cost_based_adoption_levels(climate_zone,level_of_urbanization, penetration_level_label):
    # Look into the TARE data to get the number of homes of this type, and figure out the adoption level (out of the applicable homes)

    # Read in selection data from TARE output
    tare_results_for_unit_type_filepath = os.path.join(TARE_dir, "output_results",f"{region}_{NUM_RESIDENCES}_{unit_type}_unit_residence", f"{prefix}_NPV_tare_output.csv")
    tare_results_for_unit_type = pd.read_csv(tare_results_for_unit_type_filepath)
    # tare_results_for_unit_type["Supplemental"] = False

    buildstock_path = os.path.join(resstock_dir, "resources", "national", f"{'all_national' if region == 'national_ASHP' else region}_{NUM_RESIDENCES}_all_unit_buildstock.csv")
    # print(buildstock_path)
    buildstock_df = pd.read_csv(buildstock_path)

    resstock_building_list = buildstock_df['Building']

    print(f"We begin with {tare_results_for_unit_type.shape} TARE results, out of a total of {resstock_building_list.shape} buildings")
    if climate_zone is not None:
        resstock_building_list = buildstock_df[buildstock_df['ASHRAE IECC Climate Zone 2004'] == climate_zone]['Building']
        tare_results_for_unit_type= tare_results_for_unit_type[tare_results_for_unit_type['bldg_id'].isin(resstock_building_list)]
    print(f"Filtering by climate zone leaves us {tare_results_for_unit_type.shape}")

    if level_of_urbanization is not None:
        resstock_building_list = buildstock_df[(buildstock_df['PUMA Metro Status'] == urbanization_level_dict[level_of_urbanization]) & (buildstock_df["Building"].isin(resstock_building_list))]['Building']
        tare_results_for_unit_type= tare_results_for_unit_type[tare_results_for_unit_type['bldg_id'].isin(buildstock_df[buildstock_df['PUMA Metro Status'] == urbanization_level_dict[level_of_urbanization]]['Building'])]
    print(f"Filtering by urbanization leaves us {tare_results_for_unit_type.shape}")

    excluded_buildings = set(resstock_building_list).difference(set(tare_results_for_unit_type["bldg_id"]))
    print(f"There were also {len(excluded_buildings)} buildings excluded from TARE")

    if tare_results_for_unit_type.shape[0] == 0:
        raise Exception(f"Filtering left us with zero residence load profiles to use for populating the network for {climate_zone} {level_of_urbanization}")

    new_df = pd.DataFrame({"run_num_for_load_data": tare_results_for_unit_type["bldg_id"], "is_adopting": False, "NPV": tare_results_for_unit_type["iraRef_mp8_heating_total_npv_moreWTP"]})
    new_df = new_df.set_index("run_num_for_load_data")

    
    for _, row in tare_results_for_unit_type.iterrows():
        bldg_id = row["bldg_id"]
        if penetration_level_label == "1pt0":
            # In this case, we set the flag "is_adopting" for every building with an upgrade scenario
            new_df.loc[bldg_id,"is_adopting"] = True
            continue
        iraRef_mp8_heating_adoption = row["iraRef_mp8_heating_adoption"]
        if iraRef_mp8_heating_adoption in ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility', 'Feasible per MMPV']:
            new_df.loc[bldg_id,"is_adopting"] = True
    
    return new_df, excluded_buildings


csv_format_phased_power_dict = {}

for penetration_level_label in ["COST_BASED"]:

    region = "national_ASHP"
    NUM_RESIDENCES = 8000
    unit_type = "all"
    prefix="alpha_beta"
    tare_results_for_unit_type_filepath = os.path.join(TARE_dir, "output_results",f"{region}_{NUM_RESIDENCES}_{unit_type}_unit_residence", f"{prefix}_NPV_tare_output.csv")
    tare_results_for_unit_type = pd.read_csv(tare_results_for_unit_type_filepath)

    # buildstock_path = os.path.join(resstock_dir, "resources", "national", f"{'all_national' if region == 'national_ASHP' else region}_{NUM_RESIDENCES}_all_unit_buildstock.csv")
    # # print(buildstock_path)
    # buildstock_df = pd.read_csv(buildstock_path)

    # resstock_building_list = buildstock_df['Building']

    # print(f"We begin with {tare_results_for_unit_type.shape} TARE results, out of a total of {resstock_building_list.shape} buildings")
    # if climate_zone is not None:
    #     resstock_building_list = buildstock_df[buildstock_df['ASHRAE IECC Climate Zone 2004'] == climate_zone]['Building']
    #     tare_results_for_unit_type= tare_results_for_unit_type[tare_results_for_unit_type['bldg_id'].isin(resstock_building_list)]
    # print(f"Filtering by climate zone leaves us {tare_results_for_unit_type.shape}")

    # if level_of_urbanization is not None:
    #     resstock_building_list = buildstock_df[(buildstock_df['PUMA Metro Status'] == urbanization_level_dict[level_of_urbanization]) & (buildstock_df["Building"].isin(resstock_building_list))]['Building']
    #     tare_results_for_unit_type= tare_results_for_unit_type[tare_results_for_unit_type['bldg_id'].isin(buildstock_df[buildstock_df['PUMA Metro Status'] == urbanization_level_dict[level_of_urbanization]]['Building'])]
    # print(f"Filtering by urbanization leaves us {tare_results_for_unit_type.shape}")

    new_df = pd.DataFrame({"run_num_for_load_data": tare_results_for_unit_type["bldg_id"], "is_adopting": False, "NPV": tare_results_for_unit_type["iraRef_mp8_heating_total_npv_moreWTP"], "iraRef_mp8_heating_adoption": tare_results_for_unit_type["iraRef_mp8_heating_adoption"]})
    new_df = new_df.set_index("run_num_for_load_data")
    new_df.sort_values(by="NPV", axis=0, ascending=False, inplace=True)
    
    for idx, row in new_df.iterrows():
        bldg_id = row.name
        # print(bldg_id)

        if penetration_level_label == "1pt0":
            # In this case, we set the flag "is_adopting" for every building with an upgrade scenario
            new_df.loc[bldg_id,"is_adopting"] = True
            continue
        
        iraRef_mp8_heating_adoption = row["iraRef_mp8_heating_adoption"]
        if iraRef_mp8_heating_adoption in ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility', 'Feasible per MMPV']:
            new_df.loc[bldg_id,"is_adopting"] = True

    buildstock_path = os.path.join(resstock_dir, "resources", "national", f"{'all_national' if region == 'national_ASHP' else region}_{NUM_RESIDENCES}_all_unit_buildstock.csv")
    # print(buildstock_path)
    buildstock_df = pd.read_csv(buildstock_path)
    resstock_building_list = buildstock_df['Building']

    excluded_buildings = set(resstock_building_list).difference(set(tare_results_for_unit_type["bldg_id"]))

    # num_nonzero_buildings = math.ceil(new_df.shape[0] * penetration_level)

    # # Exclude additional buildings based on the NPV, to force the penetration level
    # excluded_buildings.update(new_df.index[num_nonzero_buildings:])

    electricity_cols = ["Fuel Use: Electricity: Total"]
    POWER_FACTOR = 0.9

    per_climate_zone_and_urbanization_dfs = defaultdict(lambda: defaultdict(lambda: None))

    # Make it into the format that TPIA-MP expects
    csv_format_phased_power = None
    for building_num, (idx, row) in enumerate(new_df.iterrows()):
        if np.isnan(row.name):
            continue
        if not row['is_adopting']:
            continue

        # if building_num == 5:
        #     break
        
        bldg_id = int(row.name)
        baseline_load_file_path = os.path.join(resstock_dir, f"{region}{'' if '_' in region else '_upgrades'}_{NUM_RESIDENCES}_{unit_type}_unit_residence", f"run{bldg_id}", "run", "results_timeseries.csv")
        upgrade_load_file_path = os.path.join(resstock_dir, f"{region}{'' if '_' in region else '_upgrades'}_{NUM_RESIDENCES}_{unit_type}_unit_residence_ASHP", f"run{bldg_id}", "run", "results_timeseries.csv")
        if not os.path.exists(baseline_load_file_path):
            print(f"Couldn't find baseline data for building {bldg_id}, not adding it")
            continue
        baseline_power_timeseries_df = pd.read_csv(baseline_load_file_path, low_memory=True, usecols=lambda x: x in electricity_cols)
        baseline_power_timeseries_df = baseline_power_timeseries_df.iloc[1:].copy() # Remove the title from the series
        baseline_power_timeseries_df[str(int(idx))] = baseline_power_timeseries_df["Fuel Use: Electricity: Total"].astype(float).apply(lambda power: power / POWER_FACTOR) #   / 3 / 0.9
        baseline_power_timeseries_df = baseline_power_timeseries_df[[str(int(idx))]]
        if not os.path.exists(upgrade_load_file_path):
            print(f"Couldn't find upgrade data for building {bldg_id}, so adding a zero sequence for it")
            power_timeseries_df = pd.DataFrame(index=baseline_power_timeseries_df, columns=baseline_power_timeseries_df.columns)
            power_timeseries_df.iloc[:,0] = 0
        else:
            upgrade_power_timeseries_df = pd.read_csv(upgrade_load_file_path, low_memory=True, usecols=lambda x: x in electricity_cols)
            upgrade_power_timeseries_df = upgrade_power_timeseries_df.iloc[1:].copy() # Remove the title from the series
            upgrade_power_timeseries_df[str(int(idx))] = upgrade_power_timeseries_df["Fuel Use: Electricity: Total"].astype(float).apply(lambda power: power / POWER_FACTOR) #   / 3 / 0.9
            upgrade_power_timeseries_df = upgrade_power_timeseries_df[[str(int(idx))]]
            power_timeseries_df = upgrade_power_timeseries_df - baseline_power_timeseries_df
        
        print(power_timeseries_df)
        
        # if csv_format_phased_power is None:
        #     csv_format_phased_power = power_timeseries_df
        # else:
        #     csv_format_phased_power = csv_format_phased_power.join(power_timeseries_df, how="left")
        
        climate_zone = buildstock_df[buildstock_df['Building'] == bldg_id]['ASHRAE IECC Climate Zone 2004'].iloc[0]
        level_of_urbanization = buildstock_df[buildstock_df['Building'] == bldg_id]['PUMA Metro Status'].apply(lambda x: level_of_urbanization_dict[x]).iloc[0]

        print(f"{climate_zone}_{level_of_urbanization}")

        if per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization] is None:
            per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization] = power_timeseries_df
        else:
            per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization] = per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization].join(power_timeseries_df, how="left")


    # fig = plt.figure(figsize=(12,4))
    # plt.plot(csv_format_phased_power.sum(axis=1), label='Change from Baseline')
    # plt.title(f"Change in Total Hourly Usage Over the Year for the 8000 residence sample")
    # plt.xlabel("Hour")
    # plt.ylabel("kWh")
    # plt.legend()
    # plt.tight_layout()
    
    # line_plot_path = os.path.join(visuals_dir, "line_plot", penetration_level_label, f"all_national_8000.svg")
    # os.makedirs(os.path.dirname(line_plot_path), exist_ok=True)
    # fig.savefig(line_plot_path)
    # plt.close()

    cwd_filepath = os.getcwd()
    TARE_dir = cwd_filepath
    # TARE_dir = os.path.join(os.path.dirname(cwd_filepath), "Trane_Technologies", "cmu-tare-model")
    # TARE_dir = os.path.join(cwd_filepath, "cmu-tare-model") # For PSC runs
    TARE_dir = os.path.join("/ocean", "projects", "eng220005p", "agautam3", "cmu-tare-model")

    # resstock_dir = os.path.join("/mnt", "wsl", "instances", "Ubuntu_24_04", "home", "arnavgautam", "resstock-3.4.0")
    resstock_dir = os.path.join("/ocean", "projects", "eng220005p", "agautam3", "resstock-3.4.0") # For PSC runs

    visuals_dir = os.path.join("/ocean","projects","eng220005p","agautam3","TPIA-MP", "visuals")

    climate_zones = ["1A","2A","2B","3A","3B","3C","4A","4B","4C","5A","5B","6A","6B","7A","7B"] # 

    combinations_array = product(climate_zones, urbanization_level_dict.keys()) # , "1pt0"

    region = "national_ASHP"
    NUM_RESIDENCES = 8000
    unit_type = "all"
    prefix="alpha_beta"

    # Prepare a dict to hold all Series
    combined_box_data = {}
    for climate_zone, level_of_urbanization in combinations_array:
        df = per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization]
        if df is not None:
            label = f"{climate_zone}_{level_of_urbanization}"
            combined_box_data[label] = df.sum(axis=1) / df.shape[1]

    # Convert to DataFrame
    combined_df = pd.DataFrame(combined_box_data)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    combined_df.boxplot(ax=ax, rot=90)  # rot rotates x-axis labels to avoid overlap
    ax.set_title("Change in Total Hourly Usage Over the Year, Normalized per Residence")
    ax.set_ylabel("kWh")
    plt.tight_layout()

    combined_box_plot_path = os.path.join(visuals_dir, "combined_box_plot", "box_plot_per_climate_zone_and_urbanization_normalized_per_residence.svg")
    os.makedirs(os.path.dirname(combined_box_plot_path), exist_ok=True)
    fig.savefig(combined_box_plot_path)
    plt.close()

    # fig, ax = plt.subplots(figsize=(12,4))
    # for climate_zone, level_of_urbanization in combinations_array:
    #     if per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization] is not None:
    #         print(per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization].sum(axis=1))
    #         per_climate_zone_and_urbanization_dfs[climate_zone][level_of_urbanization].sum(axis=1).plot(kind="box", ax=ax, label=f"{climate_zone}_{level_of_urbanization}")
    # combined_box_plot_path = os.path.join(visuals_dir, "combined_box_plot", "box_plot_per_climate_zone_and_urbanization.svg")
    # os.makedirs(os.path.dirname(combined_box_plot_path), exist_ok=True)
    # fig.savefig(combined_box_plot_path)
    # plt.close()






