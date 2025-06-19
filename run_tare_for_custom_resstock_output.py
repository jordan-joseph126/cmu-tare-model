import os
import sys
from importnb import Notebook
from utils import building_unit_types

project_root = os.path.abspath(os.getcwd())
policy_scenarios = ["No Inflation Reduction Act", "AEO2023 Reference Case"]


with Notebook():
    import tare_model_IO_functions_v1_4_1 as TARE_IO
    # Gather data
    hdd_factor_lookup = TARE_IO.load_hdd_factors(project_root)
    cdd_factor_lookup = TARE_IO.load_cdd_factors(project_root)
    emis_preIRA_cambium21_lookup, emis_IRA_2024_cambium22_lookup, emis_IRA_2025_2050_cambium23_lookup = TARE_IO.load_cambium_lookup(project_root)
    emis_factor_co2e_naturalGas_ton_perkWh, emis_factor_co2e_propane_ton_perkWh, emis_factor_co2e_fuelOil_ton_perkWh = TARE_IO.load_emis_factors()
    cpi_ratio_2023_2023, cpi_ratio_2023_2022, cpi_ratio_2023_2021, cpi_ratio_2023_2020, cpi_ratio_2023_2019, cpi_ratio_2023_2018, cpi_ratio_2023_2013, cpi_ratio_2023_2010, cpi_ratio_2023_2008 = TARE_IO.load_cpi_data(project_root)
    epa_scc_usd2023_per_ton = TARE_IO.load_scc(cpi_ratio_2023_2020)
    preIRA_fuel_price_lookup, iraRef_fuel_price_lookup = TARE_IO.load_fuel_price_lookups(project_root, cpi_ratio_2023_2018, cpi_ratio_2023_2019, cpi_ratio_2023_2020, cpi_ratio_2023_2021, cpi_ratio_2023_2022)
    rsMeans_national_avg = TARE_IO.load_rsMeans_national_avg(cpi_ratio_2023_2019)
    dict_heating_equipment_cost = TARE_IO.load_dict_heating_equipment_cost(project_root)
    dict_cooling_equipment_cost = TARE_IO.load_dict_cooling_equipment_cost(project_root)
    df_puma_medianIncome = TARE_IO.load_df_puma_medianIncome(project_root, cpi_ratio_2023_2022)
    df_county_medianIncome = TARE_IO.load_df_county_medianIncome(project_root, cpi_ratio_2023_2022)
    df_state_medianIncome = TARE_IO.load_df_state_medianIncome(project_root, cpi_ratio_2023_2022)

    import tare_model_functions_v1_4_1 as base_TARE

NUM_RESIDENCES = 8000
PUBLIC_INTEREST_RATE = 0.02


ALPHA = 0.2
BETA = 0.3
MMPV_DISCOUNT_RATE = 0.07
USING_MMPV = False
MMPV_filename = 'MMPV_ALPHA0pt2_BETA0pt3_DISCOUNT0pt07'

unit_num = "all"
# region = "urban_ohio"

for region in ["national_ASHP"]:
    output_filepath = os.path.join("output_results",f"{region}_{NUM_RESIDENCES}_all_unit_residence_DEBUG", f"alpha_beta_{MMPV_filename if USING_MMPV else 'NPV'}_tare_output.csv")

    high_level_logfile_path =  os.path.join("output_results",f"{region}_{NUM_RESIDENCES}_all_unit_residence_DEBUG", "LOGS.log")

    real_original_stdout = sys.stdout

    os.makedirs(os.path.dirname(high_level_logfile_path), exist_ok=True)

    log_file = open(high_level_logfile_path, 'w')

    # print(f"switching logging to {high_level_logfile_path}")
    sys.stdout = log_file


    menu_mp = 0
    input_mp = "baseline"
    # data_folder_file_path = os.path.join("/ocean","projects", "eng220005p", "agautam3","resstock-3.4.0",f"{region}_{NUM_RESIDENCES}_{unit_num}_unit_residence")
    data_folder_file_path = os.path.join("/home","arnavgautam","resstock-3.4.0",f"{region}_{NUM_RESIDENCES}_{unit_num}_unit_residence")
    print(data_folder_file_path)

    # Load the annual metadata associated with this case. This will be loading my custom data format from my own ResStock runs
    df_resstock_run_am = TARE_IO.load_multiple_resstock_run_annual_metadata(data_folder_file_path, menu_mp, NUM_RESIDENCES)

    df_resstock_run_am = base_TARE.project_future_consumption(df=df_resstock_run_am, hdd_factor_lookup=hdd_factor_lookup, cdd_factor_lookup=cdd_factor_lookup, menu_mp=menu_mp)

    for policy_scenario in policy_scenarios:
        df_resstock_run_am = base_TARE.calculate_marginal_damages(df=df_resstock_run_am, menu_mp=menu_mp, policy_scenario=policy_scenario, df_summary=df_resstock_run_am)

    for policy_scenario in policy_scenarios:
        df_resstock_run_am = base_TARE.calculate_annual_fuelCost(df=df_resstock_run_am, menu_mp=menu_mp, policy_scenario=policy_scenario, drop_fuel_cost_columns=False)

    # Upgrade scenario
    menu_mp = 8
    input_mp = "default_option_closest_to_sales_volume_weighted_heat_pump_device"
    # data_folder_file_path = os.path.join("/ocean","projects", "eng220005p", "agautam3","resstock-3.4.0",f"{region}_{NUM_RESIDENCES}_{unit_num}_unit_residence_ASHP")
    data_folder_file_path = os.path.join("/home", "arnavgautam","resstock-3.4.0",f"{region}_{NUM_RESIDENCES}_{unit_num}_unit_residence_ASHP")
    print(data_folder_file_path)

    # Load the annual metadata associated with this case. This will be loading my custom data format from my own ResStock runs
    try:
        print("start loading data from upgrades")
        df_resstock_run_am_mp8 = TARE_IO.load_multiple_resstock_run_annual_metadata(data_folder_file_path, menu_mp, NUM_RESIDENCES, baseline_data=df_resstock_run_am)
    except Exception as e:
        print("This does not apply to this building because:", e)
        exit() #continue

    df_resstock_run_am_mp8 = base_TARE.project_future_consumption(df=df_resstock_run_am_mp8, hdd_factor_lookup=hdd_factor_lookup, cdd_factor_lookup=cdd_factor_lookup, menu_mp=menu_mp)

    for policy_scenario in policy_scenarios:
        df_resstock_run_am_mp8 = base_TARE.calculate_marginal_damages(df=df_resstock_run_am_mp8, menu_mp=menu_mp, policy_scenario=policy_scenario, df_summary=df_resstock_run_am)

    for policy_scenario in policy_scenarios:
        df_resstock_run_am_mp8 = base_TARE.calculate_annual_fuelCost(df=df_resstock_run_am_mp8, menu_mp=menu_mp, policy_scenario=policy_scenario, drop_fuel_cost_columns=False)

    df_resstock_run_am_mp8 = base_TARE.obtain_heating_system_specs(df_resstock_run_am_mp8)

    df_resstock_run_am_mp8 = TARE_IO.load_rsMeans_CCI_values(project_root, df_resstock_run_am_mp8)

    # calculate_installation_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
    print("Calculating Cost of Retrofit Upgrade: Heat Pump for Space Heating (No Enclosure Upgrade) ...")
    df_resstock_run_am_mp8 = base_TARE.calculate_installation_cost(df_resstock_run_am_mp8, dict_heating_equipment_cost, rsMeans_national_avg, menu_mp, 'heating')

    # calculate_replacement_cost(df, cost_dict, rsMeans_national_avg, menu_mp, end_use)
    print("Calculating Cost of Replacing Existing Equipment with Similar Model/Efficiency ...")
    df_resstock_run_am_mp8 = base_TARE.calculate_replacement_cost(df_resstock_run_am_mp8, dict_heating_equipment_cost, rsMeans_national_avg, menu_mp, 'heating')
    df_resstock_run_am_mp8 = base_TARE.calculate_replacement_cost(df_resstock_run_am_mp8, dict_cooling_equipment_cost, rsMeans_national_avg, menu_mp, 'cooling')

    # Call the function and calculate installation premium based on existing housing characteristics
    # calculate_heating_installation_premium(df, rsMeans_national_avg, cpi_ratio_2023_2013)
    print("Calculating Space Heating Specific Premiums (Ex: Removing Hydronic Boiler) ...")
    df_resstock_run_am_mp8 = base_TARE.calculate_heating_installation_premium(df_resstock_run_am_mp8, menu_mp, rsMeans_national_avg, cpi_ratio_2023_2013)

    df_resstock_run_am_mp8 = base_TARE.calculate_percent_AMI(df_resstock_run_am_mp8)

    df_resstock_run_am_mp8 = base_TARE.calculate_rebateIRA(df_resstock_run_am_mp8, "heating", menu_mp)

    for policy_scenario in policy_scenarios:
        df_resstock_run_am_mp8 = base_TARE.calculate_public_npv(df_resstock_run_am_mp8, df_resstock_run_am_mp8, menu_mp, policy_scenario, interest_rate=PUBLIC_INTEREST_RATE)

    for policy_scenario in policy_scenarios:
        df_resstock_run_am_mp8 = base_TARE.calculate_MMPV(df_resstock_run_am_mp8, df_resstock_run_am_mp8, MMPV_DISCOUNT_RATE, ALPHA, BETA, input_mp, menu_mp, policy_scenario)    

    for policy_scenario in policy_scenarios:
        df_resstock_run_am_mp8 = base_TARE.adoption_decision(df_resstock_run_am_mp8, policy_scenario, menu_mp, using_MMPV=USING_MMPV)


    print(f"Writing output to {output_filepath}")
    TARE_IO.write_TARE_results(df_resstock_run_am_mp8, output_filepath)

    sys.stdout = real_original_stdout
    log_file.close()
