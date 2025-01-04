import os
import pandas as pd

# from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SAVE MODEL RUN RESULTS AND EXPORT TO CSV
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# If bldg_id is now the index in all DataFrames (df_compare, df_results_IRA, and df_results_IRA_gridDecarb):
def clean_df_merge(df_compare, df_results_IRA, df_results_IRA_gridDecarb):
    # Identify common columns (excluding 'bldg_id' which is the merging key)
    common_columns_IRA = set(df_compare.columns) & set(df_results_IRA.columns)
    common_columns_IRA.discard('bldg_id')
        
    # Drop duplicate columns in df_results_IRA and merge
    df_results_IRA = df_results_IRA.drop(columns=common_columns_IRA)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columns_IRA}
    """)
    # merged_df = pd.merge(df_compare, df_results_IRA, on='bldg_id', how='inner')
    merged_df = pd.merge(df_compare, df_results_IRA, how='inner', left_index=True, right_index=True)

    # Repeat the steps above for the merged_df and df_results_IRA_gridDecarb
    common_columnsIRA_gridDecarb = set(merged_df.columns) & set(df_results_IRA_gridDecarb.columns)
    common_columnsIRA_gridDecarb.discard('bldg_id')
    df_results_IRA_gridDecarb = df_results_IRA_gridDecarb.drop(columns=common_columnsIRA_gridDecarb)
    print(f"""Dropped the following duplicate columns before merge: 
    {common_columnsIRA_gridDecarb}
    """)
        
    # Create cleaned, merged results df with no duplicate columns
    # df_results_export = pd.merge(merged_df, df_results_IRA_gridDecarb, on='bldg_id', how='inner')
    df_results_export = pd.merge(merged_df, df_results_IRA_gridDecarb, how='inner', left_index=True, right_index=True)
    print("Dataframes have been cleaned of duplicate columns and merged successfully. Ready to export!")
    return df_results_export

def export_model_run_output(df_results_export, results_category, menu_mp):
    """
    Exports data for results summaries (npv, adoption, impact) and supplemental info (consumption, damages, fuel costs)

    Parameters:
    df_results_export (pd.DataFrame): DataFrame containing data for different scenarios.
    results_category (str): Determines the type of info being exported.
        - Accepted: 'summary', 'consumption', 'damages', 'fuelCost'
    menu_mp (int or str): Determines the measure package or retrofit being conducted
    
    """
    print("-------------------------------------------------------------------------------------------------------")
    # Baseline model run results
    if results_category == 'summary':
        if menu_mp == '0' or menu_mp==0:
            results_filename = f"baseline_results_{location_id}_{results_export_formatted_date}.csv"
            print(f"BASELINE RESULTS:")
            print(f"Dataframe results will be saved in this csv file: {results_filename}")

            # Change the directory to the upload folder and export the file
            results_change_directory = "baseline_summary"

        # Measure Package model run results
        else:
            if menu_mp == '8' or menu_mp==8:
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_basic_summary"

            elif menu_mp == '9' or menu_mp==9:
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_moderate_summary"

            elif menu_mp == '10' or menu_mp==10:
                results_filename = f"mp{menu_mp}_results_{location_id}_{results_export_formatted_date}.csv"
                print(f"MEASURE PACKAGE {menu_mp} (MP{menu_mp}) RESULTS:")
                print(f"Dataframe results will be saved in this csv file: {results_filename}")

                # Change the directory to the upload folder and export the file
                results_change_directory = "retrofit_advanced_summary"

            else:
                print("No matching scenarios for this Measure Package (MP)")

    # This includes exported dataframes for calculated consumption, damages, and fuel costs
    else:
        results_filename = f"mp{menu_mp}_data_{results_category}_{location_id}_{results_export_formatted_date}.csv"
        print(f"SUPPLEMENTAL INFORMATION DATAFRAME: {results_category}")
        print(f"Dataframe results will be saved in this csv file: {results_filename}")

        # Change the directory to the upload folder and export the file
        results_change_directory = f"supplemental_data_{results_category}"

    # Export dataframe results as a csv to the specified filepath
    results_export_filepath = os.path.join(output_folder_path, results_change_directory, results_filename)
    df_results_export.to_csv(results_export_filepath)
    print(f"Dataframe for MP{menu_mp} {results_category} results were exported here: {results_export_filepath}")
    print("-------------------------------------------------------------------------------------------------------", "\n")

