def check_ira_adoption_consistency(df, category, scenario_prefix, mer_type, year_label, upgrade_column, cost_benefit_type):
    """
    df (pd.DataFrame): DataFrame containing data for different scenarios.
    category (str): Determines the type of info being exported.
        - Accepted: 'heating', 'waterHeating', 'clothesDrying', 'cooking'
    scenario_prefix (str): Determines the scenario prefix for the IRA scenarios.
        - Accepted: 'baseline_', 'preIRA_mp{menu_mp}', 'iraRef_mp{menu_mp}'
    upgrade_column (str): Determines the column name for the upgrade measure.
        - Accepted: 'upgrade_hvac_heating_efficiency', 'upgrade_water_heater_efficiency', 'upgrade_clothes_dryer', 'upgrade_cooking_range'
    mer_type (str): Determines the type of cost-benefit analysis.
        - Accepted: 'lrmer', 'srmer'
    year_label (str): Determines the year label for the scenario.
    cost_benefit_type (str): Determines the type of cost-benefit analysis.
        - Accepted: 'private', 'public'

    """
    # Copy the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    df_cols = [
        f'base_{category}_fuel',
        f'{upgrade_column}',
        f'baseline_{year_label}_{category}_consumption',
        f'mp{menu_mp}_{year_label}_{category}_consumption',
        f'mp{menu_mp}_{year_label}_{category}_reduction_consumption',
        ]
    # 
    public_cols = [
        # f'baseline_{year_label}_{category}_damages_health',
        # f'baseline_{year_label}_{category}_damages_climate_{mer_type}',
        # f'{scenario_prefix}{year_label}_{category}_damages_health',
        # f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}',
        f'{scenario_prefix}{category}_health_npv',
        f'{scenario_prefix}{category}_climate_npv_{mer_type}',
        f'{scenario_prefix}{category}_public_npv_{mer_type}',
        f'{scenario_prefix}{category}_retrofit_publicImpact_{mer_type}',
        f'{scenario_prefix}{category}_total_npv_lessWTP_{mer_type}',
        f'{scenario_prefix}{category}_total_npv_moreWTP_{mer_type}',
        f'{scenario_prefix}{category}_adoption_{mer_type}',
        ]

    private_cols = [
        # f'baseline_{year_label}_{category}_fuelCost',
        # f'{scenario_prefix}{category}_{year_label}_fuelCost',        
        # f'{scenario_prefix}{category}_{year_label}_savings_fuelCost',
        f'{scenario_prefix}{category}_total_capitalCost',
        f'{scenario_prefix}{category}_net_capitalCost',
        f'{scenario_prefix}{category}_private_npv_lessWTP',
        f'{scenario_prefix}{category}_private_npv_moreWTP',
        f'{scenario_prefix}{category}_private_npv_lessWTP',
        f'{scenario_prefix}{category}_private_npv_moreWTP',
        f'{scenario_prefix}{category}_total_npv_lessWTP_{mer_type}',
        f'{scenario_prefix}{category}_total_npv_moreWTP_{mer_type}',
        f'{scenario_prefix}{category}_adoption_{mer_type}',
        ]
    
    # Select the relevant columns based on the cost_benefit_type
    if cost_benefit_type == 'public':
        cols_to_display = df_cols + public_cols
    elif cost_benefit_type == 'private':
        cols_to_display = df_cols + private_cols
    else:
        raise ValueError("Invalid cost_benefit_type! Please choose from 'private' or 'public'.")
        
    # Filter the dataframe to show only the relevant columns
    df_filtered = df_copy[cols_to_display]

    return df_filtered

check_ira_adoption_consistency(df=df_euss_am_mp8_home,
                               category='heating',
                               scenario_prefix=f'iraRef_mp{menu_mp}_',
                               mer_type='lrmer',
                               year_label='2024',
                               upgrade_column='upgrade_hvac_heating_efficiency',
                               cost_benefit_type='private'
                               )