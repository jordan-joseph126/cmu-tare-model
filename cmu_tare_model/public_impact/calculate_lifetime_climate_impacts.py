import pandas as pd

from cmu_tare_model.constants import MER_TYPES, EQUIPMENT_SPECS, TD_LOSSES_MULTIPLIER, EPA_SCC_USD2023_PER_MT_LOW, EPA_SCC_USD2023_PER_MT_BASE, EPA_SCC_USD2023_PER_MT_HIGH
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions

def calculate_climate_impacts(df, menu_mp, scenario_prefix, lookup_emissions_electricity_climate,
                              cambium_scenario, hdd_factors_df, lookup_emissions_fossil_fuel,
                              df_baseline_damages=None):
    """
    Calculate lifetime climate impacts (emissions and damages) for each equipment category.
    
    Args:
        df (pd.DataFrame): Input data.
        menu_mp (int): Measure package identifier.
        scenario_prefix (str): Prefix for output column names.
        lookup_emissions_electricity_climate (dict): Lookup for electricity emissions factors.
        cambium_scenario (str): Scenario label.
        hdd_factors_df (dict): HDD adjustment factors by year.
        lookup_emissions_fossil_fuel (dict): Lookup for fossil fuel emissions factors.
        df_baseline_damages (pd.DataFrame, optional): Baseline data for avoided damages.
    
    Returns:
        pd.DataFrame: A DataFrame with lifetime climate columns (rounded to 2 decimals).
    """
    df_copy = df.copy()
    df_detailed = pd.DataFrame(index=df_copy.index)
    
    # Initialize accumulators for lifetime climate values
    lifetime_climate_emissions = {mer: pd.Series(0.0, index=df_copy.index) for mer in MER_TYPES}
    # For damages, store separate accumulation for each (mer, scc_assumption)
    lifetime_climate_damages = { (mer, scc): pd.Series(0.0, index=df_copy.index)
                                 for mer in MER_TYPES for scc in ['low', 'base', 'high'] }
    
    for category, lifetime in EQUIPMENT_SPECS.items():
        for year in range(1, lifetime + 1):
            year_label = year + 2023
            # Only adjust HDD factors for relevant categories
            hdd_factor = hdd_factors_df.get(year_label, pd.Series(1.0, index=df_copy.index))
            adjusted_hdd_factor = hdd_factor if category in ['heating', 'waterHeating'] else pd.Series(1.0, index=df_copy.index)
            
            # Calculate fossil fuel emissions first
            total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                df_copy, category, adjusted_hdd_factor, lookup_emissions_fossil_fuel, menu_mp
            )
            
            # Compute climate emissions and damages with SCC sensitivities
            climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
                df=df_copy,
                category=category,
                year_label=year_label,
                adjusted_hdd_factor=adjusted_hdd_factor,
                lookup_emissions_electricity_climate=lookup_emissions_electricity_climate,
                cambium_scenario=cambium_scenario,
                total_fossil_fuel_emissions=total_fossil_fuel_emissions,
                scenario_prefix=scenario_prefix,
                menu_mp=menu_mp
            )
            
            # Accumulate annual emissions (SCC-independent)
            for mer in MER_TYPES:
                lifetime_climate_emissions[mer] += annual_emissions.get(mer, 0.0)
            # Accumulate annual damages for each SCC assumption
            for key, value in annual_damages.items():
                lifetime_climate_damages[key] += value

            if climate_results:
                df_detailed = pd.concat([df_detailed, pd.DataFrame(climate_results, index=df_copy.index)], axis=1)
        
        # Create lifetime columns for climate impacts
        lifetime_dict = {}
        for mer in MER_TYPES:
            emissions_col = f'{scenario_prefix}{category}_lifetime_tons_co2e_{mer}'
            lifetime_dict[emissions_col] = lifetime_climate_emissions[mer]
            
            for scc_assumption in ['low', 'base', 'high']:
                damages_col = f'{scenario_prefix}{category}_lifetime_damages_climate_{mer}_{scc_assumption}'
                lifetime_dict[damages_col] = lifetime_climate_damages[(mer, scc_assumption)]
                # Calculate avoided damages if baseline data is provided
                if menu_mp != 0 and df_baseline_damages is not None:
                    baseline_damages_col = f'baseline_{category}_lifetime_damages_climate_{mer}_{scc_assumption}'
                    avoided_damages_col = f'{scenario_prefix}{category}_avoided_damages_climate_{mer}_{scc_assumption}'
                    lifetime_dict[avoided_damages_col] = df_baseline_damages[baseline_damages_col] - lifetime_dict[damages_col]
            # Calculate avoided emissions if baseline data is provided
            if menu_mp != 0 and df_baseline_damages is not None:
                baseline_emissions_col = f'baseline_{category}_lifetime_tons_co2e_{mer}'
                avoided_emissions_col = f'{scenario_prefix}{category}_avoided_tons_co2e_{mer}'
                lifetime_dict[avoided_emissions_col] = df_baseline_damages[baseline_emissions_col] - lifetime_dict[emissions_col]
        
        df_detailed = pd.concat([df_detailed, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)
    
    # Apply rounding only at the final stage
    df_detailed = df_detailed.round(2)
    return df_detailed


def calculate_climate_emissions_and_damages(
    df,
    category,
    year_label,
    adjusted_hdd_factor,
    lookup_emissions_electricity_climate,
    cambium_scenario,
    total_fossil_fuel_emissions,
    scenario_prefix,
    menu_mp
):
    """Calculate climate-related emissions and damages for a given category and year.

    Args:
        df (pd.DataFrame): DataFrame containing base or measure-package consumption data and region info.
        category (str): Equipment category (e.g., 'heating', 'waterHeating').
        year_label (int): The year offset from 2023 (e.g., 2024 = 1 + 2023).
        adjusted_hdd_factor (pd.Series): Heating degree-day adjustment factors for the specified category.
        lookup_emissions_electricity_climate (dict): Lookup for electricity-based CO2e emissions factors.
        cambium_scenario (str): Label identifying the Cambium scenario for emissions lookups.
        total_fossil_fuel_emissions (dict): Mapping of pollutant -> pd.Series, from fossil fuel usage.
        scenario_prefix (str): Prefix for output column naming.
        menu_mp (int): Measure package identifier.

    Returns:
        tuple:
            - dict: Annual climate emissions and damages columns (by year).
            - dict: Annual climate emissions by MER type for aggregation.
            - dict: Annual climate damages by MER type and SCC assumption for aggregation.
              (Keys are tuples of the form (mer_type, scc_assumption).)
    """
    # Results dictionaries (without rounding)
    climate_results = {}
    annual_climate_emissions = {}
    annual_climate_damages = {}
    
    total_fossil_fuel_emissions_co2e = total_fossil_fuel_emissions['co2e']
    
    # Define vectorized helper functions for LRMER/SRMER emission factor lookups
    def get_emission_factor_lrmer(region):
        return lookup_emissions_electricity_climate.get(
            (cambium_scenario, region), {}
        ).get(year_label, {}).get('lrmer_ton_per_kWh_co2e', 0)
    
    def get_emission_factor_srmer(region):
        return lookup_emissions_electricity_climate.get(
            (cambium_scenario, region), {}
        ).get(year_label, {}).get('srmer_ton_per_kWh_co2e', 0)
    
    mer_factors = {
        'lrmer': df['gea_region'].map(get_emission_factor_lrmer),
        'srmer': df['gea_region'].map(get_emission_factor_srmer)
    }
    
    # Determine electricity consumption column based on menu_mp and apply the HDD adjustment factor
    if menu_mp == 0:
        electricity_consumption = df.get(f'base_electricity_{category}_consumption', pd.Series(0.0, index=df.index)).fillna(0)
        electricity_consumption *= adjusted_hdd_factor
    else:
        consumption_col = f'mp{menu_mp}_{year_label}_{category}_consumption'
        electricity_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)
    
    # Calculate annual emissions and damages for each MER type
    for mer_type in MER_TYPES:
        annual_emissions_electricity = electricity_consumption * TD_LOSSES_MULTIPLIER * mer_factors[mer_type]
        
        # Sum fossil fuel CO2e and electricity CO2e
        total_annual_climate_emissions = total_fossil_fuel_emissions_co2e + annual_emissions_electricity
        
        # Store emissions results
        emissions_col = f'{scenario_prefix}{year_label}_{category}_tons_co2e_{mer_type}'
        climate_results[emissions_col] = total_annual_climate_emissions
        annual_climate_emissions[mer_type] = total_annual_climate_emissions
        
        # SCC Sensitivity Analysis: store each sensitivity result separately.
        for scc_assumption in ['low', 'base', 'high']:
            if scc_assumption == 'low':
                scc = EPA_SCC_USD2023_PER_MT_LOW
            elif scc_assumption == 'base':
                scc = EPA_SCC_USD2023_PER_MT_BASE
            elif scc_assumption == 'high':
                scc = EPA_SCC_USD2023_PER_MT_HIGH
            else:
                raise ValueError(f"Invalid SCC assumption: {scc_assumption}")
            
            damages_col = f'{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}_{scc_assumption}'
            total_annual_climate_damages = total_annual_climate_emissions * scc
            climate_results[damages_col] = total_annual_climate_damages
            # Store each sensitivity result keyed by (mer_type, scc_assumption)
            annual_climate_damages[(mer_type, scc_assumption)] = total_annual_climate_damages
    
    return climate_results, annual_climate_emissions, annual_climate_damages