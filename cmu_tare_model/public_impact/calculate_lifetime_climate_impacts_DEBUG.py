import pandas as pd

from cmu_tare_model.constants import (
    MER_TYPES,
    EQUIPMENT_SPECS,
    TD_LOSSES_MULTIPLIER,
    EPA_SCC_USD2023_PER_MT_LOW,
    EPA_SCC_USD2023_PER_MT_BASE,
    EPA_SCC_USD2023_PER_MT_HIGH,
)
from cmu_tare_model.public_impact.calculations.calculate_fossil_fuel_emissions import calculate_fossil_fuel_emissions
from cmu_tare_model.public_impact.emissions_scenario_settings import define_scenario_settings
from cmu_tare_model.public_impact.calculations.precompute_hdd_factors import precompute_hdd_factors


def calculate_climate_impacts(df, menu_mp, policy_scenario, df_baseline_damages=None, debug=False):
    """Calculate lifetime climate impacts (emissions and damages) for each equipment category.

    This function computes aggregated lifetime emissions and damages (as well as avoided impacts if baseline data
    is provided) and returns two DataFrames: one with the main aggregated results and one with detailed annual breakdowns.
    """
    # Create a copy of the input DataFrame and initialize the detailed DataFrame.
    df_copy = df.copy()
    # Keep a small example of the input if debugging:
    if debug:
        print("DEBUG: Input DataFrame (head):")
        print(df_copy.head())
    df_detailed = pd.DataFrame(index=df_copy.index)

    # Initialize a dictionary to store lifetime climate impacts columns.
    lifetime_columns_data = {}

    # Retrieve scenario-specific settings.
    # Unpack all 5 values and ignore the last one (unused in this function).
    scenario_prefix, cambium_scenario, lookup_emissions_fossil_fuel, lookup_emissions_electricity_climate, _ = define_scenario_settings(
        menu_mp, policy_scenario
    )
    if debug:
        print("DEBUG: Scenario settings:")
        print("  scenario_prefix:", scenario_prefix)
        print("  cambium_scenario:", cambium_scenario)

    # Precompute HDD adjustment factors by region and year.
    hdd_factors_per_year = precompute_hdd_factors(df_copy)
    if debug:
        print("DEBUG: HDD factors (head):")
        print(hdd_factors_per_year.head())

    # Loop over each equipment category and its lifetime.
    for category, lifetime in EQUIPMENT_SPECS.items():
        # Initialize accumulators for lifetime emissions and damages.
        lifetime_climate_emissions = {mer: pd.Series(0.0, index=df_copy.index) for mer in MER_TYPES}
        lifetime_climate_damages = {
            (mer, scc_value): pd.Series(0.0, index=df_copy.index)
            for mer in MER_TYPES
            for scc_value in ["low", "base", "high"]
        }

        # Loop over each year in the equipment's lifetime.
        for year in range(1, lifetime + 1):
            year_label = year + 2023

            # Retrieve HDD factor for the current year; use 1.0 if missing.
            hdd_factor = hdd_factors_per_year.get(year_label, pd.Series(1.0, index=df_copy.index))
            adjusted_hdd_factor = hdd_factor if category in ["heating", "waterHeating"] else pd.Series(1.0, index=df_copy.index)
            # COMMENTED OUT: Avoid printing repeatedly for each year
            # if debug:
            #     print(f"DEBUG: Category '{category}', Year {year_label}")
            #     print("  adjusted_hdd_factor (first 5 rows):", adjusted_hdd_factor.head())

            # Calculate fossil fuel emissions for the current category and year.
            total_fossil_fuel_emissions = calculate_fossil_fuel_emissions(
                df_copy, category, adjusted_hdd_factor, lookup_emissions_fossil_fuel, menu_mp
            )
            # COMMENTED OUT: Large repeated debug
            # if debug:
            #     print(f"DEBUG: Total fossil fuel 'co2e' emissions for '{category}' year {year_label} (first 5 rows):")
            #     print(total_fossil_fuel_emissions.get('co2e', pd.Series()).head())

            # Compute climate emissions and damages with SCC value sensitivities.
            climate_results, annual_emissions, annual_damages = calculate_climate_emissions_and_damages(
                df=df_copy,
                category=category,
                year_label=year_label,
                adjusted_hdd_factor=adjusted_hdd_factor,
                lookup_emissions_electricity_climate=lookup_emissions_electricity_climate,
                cambium_scenario=cambium_scenario,
                total_fossil_fuel_emissions=total_fossil_fuel_emissions,
                scenario_prefix=scenario_prefix,
                menu_mp=menu_mp,
                debug=debug,
            )
            # COMMENTED OUT: Repeated debug prints
            # if debug:
            #     print(f"DEBUG: Climate results for '{category}' year {year_label}:")
            #     print("  annual_emissions:", annual_emissions)
            #     print("  annual_damages:", annual_damages)

            # Accumulate annual emissions and damages.
            for mer in MER_TYPES:
                lifetime_climate_emissions[mer] += annual_emissions.get(mer, 0.0)
            for key, value in annual_damages.items():
                lifetime_climate_damages[key] += value

            if climate_results:
                df_detailed = pd.concat([df_detailed, pd.DataFrame(climate_results, index=df_copy.index)], axis=1)

        # Create lifetime columns for climate impacts.
        lifetime_dict = {}
        for mer in MER_TYPES:
            emissions_col = f"{scenario_prefix}{category}_lifetime_mt_co2e_{mer}"
            lifetime_dict[emissions_col] = lifetime_climate_emissions[mer]

            for scc_assumption in ["low", "base", "high"]:
                damages_col = f"{scenario_prefix}{category}_lifetime_damages_climate_{mer}_{scc_assumption}"
                lifetime_dict[damages_col] = lifetime_climate_damages[(mer, scc_assumption)]
                # Calculate avoided damages if baseline data is provided.
                if menu_mp != 0 and df_baseline_damages is not None:
                    baseline_damages_col = f"baseline_{category}_lifetime_damages_climate_{mer}_{scc_assumption}"
                    avoided_damages_col = f"{scenario_prefix}{category}_avoided_damages_climate_{mer}_{scc_assumption}"
                    lifetime_dict[avoided_damages_col] = df_baseline_damages[baseline_damages_col] - lifetime_dict[damages_col]
            # Calculate avoided emissions if baseline data is provided.
            if menu_mp != 0 and df_baseline_damages is not None:
                baseline_emissions_col = f"baseline_{category}_lifetime_mt_co2e_{mer}"
                avoided_emissions_col = f"{scenario_prefix}{category}_avoided_mt_co2e_{mer}"
                lifetime_dict[avoided_emissions_col] = df_baseline_damages[baseline_emissions_col] - lifetime_dict[emissions_col]

        lifetime_columns_data.update(lifetime_dict)
        df_detailed = pd.concat([df_detailed, pd.DataFrame(lifetime_dict, index=df_copy.index)], axis=1)

    # Merge lifetime results with the main DataFrame.
    df_lifetime = pd.DataFrame(lifetime_columns_data, index=df_copy.index)
    df_main = df_copy.join(df_lifetime, how="left")

    df_main = df_main.round(2)
    df_detailed = df_detailed.round(2)

    if debug:
        print("DEBUG: Final df_main columns:")
        print(df_main.columns)

    return df_main, df_detailed


def calculate_climate_emissions_and_damages(
    df,
    category,
    year_label,
    adjusted_hdd_factor,
    lookup_emissions_electricity_climate,
    cambium_scenario,
    total_fossil_fuel_emissions,
    scenario_prefix,
    menu_mp,
    debug=False,
):
    """Calculate climate-related emissions and damages for a given category and year."""

    climate_results = {}
    annual_climate_emissions = {}
    annual_climate_damages = {}

    total_fossil_fuel_emissions_co2e = total_fossil_fuel_emissions["co2e"]

    # Helper functions for emission factor lookups.
    def get_emission_factor_lrmer(region):
        # Convert region to string to ensure lookup key matches expected format.
        region_str = str(region)
        if debug:
            print("DEBUG: get_emission_factor_lrmer: region", region, "converted to", region_str)
        return (
            lookup_emissions_electricity_climate.get((cambium_scenario, region_str), {})
            .get(year_label, {})
            .get("lrmer_mt_per_kWh_co2e", 0)
        )

    def get_emission_factor_srmer(region):
        # Convert region to string for proper lookup.
        region_str = str(region)
        if debug:
            print("DEBUG: get_emission_factor_srmer: region", region, "converted to", region_str)
        return (
            lookup_emissions_electricity_climate.get((cambium_scenario, region_str), {})
            .get(year_label, {})
            .get("srmer_mt_per_kWh_co2e", 0)
        )

    # Map the region to emission factors.
    mer_factors = {
        "lrmer": df["gea_region"].map(get_emission_factor_lrmer),
        "srmer": df["gea_region"].map(get_emission_factor_srmer),
    }
    if debug:
        print("DEBUG: MER factors (first 5 rows):")
        for mer in MER_TYPES:
            print(f"  {mer} factor head:", mer_factors[mer].head())

    # Determine the appropriate electricity consumption column based on menu_mp and apply HDD adjustment.
    if menu_mp == 0:
        electricity_consumption = df.get(
            f"base_electricity_{category}_consumption", pd.Series(0.0, index=df.index)
        ).fillna(0)
        electricity_consumption *= adjusted_hdd_factor
    else:
        consumption_col = f"mp{menu_mp}_{year_label}_{category}_consumption"
        electricity_consumption = df.get(consumption_col, pd.Series(0.0, index=df.index)).fillna(0)

    if debug:
        print("DEBUG: Electricity consumption (first 5 rows):", electricity_consumption.head())

    # Loop over each MER type to calculate annual emissions and damages.
    for mer_type in MER_TYPES:
        annual_emissions_electricity = electricity_consumption * TD_LOSSES_MULTIPLIER * mer_factors[mer_type]
        total_annual_climate_emissions = total_fossil_fuel_emissions_co2e + annual_emissions_electricity
        emissions_col = f"{scenario_prefix}{year_label}_{category}_mt_co2e_{mer_type}"
        climate_results[emissions_col] = total_annual_climate_emissions
        annual_climate_emissions[mer_type] = total_annual_climate_emissions

        if debug:
            print(f"DEBUG: For MER type '{mer_type}':")
            print("  annual_emissions_electricity (first 5 rows):", annual_emissions_electricity.head())
            print("  total_annual_climate_emissions (first 5 rows):", total_annual_climate_emissions.head())

        # Calculate damages for each SCC assumption.
        for scc_assumption in ["low", "base", "high"]:
            if scc_assumption == "low":
                scc_value = EPA_SCC_USD2023_PER_MT_LOW
            elif scc_assumption == "base":
                scc_value = EPA_SCC_USD2023_PER_MT_BASE
            elif scc_assumption == "high":
                scc_value = EPA_SCC_USD2023_PER_MT_HIGH
            else:
                raise ValueError(f"Invalid scc_value assumption: {scc_assumption}")

            damages_col = f"{scenario_prefix}{year_label}_{category}_damages_climate_{mer_type}_{scc_assumption}"
            total_annual_climate_damages = total_annual_climate_emissions * scc_value
            climate_results[damages_col] = total_annual_climate_damages
            annual_climate_damages[(mer_type, scc_assumption)] = total_annual_climate_damages

            if debug:
                print(f"  DEBUG: {damages_col} (first 5 rows):", total_annual_climate_damages.head())

    return climate_results, annual_climate_emissions, annual_climate_damages
