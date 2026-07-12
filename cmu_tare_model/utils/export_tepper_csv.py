"""One-time Tepper CSV exports for the TARE model (household and county).

This module produces two additive, human-readable CSVs per measure package for
the Tepper MBA team's cash-flow analysis:

    tepper_household_mp{mp}_{location_id}_{date}.csv   (one row per home)
    tepper_county_mp{mp}_{location_id}_{date}.csv      (one row per county)

It is strictly additive. It does not touch the frozen export/load contract
(export_model_run_output / load_model_run_output), recomputes nothing, and
rounds nothing -- every value is copied straight from a loaded DataFrame.

Household export
----------------
The source is the final loaded household frame DATAFRAMES_BY_MP[mp]['fixed_base'],
indexed by bldg_id. An explicit, ordered list of included data columns selects
what to ship; bookkeeping columns (REMDB row lookups, validation flags) are
dropped. NaN is
preserved everywhere -- it means "not applicable / failed validation," never
zero -- so the exported row count equals the input row count.

Scope note: this run is PA-only, MP3 and MP4, discount rate 'fixed_base'.
"""

import os
import pathlib
from typing import List, Union

import pandas as pd

from cmu_tare_model.utils.column_names import (
    NPV_CASE_CATEGORIES,
    create_adoption_col,
    create_capital_col,
    create_npv_case_col,
)
from cmu_tare_model.utils.modeling_params import define_scenario_params

# This snapshot exports a single run. These are canonical column suffixes (the
# run's only active discount method and its REMDB cost scenario), not scenario
# prefixes, so they are named here once rather than hardcoded at each call site.
METHOD_SUFFIX = "_fixed_base"
COST_SCENARIO = "v4MID"
POLICY_SCENARIO = "2025 Reference Case"

# Output directory, kept distinct from the frozen retrofit_mp{mp}_results/ and
# baseline_summary/ trees so the Tepper snapshot never collides with them.
TEPPER_SUBDIR = "tepper_export"


def build_household_column_list(menu_mp: Union[int, str]) -> List[str]:
    """Build the ordered list of household columns to include for one MP.

    The list is explicit and grouped by theme (identifiers, geography,
    building, household income, existing HVAC, retrofit HVAC, consumption,
    fuel costs, installed costs, rebate, emissions and damages, model
    parameters, nine NPV, nine net capital, nine adopter). bldg_id is the frame
    index and is therefore not in this list; it is written as the CSV index.

    Every measure-package-specific name is derived from the scenario prefix
    helper and a plain 'mp{mp}_' token, so no scenario string is hardcoded.

    Args:
        menu_mp: Measure package number (3 or 4).

    Returns:
        Ordered list of column names to select, excluding the bldg_id index.
    """
    menu_mp = int(menu_mp)
    # 'ref2025_mp{mp}_' -- carries the scenario; drives fuel-cost, emissions,
    # total-capital, NPV, net-capital, and adopter names.
    scenario_prefix = define_scenario_params(menu_mp, POLICY_SCENARIO)[0]
    # 'mp{mp}_' -- the bare measure-package token used by the REMDB input cost
    # and rebate columns, which carry no scenario prefix.
    mp_token = f"mp{menu_mp}_"

    identifiers = [
        "weight", "state", "county", "county_fips", "puma", "county_and_puma",
    ]
    geography = [
        "census_region", "census_division", "census_division_recs",
        "building_america_climate_zone", "reeds_balancing_area", "city",
        "urbanicity", "weather_file_city", "Longitude", "Latitude",
        "gea_region",
    ]
    building = [
        "square_footage", "building_type", "occupancy", "tenure",
        "vacancy_status", "vintage",
    ]
    household_income = [
        "income", "federal_poverty_level", "household_income",
        "census_area_medianIncome", "income_level", "percent_AMI",
        "lmi_or_mui",
    ]
    existing_hvac = [
        "base_heating_fuel", "heating_type", "base_heating_efficiency",
        "base_cooling_fuel", "cooling_type", "base_cooling_efficiency",
        "fuel_type_heating", "fuel_type_cooling", "hvac_has_ducts",
        "hvac_heating_type_and_fuel", "hvac_heating_efficiency",
        "size_heating_system_primary_k_btu_h", "hvac_cooling_type",
        "hvac_cooling_efficiency", "size_cooling_system_primary_k_btu_h",
    ]
    retrofit_hvac = [
        "upgrade_hvac_heating_efficiency", "upgrade_hvac_cooling_efficiency",
    ]
    consumption = [
        "base_electricity_heating_consumption",
        "base_electricity_cooling_consumption",
        "base_fuelOil_heating_consumption",
        "base_naturalGas_heating_consumption",
        "base_propane_heating_consumption",
        "baseline_heating_consumption",
        "baseline_cooling_consumption",
        f"{mp_token}heating_consumption",
        f"{mp_token}cooling_consumption",
    ]
    fuel_costs = [
        "baseline_heating_lifetime_fuel_cost",
        f"{scenario_prefix}heating_lifetime_fuel_cost",
        f"{scenario_prefix}heating_lifetime_savings_fuel_cost",
        "baseline_cooling_lifetime_fuel_cost",
        f"{scenario_prefix}cooling_lifetime_fuel_cost",
        f"{scenario_prefix}cooling_lifetime_savings_fuel_cost",
    ]
    # The air-source heat pump provides both heating and cooling, so the single
    # heat-pump installed cost is recorded once on the heating side
    # (heating_upgrade). The cooling column is the avoided-replacement
    # (counterfactual AC) cost, not a separate cooling upgrade.
    installed_costs = [
        f"{mp_token}heating_replacement_installed_cost_{COST_SCENARIO}",
        f"{mp_token}heating_upgrade_installed_cost_{COST_SCENARIO}",
        f"{mp_token}cooling_replacement_installed_cost_{COST_SCENARIO}",
        f"{scenario_prefix}heating_total_capital_cost_{COST_SCENARIO}",
    ]
    # The heat-pump rebate applies to the whole system (heating and cooling)
    # and is recorded once on the heating side, not split by end use. Both rebate
    # policy scenarios are carried: the 2024-guidance amount and the June 2026-guidance
    # amount, plus the June 2026 program label and the whole-home savings fraction
    # that drives its HOMES tiers.
    rebate = [
        f"{mp_token}heating_rebate_amount_{COST_SCENARIO}",
        f"{mp_token}heating_rebate_amount_june2026_{COST_SCENARIO}",
        f"{mp_token}rebate_eligibility_june2026",
        f"{mp_token}modeled_savings_frac",
    ]
    # Central estimate only: LRMER mid-case tonnage and central climate
    # damages. Lower/upper bounds and the SRMER series are omitted from the
    # Tepper snapshot to keep the damages block focused on the headline value.
    emissions_damages = [
        "baseline_heating_lifetime_mt_co2e_lrmer",
        "baseline_heating_lifetime_damages_climate_lrmer_central",
        "baseline_cooling_lifetime_mt_co2e_lrmer",
        "baseline_cooling_lifetime_damages_climate_lrmer_central",
        f"{scenario_prefix}heating_lifetime_mt_co2e_lrmer",
        f"{scenario_prefix}heating_lifetime_damages_climate_lrmer_central",
        f"{scenario_prefix}heating_avoided_damages_climate_lrmer_central",
        f"{scenario_prefix}heating_avoided_mt_co2e_lrmer",
        f"{scenario_prefix}cooling_lifetime_mt_co2e_lrmer",
        f"{scenario_prefix}cooling_lifetime_damages_climate_lrmer_central",
        f"{scenario_prefix}cooling_avoided_damages_climate_lrmer_central",
        f"{scenario_prefix}cooling_avoided_mt_co2e_lrmer",
    ]
    model_parameters = [
        "public_discount_rate",
        "private_discount_rate_fixed_base",
        "private_discount_rate_variable",
    ]
    # All nine NPV cases (three scopes x three rebate policy scenarios: unsub, 2024 sub,
    # June 2026 sub), their net capital costs, and their economic-adopter flags
    # -- carried in full, never collapsed.
    nine_npv = [
        create_npv_case_col(scenario_prefix, case, METHOD_SUFFIX)
        for case in NPV_CASE_CATEGORIES
    ]
    nine_net_capital = [
        create_capital_col(scenario_prefix, case, net=True,
                           cost_scenario=COST_SCENARIO)
        for case in NPV_CASE_CATEGORIES
    ]
    nine_adopter = [
        create_adoption_col(scenario_prefix, case, METHOD_SUFFIX)
        for case in NPV_CASE_CATEGORIES
    ]

    return (
        identifiers + geography + building + household_income + existing_hvac
        + retrofit_hvac + consumption + fuel_costs + installed_costs + rebate
        + emissions_damages + model_parameters + nine_npv + nine_net_capital
        + nine_adopter
    )


def export_tepper_household(
    df_household: pd.DataFrame,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
) -> str:
    """Write the household CSV of included columns for one measure package.

    Selects the ordered list of included columns with df.loc[:, cols]
    (preserving column order) and writes with the bldg_id index. Copies values
    verbatim: no
    fillna, no rounding, no row drops, so NaN placement and row count survive
    the round trip.

    Args:
        df_household: Final loaded household frame
            (DATAFRAMES_BY_MP[mp]['fixed_base']), indexed by bldg_id.
        menu_mp: Measure package number (3 or 4).
        output_folder_path: Base directory for exports.
        location_id: Location identifier for the filename (e.g. 'PA').
        results_export_formatted_date: Date string for the filename.

    Returns:
        The full path of the written CSV.

    Raises:
        TypeError: If df_household is not a DataFrame.
        ValueError: If the frame is not indexed by bldg_id.
        KeyError: If any included column is absent (names every missing one).
        OSError: If the directory cannot be created or the file cannot be written.
    """
    if not isinstance(df_household, pd.DataFrame):
        raise TypeError(
            f"df_household must be a DataFrame, got {type(df_household)!r}"
        )
    if df_household.index.name != "bldg_id":
        raise ValueError(
            "df_household must be indexed by 'bldg_id'; got index name "
            f"{df_household.index.name!r}"
        )

    menu_mp = int(menu_mp)
    included_columns = build_household_column_list(menu_mp)

    # Fail loud with the full list of missing names rather than exporting a
    # partial frame.
    missing = [c for c in included_columns if c not in df_household.columns]
    if missing:
        raise KeyError(
            f"MP{menu_mp} household frame is missing "
            f"{len(missing)} required column(s): {missing}"
        )

    df_out = df_household.loc[:, included_columns]

    directory = os.path.join(output_folder_path, TEPPER_SUBDIR)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    filename = (
        f"tepper_household_mp{menu_mp}_{location_id}_"
        f"{results_export_formatted_date}.csv"
    )
    full_path = os.path.join(directory, filename)

    try:
        df_out.to_csv(full_path)  # index=True writes bldg_id as the first column
    except Exception as exc:
        raise OSError(f"Error writing Tepper household CSV to {full_path}: {exc}")

    print(f"""\
[OK] Tepper household export written
     MP{menu_mp} | {df_out.shape[0]:,} rows x {df_out.shape[1]} columns (+ bldg_id index)
     Path: {full_path}""")

    return full_path


# Columns each county result table must supply, and the final ordered output
# schema. Kept explicit so an upstream schema change surfaces here as a KeyError
# rather than silently reshaping the export.
COUNTY_ADOPTION_COLS = ["county", "state", "home_count", "adoption_rate_pct"]
COUNTY_BILL_COLS = ["county", "operating_cost_pct_change"]
COUNTY_DEMAND_METRIC_COLS = [
    "baseline_elec_gwh", "retrofit_elec_gwh", "elec_change_gwh",
    "site_energy_change_gwh", "pct_elec_demand_change", "pct_site_energy_change",
]
# Eleven-column output: adoption block, then bill savings, then demand metrics.
COUNTY_OUTPUT_ORDER = (
    COUNTY_ADOPTION_COLS + ["operating_cost_pct_change"]
    + COUNTY_DEMAND_METRIC_COLS
)


def export_tepper_county(
    df_adoption: pd.DataFrame,
    df_bill_savings: pd.DataFrame,
    df_demand: pd.DataFrame,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
) -> str:
    """Assemble and write the county-level Tepper CSV for one measure package.

    The county results arrive as three separate county-aggregated tables. This
    function joins them on 'county' alone and writes the eleven-column result;
    it recomputes nothing. 'state' and 'home_count' appear in both the adoption
    and demand tables, so they are taken from the adoption table and dropped
    from the demand table before the merge -- pandas therefore never emits
    _x/_y suffixes. The two home_count values are compared first and any
    disagreement is reported.

    Args:
        df_adoption: econ_adoption_rate_results[mp] with columns county, state,
            home_count, adoption_rate_pct.
        df_bill_savings: bill_savings_results[mp] with columns county,
            operating_cost_pct_change.
        df_demand: demand_results[mp] with columns county, state, home_count,
            baseline_elec_gwh, retrofit_elec_gwh, elec_change_gwh,
            site_energy_change_gwh, pct_elec_demand_change,
            pct_site_energy_change.
        menu_mp: Measure package number (3 or 4).
        output_folder_path: Base directory for exports.
        location_id: Location identifier for the filename (e.g. 'PA').
        results_export_formatted_date: Date string for the filename.

    Returns:
        The full path of the written CSV.

    Raises:
        TypeError: If any input is not a DataFrame.
        KeyError: If any required column is absent from its source table.
        ValueError: If 'county' has duplicate keys (raised by merge validate).
        OSError: If the directory cannot be created or the file cannot be written.
    """
    # Step 1 -- validate inputs.
    for name, frame in (("df_adoption", df_adoption),
                        ("df_bill_savings", df_bill_savings),
                        ("df_demand", df_demand)):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame, got {type(frame)!r}")

    menu_mp = int(menu_mp)

    def _require(frame: pd.DataFrame, cols: List[str], label: str) -> None:
        missing = [c for c in cols if c not in frame.columns]
        if missing:
            raise KeyError(f"{label} is missing required column(s): {missing}")

    _require(df_adoption, COUNTY_ADOPTION_COLS, "df_adoption (adoption table)")
    _require(df_bill_savings, COUNTY_BILL_COLS,
             "df_bill_savings (bill-savings table)")
    _require(df_demand, ["county", "state", "home_count"]
             + COUNTY_DEMAND_METRIC_COLS, "df_demand (demand table)")

    # Step 2 -- reconcile home_count between the adoption and demand tables.
    # home_count is the per-county home tally each KPI grouped over; a mismatch
    # means the two tables saw different home sets and is worth surfacing, but
    # it does not block the export.
    hc_check = df_adoption[["county", "home_count"]].merge(
        df_demand[["county", "home_count"]], on="county", how="outer",
        suffixes=("_adoption", "_demand"),
    )
    disagree = hc_check[
        hc_check["home_count_adoption"] != hc_check["home_count_demand"]
    ]
    if len(disagree):
        print(f"[WARN] home_count disagrees for {len(disagree)} county(ies) "
              f"between the adoption and demand tables:")
        for _, row in disagree.iterrows():
            print(f"       {row['county']}: adoption="
                  f"{row['home_count_adoption']} demand="
                  f"{row['home_count_demand']}")
    else:
        print("[OK] home_count agrees across the adoption and demand tables.")

    # Step 3 -- drop the duplicated state/home_count from demand so the merge
    # keys on county alone and emits no _x/_y suffixes.
    df_demand_metrics = df_demand[["county"] + COUNTY_DEMAND_METRIC_COLS]

    # Step 4 -- outer-join the three tables on county. validate='one_to_one'
    # raises if any table has a duplicate county key.
    merged = (
        df_adoption[COUNTY_ADOPTION_COLS]
        .merge(df_bill_savings[COUNTY_BILL_COLS], on="county",
               how="outer", validate="one_to_one")
        .merge(df_demand_metrics, on="county",
               how="outer", validate="one_to_one")
    )

    # Step 5 -- order to the eleven-column schema. county stays a GISJOIN-style
    # string; it is neither cast to int nor zero-stripped.
    merged = merged.loc[:, list(COUNTY_OUTPUT_ORDER)]

    # Step 6 -- write. county is a plain column, so do not write the index.
    directory = os.path.join(output_folder_path, TEPPER_SUBDIR)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    filename = (
        f"tepper_county_mp{menu_mp}_{location_id}_"
        f"{results_export_formatted_date}.csv"
    )
    full_path = os.path.join(directory, filename)

    try:
        merged.to_csv(full_path, index=False)
    except Exception as exc:
        raise OSError(f"Error writing Tepper county CSV to {full_path}: {exc}")

    print(f"""\
[OK] Tepper county export written
     MP{menu_mp} | {merged.shape[0]:,} counties x {merged.shape[1]} columns
     Path: {full_path}""")

    return full_path
