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

Scope note: MP3 and MP4, discount rate 'fixed_base'. Each run writes one
household CSV and one county CSV per export scope. A scope is a column and a
value -- 'state' = 'PA', or 'county' = a Census GISJOIN code -- or no filter at
all for the whole run. Filtering happens here at export time, not at model run
scope, so a national run can emit a national file and any number of state or
county files from the same results.
"""

import os
import pathlib
import shutil
from typing import List, Optional, Union

import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    ANCHOR_YEAR,
    EQUIPMENT_SPECS,
    REBATE_GUIDANCE_JUNE2026,
)
from cmu_tare_model.utils.column_names import (
    BASE_CASE_NPV_CASE,
    create_adoption_col,
    create_annual_consumption_col,
    create_capital_col,
    create_cooling_credit_applied_col,
    create_cost_col,
    create_discounted_savings_col,
    create_npv_case_col,
    create_rebate_col,
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


def filter_to_export_scope(
    df: pd.DataFrame,
    scope_column: Optional[str],
    scope_value: Optional[object],
) -> pd.DataFrame:
    """Return the rows of a frame that fall inside one export scope.

    The household frame and the three county tables all carry a 'state'
    column and a 'county' column holding the Census GISJOIN code, so one
    scope definition filters either kind of frame. That is why this takes a
    column name rather than a fixed geography.

    Args:
        df: Frame to filter -- a household frame or a county table.
        scope_column: Column to match on, for example 'county' or 'state'.
            None means keep every row.
        scope_value: Value that column must equal, for example 'G4200030'
            or 'PA'. Ignored when scope_column is None.

    Returns:
        The matching rows, or the frame unchanged when scope_column is None.

    Raises:
        KeyError: If scope_column is not a column of this frame.
        ValueError: If no row matches, which would write an empty file.
    """
    if scope_column is None:
        return df

    if scope_column not in df.columns:
        raise KeyError(
            f"Export scope column {scope_column!r} is not in this frame. "
            f"The first few columns are {sorted(df.columns)[:10]}.")

    df_scope = df[df[scope_column] == scope_value]
    if len(df_scope) == 0:
        raise ValueError(
            f"Export scope {scope_column} = {scope_value!r} matched no rows, "
            "so the export would be empty.")
    return df_scope


def export_source_data_copies(output_folder_path: str) -> List[str]:
    """Copy the three vendored input CSVs into the export directory.

    A reader who wants to vary fuel prices needs the same inputs the model
    used: the EIA fuel prices, the AEO2026 fuel price projection factors, and
    the AEO2026 degree day factors.

    The files are copied unchanged, so what the reader holds is exactly what
    produced the exported numbers. Two things about them are explained in the
    data dictionary rather than edited out here. The 'National' rows are a
    real fallback the model uses for a region it does not recognise, and
    removing them would mean shipping a file the model never ran on. The
    projection tables run to 2050 while the exported numbers stop at 2039.

    Args:
        output_folder_path: Base directory for exports.

    Returns:
        The full paths of the three written CSVs.

    Raises:
        OSError: If the directory cannot be created or a file cannot be copied.
    """
    source_paths = [
        os.path.join(PROJECT_ROOT, "cmu_tare_model", "data", "fuel_prices",
                     "eia_fuel_price_data_2025_usd2025.csv"),
        os.path.join(PROJECT_ROOT, "cmu_tare_model", "data", "projections",
                     "aeo2026_fuel_price_factors_2025_2050.csv"),
        os.path.join(PROJECT_ROOT, "cmu_tare_model", "data", "projections",
                     "aeo2026_degree_day_factors_2025_2050.csv"),
    ]

    directory = os.path.join(output_folder_path, TEPPER_SUBDIR, "source_data")
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

    written_paths = []
    for source_path in source_paths:
        destination_path = os.path.join(
            directory, os.path.basename(source_path))
        shutil.copy2(source_path, destination_path)
        written_paths.append(destination_path)

    print(f"""\
[OK] Source data copied for readers
     {len(written_paths)} files, unchanged -> {directory}""")
    return written_paths


def build_annual_consumption_column_list(menu_mp: Union[int, str]) -> List[str]:
    """Build the 60 per-year consumption column names for one measure package.

    Four streams -- baseline heating, retrofit heating, baseline cooling,
    retrofit cooling -- each covering ANCHOR_YEAR through the end of the
    equipment lifetime (2025-2039 for a 15-year lifetime). Grouped by stream
    rather than interleaved by year, so a spreadsheet user can select one
    stream as a single block of columns.

    These 60 columns live in the supplemental fuel-cost frame, not in the
    household summary frame, which is why they are listed separately from
    build_household_column_list.

    Args:
        menu_mp: Measure package number (3 or 4).

    Returns:
        Ordered list of 60 column names.
    """
    menu_mp = int(menu_mp)
    scenario_prefix = define_scenario_params(menu_mp, POLICY_SCENARIO)[0]

    columns = []
    for category in ("heating", "cooling"):
        years = range(ANCHOR_YEAR, ANCHOR_YEAR + EQUIPMENT_SPECS[category])
        for prefix in ("baseline_", scenario_prefix):
            columns.extend(
                create_annual_consumption_col(prefix, year, category)
                for year in years
            )
    return columns


def build_household_column_list(menu_mp: Union[int, str]) -> List[str]:
    """Build the ordered list of household columns to include for one MP.

    The list is explicit and ordered so it reads left to right as the
    derivation of the NPV: who the home is, what it consumes each year, what
    that consumption costs, what the equipment costs, and finally the NPV and
    the adoption flag. bldg_id is the frame index and is therefore not in this
    list; it is written as the CSV index.

    Three notes on what is and is not here.

    The 60 per-year consumption columns come from a different frame
    (build_annual_consumption_column_list) and are spliced in at export time.

    Only one of the nine NPV cases is shipped: the unsubsidized case that
    credits both the avoided heating replacement and the avoided cooling
    replacement. The model still computes and stores all nine unchanged. This
    export carries one because its readers model the unsubsidized case and
    apply their own rebate assumptions on top.

    Emissions and climate damages are not shipped at all. They play no part in
    the adoption decision, which is based on the private NPV alone.

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
    # Whether the model produced a result for this home at all. Shipped so a
    # reader can tell a blank meaning "not applicable" from a real zero.
    applicability = [
        "include_heating",
        "include_cooling",
    ]
    base_year_consumption = [
        "base_electricity_heating_consumption",
        "base_electricity_cooling_consumption",
        "base_fuelOil_heating_consumption",
        "base_naturalGas_heating_consumption",
        "base_propane_heating_consumption",
        "baseline_heating_consumption",
        "baseline_cooling_consumption",
        f"{mp_token}heating_consumption",
        f"{mp_token}cooling_consumption",
        # Whole-home ELECTRICITY totals (kWh), baseline and retrofit -- the
        # electricity change is their difference. Not the all-fuel site energy
        # (baseline_total_site_consumption); read the '...electricity...' token.
        "base_total_electricity_consumption",
        f"{mp_token}total_electricity_consumption",
        # Denominator of mp{mp}_modeled_savings_frac below, which drives the
        # HOMES rebate tiers. Without it that fraction cannot be checked.
        "baseline_total_site_consumption",
    ]
    # Per-year projected consumption, 2025-2039, four streams. Sourced from
    # the supplemental fuel-cost frame, not the household summary frame.
    annual_consumption = build_annual_consumption_column_list(menu_mp)
    # Per-home peak demand pass-through for a short-term peak-load approximation
    # done per building ID outside this model (a simple annual max per home, not
    # aligned in time across homes). Baseline values have no savings variant;
    # the post-retrofit block carries ResStock's baseline-minus-upgrade delta as
    # the '..._savings' columns. The kW electric-demand pair and the kBtu/hr
    # thermal-load pair are distinct quantities -- both are kept, not combined.
    peak = [
        "base_peak_electricity_cooling_kw",
        "base_peak_electricity_heating_kw",
        "base_peak_load_cooling_kbtu_hr",
        "base_peak_load_heating_kbtu_hr",
        f"{mp_token}peak_electricity_cooling_kw",
        f"{mp_token}peak_electricity_heating_kw",
        f"{mp_token}peak_electricity_cooling_kw_savings",
        f"{mp_token}peak_electricity_heating_kw_savings",
        f"{mp_token}peak_load_cooling_kbtu_hr",
        f"{mp_token}peak_load_heating_kbtu_hr",
        f"{mp_token}peak_load_cooling_kbtu_hr_savings",
        f"{mp_token}peak_load_heating_kbtu_hr_savings",
    ]
    lifetime_fuel_costs = [
        "baseline_heating_lifetime_fuel_cost",
        f"{scenario_prefix}heating_lifetime_fuel_cost",
        f"{scenario_prefix}heating_lifetime_savings_fuel_cost",
        "baseline_cooling_lifetime_fuel_cost",
        f"{scenario_prefix}cooling_lifetime_fuel_cost",
        f"{scenario_prefix}cooling_lifetime_savings_fuel_cost",
        # True where the heat pump uses more cooling energy than the existing
        # air conditioner, so cooling savings are negative. A real result, not
        # an error: a room air conditioner cools one room, the heat pump cools
        # the whole home.
        f"{scenario_prefix}cooling_lifetime_savings_negative",
    ]
    # The air-source heat pump provides both heating and cooling, so the single
    # heat-pump installed cost is recorded once on the heating side
    # (heating_upgrade). The cooling column is the avoided-replacement
    # (counterfactual AC) cost, not a separate cooling upgrade.
    installed_costs = [
        create_cost_col(menu_mp, "heating", "replacement", COST_SCENARIO),
        create_cost_col(menu_mp, "heating", "upgrade", COST_SCENARIO),
        create_cost_col(menu_mp, "cooling", "replacement", COST_SCENARIO),
        # The cooling credit the NPV actually subtracted, which is 0.0 for a
        # home with no air conditioner and 0.0 where the cooling replacement
        # cost above is blank.
        create_cooling_credit_applied_col(menu_mp, COST_SCENARIO),
    ]
    # The heat-pump rebate applies to the whole system (heating and cooling)
    # and is recorded once on the heating side, not split by end use. Only the
    # June 2026-guidance amount is carried, with its program label and the
    # whole-home savings fraction that drives the HOMES tiers. The
    # 2024-guidance amount is not shipped: this export is unsubsidized, and its
    # readers apply their own rebate assumptions.
    rebate = [
        create_rebate_col(menu_mp, "heating", COST_SCENARIO,
                          guidance=REBATE_GUIDANCE_JUNE2026),
        f"{mp_token}rebate_eligibility_june2026",
        f"{mp_token}modeled_savings_frac",
    ]
    model_parameters = [
        "public_discount_rate",
        f"private_discount_rate{METHOD_SUFFIX}",
    ]
    # The savings half of the NPV, discounted to the anchor year. Adding these
    # two and subtracting the net capital cost below gives the NPV column.
    discounted_savings = [
        create_discounted_savings_col(scenario_prefix, "heating", METHOD_SUFFIX),
        create_discounted_savings_col(scenario_prefix, "cooling", METHOD_SUFFIX),
    ]
    net_capital = [
        create_capital_col(scenario_prefix, BASE_CASE_NPV_CASE, net=True,
                           cost_scenario=COST_SCENARIO)
    ]
    npv = [
        create_npv_case_col(scenario_prefix, BASE_CASE_NPV_CASE, METHOD_SUFFIX)
    ]
    adopter = [
        create_adoption_col(scenario_prefix, BASE_CASE_NPV_CASE, METHOD_SUFFIX)
    ]

    return (
        identifiers + geography + building + household_income + existing_hvac
        + retrofit_hvac + applicability + peak + base_year_consumption
        + annual_consumption + lifetime_fuel_costs + installed_costs
        + rebate + model_parameters + discounted_savings + net_capital
        + npv + adopter
    )


def export_tepper_household(
    df_household: pd.DataFrame,
    df_annual_consumption: pd.DataFrame,
    menu_mp: Union[int, str],
    output_folder_path: str,
    location_id: str,
    results_export_formatted_date: str,
) -> str:
    """Write the household CSV of included columns for one measure package.

    Draws from two frames. The household summary frame supplies 94 columns;
    the supplemental fuel-cost frame supplies the 60 per-year consumption
    columns. Both are indexed by bldg_id. Values are copied verbatim from
    both: no fillna, no rounding, no row drops, so blanks and row count
    survive the round trip.

    The annual frame may hold more homes than the household frame, which is
    what happens when exporting one county out of a national run: the caller
    passes the county subset as df_household and the full national
    supplemental frame as df_annual_consumption. It may not hold fewer.

    Args:
        df_household: Final loaded household frame
            (DATAFRAMES_BY_MP[mp]['fixed_base']), indexed by bldg_id.
        df_annual_consumption: Supplemental fuel-cost frame for the same
            measure package and run, indexed by bldg_id. Must cover every home
            in df_household.
        menu_mp: Measure package number (3 or 4).
        output_folder_path: Base directory for exports.
        location_id: Location identifier for the filename (e.g. 'National',
            'Allegheny'). Used for naming only; it does not filter rows.
        results_export_formatted_date: Date string for the filename.

    Returns:
        The full path of the written CSV.

    Raises:
        TypeError: If either input is not a DataFrame.
        ValueError: If either frame is not indexed by bldg_id, or the annual
            frame is missing homes that the household frame contains.
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
    if not isinstance(df_annual_consumption, pd.DataFrame):
        raise TypeError(
            "df_annual_consumption must be a DataFrame, got "
            f"{type(df_annual_consumption)!r}"
        )
    if df_annual_consumption.index.name != "bldg_id":
        raise ValueError(
            "df_annual_consumption must be indexed by 'bldg_id'; got index "
            f"name {df_annual_consumption.index.name!r}"
        )

    menu_mp = int(menu_mp)
    included_columns = build_household_column_list(menu_mp)
    annual_columns = build_annual_consumption_column_list(menu_mp)

    # Split the export list by which frame each column comes from.
    annual_column_set = set(annual_columns)
    summary_columns = []
    for column in included_columns:
        if column not in annual_column_set:
            summary_columns.append(column)

    # Every home in the household frame must appear in the annual frame, or
    # the export would silently ship blank consumption for the difference.
    homes_without_consumption = df_household.index.difference(
        df_annual_consumption.index)
    if len(homes_without_consumption) > 0:
        raise ValueError(
            f"MP{menu_mp}: df_annual_consumption is missing "
            f"{len(homes_without_consumption):,} of the "
            f"{len(df_household):,} homes in df_household. "
            f"First few: {list(homes_without_consumption[:5])}"
        )
    if not df_annual_consumption.index.is_unique:
        raise ValueError(
            f"MP{menu_mp}: df_annual_consumption has repeated bldg_id values, "
            "so each home would not map to exactly one row of consumption."
        )

    # Fail loud with the full list of missing names rather than exporting a
    # partial frame. Each list is checked against the frame it comes from, so
    # the error message names the right file.
    missing_from_summary = []
    for column in summary_columns:
        if column not in df_household.columns:
            missing_from_summary.append(column)
    if missing_from_summary:
        raise KeyError(
            f"MP{menu_mp} household summary frame is missing "
            f"{len(missing_from_summary)} required column(s): "
            f"{missing_from_summary}"
        )

    missing_from_annual = []
    for column in annual_columns:
        if column not in df_annual_consumption.columns:
            missing_from_annual.append(column)
    if missing_from_annual:
        raise KeyError(
            f"MP{menu_mp} supplemental fuel-cost frame is missing "
            f"{len(missing_from_annual)} required column(s): "
            f"{missing_from_annual}"
        )

    # Line the annual frame up with the household frame, then take the columns
    # in their declared order. The two frames share no column names, so
    # nothing is overwritten when they are joined.
    df_annual_aligned = df_annual_consumption.loc[
        df_household.index, annual_columns]
    df_joined = pd.concat([df_household, df_annual_aligned], axis=1)
    df_out = df_joined.loc[:, included_columns]

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
# NOTE: 'site_energy_change_gwh' and 'pct_site_energy_change' are ALIASES of the
# electricity metrics (see demand.py) -- with whole-home electrification measured
# on the electricity total, site-energy change equals electricity change. They
# are NOT independent all-fuel numbers; prefer 'elec_change_gwh' /
# 'pct_elec_demand_change' for an electricity read.
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
    _x/_y suffixes. Per-county home_count reconciliation between the two tables
    is done upstream in the notebook, with a one-home tolerance from the
    sampling weight.

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

    # Step 2 -- drop the duplicated state/home_count from demand so the merge
    # keys on county alone and emits no _x/_y suffixes. (Per-county home_count
    # reconciliation is done upstream in the notebook, with a one-home tolerance
    # from the sampling weight, so it is not repeated here.)
    df_demand_metrics = df_demand[["county"] + COUNTY_DEMAND_METRIC_COLS]

    # Step 3 -- outer-join the three tables on county. validate='one_to_one'
    # raises if any table has a duplicate county key.
    merged = (
        df_adoption[COUNTY_ADOPTION_COLS]
        .merge(df_bill_savings[COUNTY_BILL_COLS], on="county",
               how="outer", validate="one_to_one")
        .merge(df_demand_metrics, on="county",
               how="outer", validate="one_to_one")
    )

    # Step 4 -- order to the eleven-column schema. county stays a GISJOIN-style
    # string; it is neither cast to int nor zero-stripped.
    merged = merged.loc[:, list(COUNTY_OUTPUT_ORDER)]

    # Step 5 -- write. county is a plain column, so do not write the index.
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
