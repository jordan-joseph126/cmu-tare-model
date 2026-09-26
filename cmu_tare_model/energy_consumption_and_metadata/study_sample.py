"""
The study sample: the one set of homes every TARE result covers.

A home is in the sample if its baseline heating system is one the study can
replace and cost (include_heating = True: electricity, natural gas, propane, or
fuel oil, heated by a furnace, boiler, or electric baseboard -- not an existing
heat pump, wall/floor furnace, or shared system), and if ResStock applied every
measure package in the run to it. df_enduse_refactored sets this once as the
include_sample column; this module only reads that column, so the rule lives in
one place. The ids are passed to results that do not use the TARE frame
(county demand, peaks, custom weighting).
"""

from typing import Dict, List, Union

import pandas as pd

from cmu_tare_model.utils.calculation_utils import (
    EXISTING_HEAT_PUMP_TYPES,
    compute_funnel_stage_row,
)


def build_tare_sample_ids(
    df_enduse: pd.DataFrame,
    release: str,
) -> Dict[str, Union[str, pd.Index, Dict[str, List[int]]]]:
    """Builds TARE_SAMPLE_IDS from the include_sample flag on the baseline frame.

    Args:
        df_enduse: Baseline frame from df_enduse_refactored, indexed by bldg_id,
            with include_sample and county_fips.
        release: ResStock release, e.g. '2022.1.1'.

    Returns:
        {'release': release,
         'all': sorted pd.Index of sample bldg_ids,
         'by_county': {county FIPS: sorted list of sample bldg_ids}}.

    Raises:
        KeyError: If include_sample or county_fips is missing.
        ValueError: If the sample is empty.
    """
    required = ('include_sample', 'county_fips')
    missing = [col for col in required if col not in df_enduse.columns]
    if missing:
        raise KeyError(f"build_tare_sample_ids needs columns {missing}")

    # Step 1 -- read the flag; the rule itself lives in df_enduse_refactored
    in_sample = df_enduse['include_sample'].astype(bool)
    sample = df_enduse.index[in_sample].sort_values()
    if sample.empty:
        raise ValueError("The study sample is empty; check the filters.")

    # Step 2 -- group by county for the county-level demand and peak results
    # zfill: a saved run reads county_fips back as a number (6001, not '06001')
    county_fips = df_enduse.loc[sample, 'county_fips'].astype(str).str.zfill(5)
    by_county = {
        fips: sorted(county_bldg_ids)
        for fips, county_bldg_ids in county_fips.groupby(county_fips).groups.items()
    }
    return {'release': release, 'all': sample, 'by_county': by_county}


def build_sample_funnel(
    df_funnel_packages: List[pd.DataFrame],
    applicable_bldg_ids: List[pd.Index],
    df_enduse: pd.DataFrame,
) -> pd.DataFrame:
    """Funnel table from the ResStock stock to the study sample (SI table).

    Args:
        df_funnel_packages: The funnel load_and_filter_upgrade returned for each
            package in the run (load, applicability, occupancy, housing type...).
        applicable_bldg_ids: The same packages' bldg_ids, as passed to
            df_enduse_refactored.
        df_enduse: Baseline frame from df_enduse_refactored.

    Returns:
        One row per step: stage, rdu_count, weighted_count, removed_rdu,
        removed_homes, and each fuel's share; then the two AC rows.

    Raises:
        ValueError: If the packages disagree at an early step, or the last step
            does not equal the include_sample count.
    """
    # Step 1 -- early steps from the package funnels (applicability first).
    # Every package must give the same counts, so one table describes them all.
    stage_counts_by_package = [
        tuple(df_funnel_package['rdu_count'])
        for df_funnel_package in df_funnel_packages]
    if len(set(stage_counts_by_package)) != 1:
        raise ValueError(
            f"Packages disagree at an early funnel step: {stage_counts_by_package}")
    df_scope = df_funnel_packages[0].copy()

    # Step 2 -- heating steps on the homes left after those filters
    in_scope_ids = df_enduse.index
    for package_ids in applicable_bldg_ids:
        in_scope_ids = in_scope_ids.intersection(package_ids)
    df_in_scope = df_enduse.loc[in_scope_ids]
    valid_fuel = df_in_scope['valid_fuel_heating'].astype(bool)
    no_heat_pump = ~df_in_scope['heating_type'].isin(EXISTING_HEAT_PUMP_TYPES)
    heating_steps = [
        ('heating_fuel', valid_fuel),
        ('no_existing_heat_pump', valid_fuel & no_heat_pump),
        ('replaceable_heating_system', df_in_scope['include_heating'].astype(bool)),
    ]
    heating_rows = [compute_funnel_stage_row(df_in_scope, label, mask)
                    for label, mask in heating_steps]

    # Step 3 -- the last step must be exactly the model's sample
    sample = df_enduse['include_sample'].astype(bool)
    n_sample = int(sample.sum())
    if heating_rows[-1]['rdu_count'] != n_sample:
        raise ValueError(f"Funnel ends at {heating_rows[-1]['rdu_count']:,} rdu but "
                         f"include_sample has {n_sample:,}")

    # Step 4 -- removed counts, then the AC split of the sample (not filters)
    df_funnel = pd.concat([df_scope, pd.DataFrame(heating_rows)], ignore_index=True)
    # previous minus current, so a step that removes nothing shows 0, not -0.0
    df_funnel['removed_rdu'] = df_funnel['rdu_count'].shift() - df_funnel['rdu_count']
    df_funnel['removed_homes'] = (
        df_funnel['weighted_count'].shift() - df_funnel['weighted_count'])
    has_ac = df_enduse['include_cooling'].astype(bool)
    ac_rows = [
        compute_funnel_stage_row(df_enduse, 'sample_with_ac', sample & has_ac),
        compute_funnel_stage_row(df_enduse, 'sample_no_ac', sample & ~has_ac),
    ]
    return pd.concat([df_funnel, pd.DataFrame(ac_rows)], ignore_index=True)
