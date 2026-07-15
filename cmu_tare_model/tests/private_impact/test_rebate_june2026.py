"""Tests for the consolidated rebate function (HEEHR + HOMES, both vintages).

Covers calculate_rebate_june2026 and calculate_rebateIRA (both thin wrappers over
calculate_rebate_program) in
private_impact/data_processing/determine_rebate_eligibility_and_amount.py and the
guidance token on create_rebate_col. Includes the 2024 HOMES addition (2024 is
now HEEHR + fuel-neutral HOMES) and locks the intentionally-deferred June 2026
HOMES electric gate.
"""

import numpy as np
import pandas as pd
import pytest

from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
    calculate_rebate_june2026,
)
from cmu_tare_model.utils.column_names import create_rebate_col


COST = 'v4MID'


def _rebate_col(mp):
    return f'mp{mp}_heating_rebate_amount_june2026_{COST}'


def _elig_col(mp):
    return f'mp{mp}_rebate_eligibility_june2026'


@pytest.fixture
def june2026_df():
    """Six valid heating retrofits spanning the June 2026 decision grid.

    Heating upgrade cost 10,000 and cooling upgrade cost 6,000 for every home,
    so HEEHR project cost = 10,000 and HOMES project cost = 16,000. Homes:
      0 electric,  50% AMI            -> HEEHR full (min(8000, 1.00*10000)=8000)
      1 electric, 120% AMI            -> HEEHR half (min(8000, 0.50*10000)=5000)
      2 electric, 200% AMI, 25% save  -> HOMES tier 1 (min(2000, 0.50*16000)=2000)
      3 electric, 200% AMI, 40% save  -> HOMES tier 2 (min(4000, 0.50*16000)=4000)
      4 electric, 200% AMI, 10% save  -> below 20% floor -> 0 / None
      5 fossil,    50% AMI            -> fuel gate -> 0 / None
    """
    n = 6
    data = {
        'include_heating': [True] * n,
        'valid_fuel_heating': [True] * n,
        'valid_tech_heating': [True] * n,
        'upgrade_hvac_heating_efficiency': ['ASHP'] * n,
        'base_heating_fuel': [
            'Electricity', 'Electricity', 'Electricity',
            'Electricity', 'Electricity', 'Natural Gas'],
        'state': ['CA', 'TX', 'NY', 'FL', 'IL', 'CA'],
        'percent_AMI': [50.0, 120.0, 200.0, 200.0, 200.0, 50.0],
        'mp4_heating_upgrade_installed_cost_v4MID': [10000.0] * n,
        'mp4_cooling_upgrade_installed_cost_v4MID': [6000.0] * n,
        'mp4_modeled_savings_frac': [0.50, 0.50, 0.25, 0.40, 0.10, 0.50],
        # MP3 mirrors MP4 so MP3 (now ENERGY STAR-respecified and rebate-eligible)
        # can be exercised on the same decision grid.
        'mp3_heating_upgrade_installed_cost_v4MID': [10000.0] * n,
        'mp3_cooling_upgrade_installed_cost_v4MID': [6000.0] * n,
        'mp3_modeled_savings_frac': [0.50, 0.50, 0.25, 0.40, 0.10, 0.50],
    }
    return pd.DataFrame(data)


def test_create_rebate_col_guidance_token():
    """The guidance token is inserted; default stays byte-identical."""
    assert create_rebate_col(4, 'heating', 'v4MID') == \
        'mp4_heating_rebate_amount_v4MID'
    assert create_rebate_col(4, 'heating', 'v4MID', guidance='june2026') == \
        'mp4_heating_rebate_amount_june2026_v4MID'


def test_june2026_amounts_across_fuel_and_income(june2026_df):
    """HEEHR/HOMES amounts and eligibility match the rule on each home."""
    result = calculate_rebate_june2026(
        df_results_IRA=june2026_df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)

    expected_amount = [8000.0, 5000.0, 2000.0, 4000.0, 0.0, 0.0]
    expected_elig = ['HEEHR', 'HEEHR', 'HOMES', 'HOMES', 'None', 'None']

    assert result[_rebate_col(4)].tolist() == expected_amount
    assert result[_elig_col(4)].tolist() == expected_elig
    assert result[_rebate_col(4)].dtype == 'float64'


def test_june2026_mp3_is_eligible(june2026_df):
    """MP3 is now rebate-eligible (12-Jul ENERGY STAR respecification).

    With MP3 in REBATE_ELIGIBLE_HEATING_MPS it earns the same amounts and
    eligibility labels as MP4 on the identical decision grid.
    """
    result = calculate_rebate_june2026(
        df_results_IRA=june2026_df, category='heating', menu_mp=3,
        cost_scenario=COST, verbose=False)

    expected_amount = [8000.0, 5000.0, 2000.0, 4000.0, 0.0, 0.0]
    expected_elig = ['HEEHR', 'HEEHR', 'HOMES', 'HOMES', 'None', 'None']
    assert result[_rebate_col(3)].tolist() == expected_amount
    assert result[_elig_col(3)].tolist() == expected_elig


def test_june2026_cooling_is_noop(june2026_df):
    """Cooling is covered by the heating rebate; the frame passes through."""
    before = june2026_df.copy()
    result = calculate_rebate_june2026(
        df_results_IRA=june2026_df, category='cooling', menu_mp=4,
        cost_scenario=COST, verbose=False)
    pd.testing.assert_frame_equal(result, before)


def test_june2026_excluded_home_is_nan():
    """An excluded home (no valid heating) is NaN, not 0, in both columns."""
    df = pd.DataFrame({
        'include_heating': [True, False],
        'valid_fuel_heating': [True, False],
        'valid_tech_heating': [True, False],
        'upgrade_hvac_heating_efficiency': ['ASHP', None],
        'base_heating_fuel': ['Electricity', 'Electricity'],
        'state': ['CA', 'CA'],
        'percent_AMI': [50.0, 50.0],
        'mp4_heating_upgrade_installed_cost_v4MID': [10000.0, 10000.0],
        'mp4_cooling_upgrade_installed_cost_v4MID': [6000.0, 6000.0],
        'mp4_modeled_savings_frac': [0.5, 0.5],
    })
    result = calculate_rebate_june2026(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)

    assert result[_rebate_col(4)].iloc[0] == 8000.0
    assert pd.isna(result[_rebate_col(4)].iloc[1])
    assert pd.isna(result[_elig_col(4)].iloc[1])


def test_june2026_south_dakota_excluded():
    """An otherwise-eligible electric MP4 home in SD gets no rebate."""
    df = pd.DataFrame({
        'include_heating': [True, True],
        'valid_fuel_heating': [True, True],
        'valid_tech_heating': [True, True],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'ASHP'],
        'base_heating_fuel': ['Electricity', 'Electricity'],
        'state': ['SD', 'MN'],
        'percent_AMI': [50.0, 50.0],
        'mp4_heating_upgrade_installed_cost_v4MID': [10000.0, 10000.0],
        'mp4_cooling_upgrade_installed_cost_v4MID': [6000.0, 6000.0],
        'mp4_modeled_savings_frac': [0.5, 0.5],
    })
    result = calculate_rebate_june2026(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)
    # SD home: excluded -> 0 / None. MN home: normal HEEHR full.
    assert result[_rebate_col(4)].iloc[0] == 0.0
    assert result[_elig_col(4)].iloc[0] == 'None'
    assert result[_rebate_col(4)].iloc[1] == 8000.0
    assert result[_elig_col(4)].iloc[1] == 'HEEHR'


def test_2024_south_dakota_excluded():
    """SD home gets no 2024 HEEHR rebate; a participating-state peer does."""
    from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
        calculate_rebateIRA,
    )
    df = pd.DataFrame({
        'include_heating': [True, True],
        'valid_fuel_heating': [True, True],
        'valid_tech_heating': [True, True],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'ASHP'],
        'income_level': ['Low-Income', 'Low-Income'],
        # percent_AMI is required by the consolidated rebate function (the real
        # pipeline sets it in calculate_percent_AMI, consistent with income_level).
        'percent_AMI': [50.0, 50.0],
        'state': ['SD', 'MN'],
        'mp4_heating_upgrade_installed_cost_v4MID': [10000.0, 10000.0],
    })
    result = calculate_rebateIRA(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)
    col = f'mp4_heating_rebate_amount_{COST}'
    assert result[col].iloc[0] == 0.00    # SD excluded
    assert result[col].iloc[1] == 8000.00  # MN full HEEHR


def test_summarize_june2026_rebate_totals(june2026_df):
    """Weighted HEEHR/HOMES dollar totals aggregate correctly."""
    from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
        summarize_june2026_rebate_totals,
    )
    df = june2026_df.copy()
    df['weight'] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    scored = calculate_rebate_june2026(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)
    by_state, national = summarize_june2026_rebate_totals(
        scored, menu_mp=4, cost_scenario=COST)
    # HEEHR homes 0,1 (CA,TX): (8000+5000)*2 = 26000.
    # HOMES homes 2,3 (NY,FL): (2000+4000)*2 = 12000. Homes 4,5 earn nothing.
    assert national['HEEHR'] == 26000.0
    assert national['HOMES'] == 12000.0
    assert national['total'] == 38000.0


def test_funding_by_program_and_fossil_gate_june2026(june2026_df):
    """June 2026: HEEHR/HOMES totals correct AND fossil baselines get $0."""
    from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
        summarize_rebate_funding,
    )
    df = june2026_df.copy()
    df['weight'] = 10.0
    scored = calculate_rebate_june2026(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)
    by_program, by_fuel = summarize_rebate_funding(
        scored, menu_mp=4, cost_scenario=COST, guidance='june2026')
    assert by_program.loc['HEEHR', 'total_eligible'] == 130000.0   # (8000+5000)*10
    assert by_program.loc['HOMES', 'total_eligible'] == 60000.0    # (2000+4000)*10
    # THE CORRECTNESS CHECK: fossil baseline receives nothing under June 2026.
    assert by_fuel.loc['Natural Gas', 'total_eligible'] == 0.0
    assert by_fuel.loc['Electricity', 'total_eligible'] == 190000.0


def test_funding_adopters_only_june2026(june2026_df):
    """adopters_only sums rebate dollars for adopters, per program."""
    from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
        summarize_rebate_funding,
    )
    df = june2026_df.copy()
    df['weight'] = 10.0
    df['adopt'] = [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # home 0 (HEEHR), home 2 (HOMES)
    scored = calculate_rebate_june2026(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)
    by_program, _ = summarize_rebate_funding(
        scored, menu_mp=4, cost_scenario=COST, guidance='june2026',
        adopter_col='adopt')
    assert by_program.loc['HEEHR', 'adopters_only'] == 80000.0   # home 0: 8000*10
    assert by_program.loc['HOMES', 'adopters_only'] == 20000.0   # home 2: 2000*10
    assert by_program.loc['HEEHR', 'total_eligible'] == 130000.0  # unchanged


def test_funding_2024_heehr_and_homes_fuel_neutral():
    """2024 now models HEEHR + fuel-neutral HOMES; HEEHR still allows fossil.

    summarize_rebate_funding reads the explicit 'ira2024' eligibility label
    (guidance=None points at the guidance-less amount column). HEEHR funds a
    fossil baseline (2024 has no HEEHR fuel gate); HOMES funds any fuel above
    150% AMI (fuel-neutral).
    """
    from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
        summarize_rebate_funding,
    )
    df = pd.DataFrame({
        'weight': [10.0, 10.0, 10.0],
        'base_heating_fuel': ['Electricity', 'Natural Gas', 'Natural Gas'],
        f'mp4_heating_rebate_amount_{COST}': [8000.0, 8000.0, 2000.0],
        # Explicit 2024 program labels: two HEEHR homes and one fossil HOMES home.
        'mp4_rebate_eligibility_ira2024': ['HEEHR', 'HEEHR', 'HOMES'],
    })
    by_program, by_fuel = summarize_rebate_funding(
        df, menu_mp=4, cost_scenario=COST, guidance=None)
    assert by_program.loc['HEEHR', 'total_eligible'] == 160000.0
    assert by_program.loc['HOMES', 'total_eligible'] == 20000.0   # fossil HOMES
    # Natural Gas total = one HEEHR ($8,000) + one HOMES ($2,000), x10 weight.
    assert by_fuel.loc['Natural Gas', 'total_eligible'] == 100000.0


def test_2024_homes_is_fuel_neutral(june2026_df):
    """2024 HOMES credits homes above 150% AMI regardless of baseline fuel.

    The June 2026 fossil-removal restriction is HEEHR-only, so 2024 HOMES is
    fuel-neutral. Home 5 in the fixture (fossil, 50% AMI) still routes to HEEHR
    under 2024 (no HEEHR fuel gate). A fossil home above 150% AMI with enough
    savings earns HOMES.
    """
    from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
        calculate_rebateIRA,
    )
    df = june2026_df.copy()
    # Turn home 4 (electric, 200% AMI, 10% savings -> below floor) into a fossil
    # home above 150% AMI with tier-2 savings so it must earn HOMES under 2024.
    df.loc[4, 'base_heating_fuel'] = 'Natural Gas'
    df.loc[4, 'mp4_modeled_savings_frac'] = 0.40

    result = calculate_rebateIRA(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)
    amt = f'mp4_heating_rebate_amount_{COST}'
    elig = 'mp4_rebate_eligibility_ira2024'
    # Home 4: fossil, >150% AMI, 40% savings -> HOMES tier 2 = min(4000, .5*16000).
    assert result[amt].iloc[4] == 4000.0
    assert result[elig].iloc[4] == 'HOMES'
    # Home 5: fossil, 50% AMI -> HEEHR full (2024 allows fuel switching).
    assert result[amt].iloc[5] == 8000.0
    assert result[elig].iloc[5] == 'HEEHR'


def test_june2026_homes_still_electric_gated(june2026_df):
    """June 2026 HOMES remains electric-gated this session (deferred fix).

    Locks the intentional byte-identity choice: making 2026 HOMES fuel-neutral
    would move the '_sub_june2026' golden and is deferred to the full-run
    re-derivation. A fossil home above 150% AMI earns $0 under June 2026.
    """
    df = june2026_df.copy()
    df.loc[4, 'base_heating_fuel'] = 'Natural Gas'
    df.loc[4, 'mp4_modeled_savings_frac'] = 0.40
    result = calculate_rebate_june2026(
        df_results_IRA=df, category='heating', menu_mp=4,
        cost_scenario=COST, verbose=False)
    # Fossil home above 150% AMI: HOMES electric gate -> $0 / None under June 2026.
    assert result[_rebate_col(4)].iloc[4] == 0.0
    assert result[_elig_col(4)].iloc[4] == 'None'
