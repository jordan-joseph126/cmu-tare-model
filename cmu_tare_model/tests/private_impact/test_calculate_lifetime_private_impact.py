"""Tests for calculate_lifetime_private_impact (private_impact/calculate_lifetime_private_impact.py)."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from cmu_tare_model.tests.conftest import (
    FULL_EQUIPMENT_SPECS, FULL_UPGRADE_COLUMNS, FULL_FUEL_MAPPING, BASE_YEAR,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def mock_constants(monkeypatch):
    """Mock constants with full production lifetimes."""
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', FULL_EQUIPMENT_SPECS)
    monkeypatch.setattr('cmu_tare_model.constants.UPGRADE_COLUMNS', FULL_UPGRADE_COLUMNS)
    monkeypatch.setattr('cmu_tare_model.constants.FUEL_MAPPING', FULL_FUEL_MAPPING)
    monkeypatch.setattr('cmu_tare_model.constants.VERBOSE', False)
    monkeypatch.setattr('cmu_tare_model.constants.PRIVATE_DISCOUNTING_METHOD_SUFFIXES', {
        'private_discount_rate_fixed_low': '_fixed_low',
        'private_discount_rate_fixed_base': '_fixed_base',
        'private_discount_rate_fixed_high': '_fixed_high',
        'private_discount_rate_variable': '_variable',
    })


@pytest.fixture
def private_impact_df():
    """DataFrame with columns required for capital cost / NPV unit tests (MP8)."""
    n = 6
    np.random.seed(42)
    data = {
        'state': ['CA', 'TX', 'NY', 'FL', 'IL', 'CA'],
        'census_division': ['Pacific', 'WSC', 'MA', 'SA', 'ENC', 'Pacific'],
        'include_heating': [True, True, False, True, True, False],
        'include_waterHeating': [True, False, True, True, False, True],
        'include_clothesDrying': [True, True, True, False, True, True],
        'include_cooking': [True, True, True, True, False, True],
        'valid_fuel_heating': [True, True, False, True, True, False],
        'valid_tech_heating': [True, True, False, True, True, False],
        'upgrade_hvac_heating_efficiency': ['ASHP', 'MSHP', None, 'ASHP', 'MSHP', None],
        'upgrade_water_heater_efficiency': ['HP', None, 'HP', None, None, 'HP'],
        'upgrade_clothes_dryer': [None, 'HP', None, None, 'HP', None],
        'upgrade_cooking_range': [None, 'Electric', None, 'Electric', None, 'Electric'],
        'private_discount_rate_fixed_low': [0.02] * n,
        'private_discount_rate_fixed_base': [0.07] * n,
        'private_discount_rate_fixed_high': [0.12] * n,
        'private_discount_rate_variable': np.random.uniform(0.07, 0.25, n),
    }

    # Add cost columns for MP8
    for cat in FULL_EQUIPMENT_SPECS:
        data[f'mp8_{cat}_upgrade_installed_cost_v4MID'] = np.random.uniform(5000, 15000, n)
        data[f'mp8_{cat}_replacement_installed_cost_v4MID'] = np.random.uniform(2000, 8000, n)
        data[f'mp8_{cat}_rebate_amount_v4MID'] = np.random.uniform(500, 3000, n)

    return pd.DataFrame(data)


@pytest.fixture
def fuel_costs_dfs(private_impact_df):
    """Create baseline and measure fuel cost DataFrames with annual columns."""
    n = len(private_impact_df)
    df_baseline = pd.DataFrame(index=private_impact_df.index)
    df_measure = pd.DataFrame(index=private_impact_df.index)

    for cat, lifetime in FULL_EQUIPMENT_SPECS.items():
        for year in range(1, lifetime + 1):
            year_label = year + (BASE_YEAR - 1)
            baseline_col = f'baseline_{year_label}_{cat}_fuel_cost'
            measure_col = f'ref2025_mp8_{year_label}_{cat}_fuel_cost'
            df_baseline[baseline_col] = np.random.uniform(500, 2000, n)
            df_measure[measure_col] = np.random.uniform(200, 1000, n)

        # Lifetime totals
        df_baseline[f'baseline_{cat}_lifetime_fuel_cost'] = np.random.uniform(5000, 25000, n)
        df_measure[f'ref2025_mp8_{cat}_lifetime_fuel_cost'] = np.random.uniform(2000, 15000, n)

    # Copy validation columns
    for col in private_impact_df.columns:
        if col.startswith(('include_', 'valid_')):
            df_baseline[col] = private_impact_df[col]
            df_measure[col] = private_impact_df[col]

    return df_baseline, df_measure


# =============================================================================
# _validate_required_columns
# =============================================================================

def test_validate_required_columns_all_present(private_impact_df):
    """Returns empty list when all columns exist."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import _validate_required_columns

    required = ['state', 'census_division']
    missing = _validate_required_columns(private_impact_df, required, 'test context')
    assert missing == []


def test_validate_required_columns_some_missing(private_impact_df):
    """Returns list of missing columns."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import _validate_required_columns

    required = ['state', 'nonexistent_col1', 'nonexistent_col2']
    missing = _validate_required_columns(private_impact_df, required, 'test context')
    assert 'nonexistent_col1' in missing
    assert 'nonexistent_col2' in missing
    assert 'state' not in missing


# =============================================================================
# calculate_capital_costs
# =============================================================================

def test_calculate_capital_costs_excludes_installation_premium(private_impact_df):
    """Single scenario: heating total = upgrade - rebate, with no installation premium."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import calculate_capital_costs

    valid_mask = private_impact_df['include_heating']

    total, net = calculate_capital_costs(
        df_copy=private_impact_df,
        category='heating',
        input_mp='upgrade03',
        menu_mp=8,
        policy_scenario='2025 Reference Case',
        cost_scenario='v4MID',
        valid_mask=valid_mask,
    )

    # Invalid homes should be NaN.
    assert total.loc[~valid_mask].isna().all()
    assert net.loc[~valid_mask].isna().all()

    # MP8 is rebate-eligible and input_mp='upgrade03' adds no weatherization, so
    # total = upgrade - rebate with no installation premium term included.
    expected_total = (
        private_impact_df['mp8_heating_upgrade_installed_cost_v4MID']
        - private_impact_df['mp8_heating_rebate_amount_v4MID'])
    for idx in valid_mask[valid_mask].index:
        assert total.loc[idx] == pytest.approx(expected_total.loc[idx], abs=0.01)


def test_calculate_capital_costs_with_ira(private_impact_df):
    """Single scenario: heating total = upgrade - rebate; net = total - replacement."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import calculate_capital_costs

    valid_mask = private_impact_df['include_heating']

    total, net = calculate_capital_costs(
        df_copy=private_impact_df,
        category='heating',
        input_mp='upgrade03',
        menu_mp=8,
        policy_scenario='2025 Reference Case',
        cost_scenario='v4MID',
        valid_mask=valid_mask,
    )

    # Should produce finite values for valid homes
    assert total.loc[valid_mask].notna().all()
    # Net = total - replacement
    valid_idx = valid_mask[valid_mask].index
    for idx in valid_idx:
        expected_net = total.loc[idx] - private_impact_df.loc[idx, 'mp8_heating_replacement_installed_cost_v4MID']
        assert net.loc[idx] == pytest.approx(expected_net, abs=0.01)


def test_calculate_capital_costs_missing_columns_raises(private_impact_df):
    """Raises KeyError when required cost columns are missing."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import calculate_capital_costs

    valid_mask = private_impact_df['include_heating']
    df_no_cost = private_impact_df.drop(columns=['mp8_heating_upgrade_installed_cost_v4MID'])

    with pytest.raises(KeyError, match="Missing required columns"):
        calculate_capital_costs(
            df_copy=df_no_cost,
            category='heating',
            input_mp='upgrade03',
            menu_mp=8,
            policy_scenario='2025 Reference Case',
            cost_scenario='v4MID',
            valid_mask=valid_mask,
        )


# =============================================================================
# calculate_private_npv (three-case integration)
# =============================================================================

@pytest.fixture
def npv_cases_df():
    """6-home DataFrame for the three-case NPV path (heating + cooling, MP3).

    Homes 0-3 are valid heating retrofits; home 2 has no AC (include_cooling
    False); homes 4-5 are excluded (invalid heating / no retrofit). Cost values
    are fixed constants so the NPV arithmetic is deterministic.
    """
    n = 6
    data = {
        'include_heating': [True, True, True, True, False, True],
        'valid_fuel_heating': [True, True, True, True, False, True],
        'valid_tech_heating': [True, True, True, True, False, True],
        'upgrade_hvac_heating_efficiency': [
            'ASHP', 'ASHP', 'ASHP', 'ASHP', None, None],
        # Home 2 has no AC; the rest do.
        'include_cooling': [True, True, False, True, True, True],
        'private_discount_rate_fixed_base': [0.07] * n,
        # Fixed cost columns (MP3, v4MID).
        'mp3_heating_upgrade_installed_cost_v4MID': [12000.0] * n,
        'mp3_heating_replacement_installed_cost_v4MID': [5000.0] * n,
        # Cooling replacement is non-NaN even for the no-AC home, to prove the
        # include_cooling mask (not the data) zeroes the credit.
        'mp3_cooling_replacement_installed_cost_v4MID': [4000.0] * n,
    }
    return pd.DataFrame(data)


@pytest.fixture
def npv_cases_fuel_costs(npv_cases_df):
    """Baseline and measure annual fuel costs for the three-case NPV path.

    Baseline always exceeds measure, so per-year avoided cost is positive and
    the lifetime savings are deterministic. Home 2's cooling columns are NaN to
    mimic the no-AC masking applied upstream.
    """
    n = len(npv_cases_df)
    lifetime = 15  # EQUIPMENT_SPECS heating == cooling == 15
    df_baseline = pd.DataFrame(index=npv_cases_df.index)
    df_measure = pd.DataFrame(index=npv_cases_df.index)

    for year in range(BASE_YEAR, BASE_YEAR + lifetime):
        # Heating: avoided 600/yr for every home.
        df_baseline[f'baseline_{year}_heating_fuel_cost'] = [1000.0] * n
        df_measure[f'ref2025_mp3_{year}_heating_fuel_cost'] = [400.0] * n

        # Cooling: avoided 200/yr; home 2 (no AC) is NaN.
        df_baseline[f'baseline_{year}_cooling_fuel_cost'] = [500.0, 500.0, np.nan, 500.0, 500.0, 500.0]
        df_measure[f'ref2025_mp3_{year}_cooling_fuel_cost'] = [300.0, 300.0, np.nan, 300.0, 300.0, 300.0]

    return df_baseline, df_measure


@pytest.fixture
def heating_cooling_specs(mock_constants, monkeypatch):
    """Point EQUIPMENT_SPECS at the real heating+cooling spec for the NPV path.

    The autouse mock_constants fixture sets EQUIPMENT_SPECS to a spec without
    'cooling'; the three-case NPV path needs both heating and cooling, so this
    fixture overrides the module-level binding (depends on mock_constants to run
    after it).
    """
    specs = {'heating': 15, 'cooling': 15}
    monkeypatch.setattr('cmu_tare_model.constants.EQUIPMENT_SPECS', specs)
    monkeypatch.setattr(
        'cmu_tare_model.private_impact.calculate_lifetime_private_impact.EQUIPMENT_SPECS',
        specs,
    )
    return specs


@patch('cmu_tare_model.private_impact.calculate_lifetime_private_impact.define_scenario_params')
@patch('cmu_tare_model.private_impact.calculate_lifetime_private_impact.calculate_discount_factors')
def test_private_npv_three_cases_columns_and_dtype(mock_discount, mock_params, npv_cases_df, npv_cases_fuel_costs, heating_cooling_specs):
    """Produces one private NPV column per case (no WTP/cost-scenario token) as float64."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import calculate_private_npv
    from cmu_tare_model.utils.column_names import NPV_CASE_CATEGORIES

    df_baseline, df_measure = npv_cases_fuel_costs
    mock_params.return_value = ('ref2025_mp3_', 'MidCase', {}, {}, {})
    mock_discount.return_value = pd.Series(0.95, index=npv_cases_df.index)

    result = calculate_private_npv(
        df=npv_cases_df,
        df_fuel_costs=df_measure,
        df_baseline_costs=df_baseline,
        input_mp='upgrade03',
        menu_mp=3,
        policy_scenario='2025 Reference Case',
        discount_rate_col_name='private_discount_rate_fixed_base',
        cost_scenario='v4MID',
        base_year=BASE_YEAR,
        verbose=False,
    )

    for npv_case in NPV_CASE_CATEGORIES:
        col = f'ref2025_mp3_{npv_case}_private_npv_fixed_base'
        assert col in result.columns, f"Missing {col}"
        assert result[col].dtype == 'float64'


@patch('cmu_tare_model.private_impact.calculate_lifetime_private_impact.define_scenario_params')
@patch('cmu_tare_model.private_impact.calculate_lifetime_private_impact.calculate_discount_factors')
def test_private_npv_three_cases_ordering(mock_discount, mock_params, npv_cases_df, npv_cases_fuel_costs, heating_cooling_specs):
    """Per home: NPV1 <= NPV2 <= NPV3; no-AC home has NPV1 == NPV2 == NPV3."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import calculate_private_npv

    df_baseline, df_measure = npv_cases_fuel_costs
    mock_params.return_value = ('ref2025_mp3_', 'MidCase', {}, {}, {})
    mock_discount.return_value = pd.Series(0.95, index=npv_cases_df.index)

    result = calculate_private_npv(
        df=npv_cases_df,
        df_fuel_costs=df_measure,
        df_baseline_costs=df_baseline,
        input_mp='upgrade03',
        menu_mp=3,
        policy_scenario='2025 Reference Case',
        discount_rate_col_name='private_discount_rate_fixed_base',
        cost_scenario='v4MID',
        base_year=BASE_YEAR,
        verbose=False,
    )

    base = 'ref2025_mp3_{case}_private_npv_fixed_base'
    npv_sav_lcc = result[base.format(case='heatingSavings_coolingLCC_sub')]
    npv_lcc_sav = result[base.format(case='heatingLCC_coolingSavings_sub')]
    npv_lcc_lcc = result[base.format(case='heatingLCC_coolingLCC_sub')]

    valid = npv_cases_df['include_heating'] & \
        npv_cases_df['upgrade_hvac_heating_efficiency'].notna()

    # heatingLCC_coolingLCC credits both replacements -- always the highest.
    assert (npv_lcc_lcc[valid] >= npv_lcc_sav[valid]).all()
    assert (npv_lcc_lcc[valid] >= npv_sav_lcc[valid]).all()

    # Home 2 has no AC: cooling savings = 0, cooling replacement credit = 0.
    # heatingLCC_coolingLCC collapses to heatingLCC_coolingSavings.
    assert npv_lcc_lcc.iloc[2] == npv_lcc_sav.iloc[2]

    # Exact arithmetic spot-check on AC home 0:
    #   heating savings = 600 * 0.95 * 15 = 8550
    #   cooling savings = 200 * 0.95 * 15 = 2850
    #   total capital = 12000; net_capital_heating = 7000; net_capital_heating_and_cooling = 3000
    #   net_capital_cooling_only = 12000 - 4000 = 8000
    assert npv_sav_lcc.iloc[0] == pytest.approx(8550 + 2850 - 8000)   # 3400
    assert npv_lcc_sav.iloc[0] == pytest.approx(8550 + 2850 - 7000)   # 4400
    assert npv_lcc_lcc.iloc[0] == pytest.approx(8550 + 2850 - 3000)   # 8400


@patch('cmu_tare_model.private_impact.calculate_lifetime_private_impact.define_scenario_params')
@patch('cmu_tare_model.private_impact.calculate_lifetime_private_impact.calculate_discount_factors')
def test_private_npv_three_cases_invalid_homes_masked(mock_discount, mock_params, npv_cases_df, npv_cases_fuel_costs, heating_cooling_specs):
    """Excluded homes (invalid heating or no retrofit) are NaN in every case."""
    from cmu_tare_model.private_impact.calculate_lifetime_private_impact import calculate_private_npv
    from cmu_tare_model.utils.column_names import NPV_CASE_CATEGORIES

    df_baseline, df_measure = npv_cases_fuel_costs
    mock_params.return_value = ('ref2025_mp3_', 'MidCase', {}, {}, {})
    mock_discount.return_value = pd.Series(0.95, index=npv_cases_df.index)

    result = calculate_private_npv(
        df=npv_cases_df,
        df_fuel_costs=df_measure,
        df_baseline_costs=df_baseline,
        input_mp='upgrade03',
        menu_mp=3,
        policy_scenario='2025 Reference Case',
        discount_rate_col_name='private_discount_rate_fixed_base',
        cost_scenario='v4MID',
        base_year=BASE_YEAR,
        verbose=False,
    )

    excluded = ~(npv_cases_df['include_heating']
                 & npv_cases_df['upgrade_hvac_heating_efficiency'].notna())
    for npv_case in NPV_CASE_CATEGORIES:
        col = f'ref2025_mp3_{npv_case}_private_npv_fixed_base'
        assert result.loc[excluded, col].isna().all()
