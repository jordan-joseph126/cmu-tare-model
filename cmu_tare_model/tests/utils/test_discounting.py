"""Tests for cmu_tare_model.utils.discounting module.

Verifies discount rate preparation and discount factor calculations
across fixed and variable (AMI-based) methods.
"""

import pytest
import pandas as pd
import numpy as np

from cmu_tare_model.constants import ANCHOR_YEAR

from cmu_tare_model.utils.discounting import (
    prepare_discount_rates,
    calculate_discount_factors,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ami_df():
    """DataFrame with percent_AMI column for discount rate tests."""
    return pd.DataFrame({
        'percent_AMI': [0, 50, 100, 150, 200],
        'home_id': [1, 2, 3, 4, 5],
    })


# ── prepare_discount_rates ───────────────────────────────────────────────────

def test_prepare_discount_rates_adds_active_columns(ami_df):
    result = prepare_discount_rates(ami_df)
    # fixed_low and fixed_high are retired; only the active rates are produced.
    expected_cols = [
        'public_discount_rate',
        'private_discount_rate_fixed_base',
        'private_discount_rate_variable',
    ]
    for col in expected_cols:
        assert col in result.columns
    assert 'private_discount_rate_fixed_low' not in result.columns
    assert 'private_discount_rate_fixed_high' not in result.columns


def test_prepare_discount_rates_fixed_values(ami_df):
    result = prepare_discount_rates(ami_df)
    assert (result['public_discount_rate'] == 0.02).all()
    assert (result['private_discount_rate_fixed_base'] == 0.07).all()


def test_prepare_discount_rates_variable_inverse_relationship(ami_df):
    """Higher AMI should get LOWER variable discount rate."""
    result = prepare_discount_rates(ami_df)
    var_rate = result['private_discount_rate_variable']
    # AMI=0 gets max rate, AMI=150 gets min rate
    assert var_rate.iloc[0] > var_rate.iloc[3]


def test_prepare_discount_rates_variable_bounds(ami_df):
    """Variable rate should be clamped between min and max."""
    result = prepare_discount_rates(ami_df)
    var_rate = result['private_discount_rate_variable']
    assert var_rate.min() >= 0.07  # VARIABLE_RATE_MIN
    assert var_rate.max() <= 0.45  # VARIABLE_RATE_MAX


def test_prepare_discount_rates_above_ami_threshold(ami_df):
    """AMI above 150% should still get minimum variable rate."""
    result = prepare_discount_rates(ami_df)
    var_rate = result['private_discount_rate_variable']
    # AMI=200 (above threshold=150) should get min rate
    assert var_rate.iloc[4] == pytest.approx(0.07)


def test_prepare_discount_rates_zero_ami(ami_df):
    """AMI=0 should get maximum variable rate."""
    result = prepare_discount_rates(ami_df)
    var_rate = result['private_discount_rate_variable']
    assert var_rate.iloc[0] == pytest.approx(0.45)


def test_prepare_discount_rates_returns_copy(ami_df):
    """Should NOT modify the original DataFrame."""
    original_cols = set(ami_df.columns)
    prepare_discount_rates(ami_df)
    assert set(ami_df.columns) == original_cols


def test_prepare_discount_rates_missing_ami_raises():
    df = pd.DataFrame({'home_id': [1, 2]})
    with pytest.raises(ValueError, match="percent_AMI"):
        prepare_discount_rates(df)


# ── calculate_discount_factors ───────────────────────────────────────────────

def test_discount_factors_same_year(ami_df):
    """Discount factor for same year should be 1.0."""
    df = prepare_discount_rates(ami_df)
    factors = calculate_discount_factors(df, base_year=ANCHOR_YEAR, target_year=ANCHOR_YEAR,
                                          discount_rate_col_name='public_discount_rate')
    np.testing.assert_allclose(factors.values, 1.0)


def test_discount_factors_future_year(ami_df):
    """Discount factor decreases with time: PV = FV / (1+r)^t."""
    df = prepare_discount_rates(ami_df)
    factors = calculate_discount_factors(df, base_year=ANCHOR_YEAR,
                                          target_year=ANCHOR_YEAR + 6,
                                          discount_rate_col_name='public_discount_rate')
    years = 6
    expected = 1 / ((1 + 0.02) ** years)
    np.testing.assert_allclose(factors.values, expected)


def test_discount_factors_returns_series(ami_df):
    """Always returns a Series, not scalar."""
    df = prepare_discount_rates(ami_df)
    factors = calculate_discount_factors(df, base_year=ANCHOR_YEAR,
                                          target_year=ANCHOR_YEAR + 1,
                                          discount_rate_col_name='public_discount_rate')
    assert isinstance(factors, pd.Series)
    assert len(factors) == len(df)


def test_discount_factors_higher_rate_lower_factor(ami_df):
    """Higher discount rate should produce lower discount factor."""
    df = prepare_discount_rates(ami_df)
    # Compare the public rate (2%) against the private fixed_base rate (7%).
    low_factors = calculate_discount_factors(df, ANCHOR_YEAR, ANCHOR_YEAR + 6,
                                          'public_discount_rate')
    high_factors = calculate_discount_factors(df, ANCHOR_YEAR, ANCHOR_YEAR + 6,
                                           'private_discount_rate_fixed_base')
    assert (low_factors > high_factors).all()


def test_discount_factors_missing_column_raises(ami_df):
    df = prepare_discount_rates(ami_df)
    with pytest.raises(ValueError, match="nonexistent_rate"):
        calculate_discount_factors(df, ANCHOR_YEAR, ANCHOR_YEAR + 6, 'nonexistent_rate')


def test_discount_factors_past_target_is_one(ami_df):
    """Target year before base year should clamp to 0 years difference (factor=1)."""
    df = prepare_discount_rates(ami_df)
    factors = calculate_discount_factors(df, base_year=ANCHOR_YEAR + 5,
                                          target_year=ANCHOR_YEAR,
                                          discount_rate_col_name='public_discount_rate')
    np.testing.assert_allclose(factors.values, 1.0)


def test_discount_factors_variable_rate_varies(ami_df):
    """With variable rates, factors differ per household."""
    df = prepare_discount_rates(ami_df)
    factors = calculate_discount_factors(df, ANCHOR_YEAR, ANCHOR_YEAR + 6,
                                       'private_discount_rate_variable')
    assert factors.nunique() > 1
