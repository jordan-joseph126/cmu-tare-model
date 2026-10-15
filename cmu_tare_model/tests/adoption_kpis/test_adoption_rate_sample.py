"""Tests that compute_adoption_rate is a share of study-sample homes
(adoption_kpis/compute_adoption_rate.py).

The adoption_kpis package imports its geospatial module on load, so these tests
need geopandas and are skipped where it is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")

from cmu_tare_model.adoption_kpis.compute_adoption_rate import compute_adoption_rate


@pytest.fixture
def df_homes():
    """County A: 2 sample homes (1 adopter) + 2 outside the sample.
    County B: only homes outside the sample."""
    return pd.DataFrame({
        'county': ['A', 'A', 'A', 'A', 'B'],
        'weight': [242.131013] * 5,
        'include_sample': [True, True, False, False, False],
        'adopter': [1.0, 0.0, np.nan, np.nan, np.nan],
    }, index=pd.Index([1, 2, 3, 4, 5], name='bldg_id'))


def test_rate_is_share_of_sample_homes(df_homes):
    df_rate = compute_adoption_rate(
        df_homes, adoption_col='adopter', adopter_tiers=[True])
    county_a = df_rate.set_index('county').loc['A']
    assert county_a['adoption_rate_pct'] == 50.0
    assert np.isclose(county_a['home_count'], 2 * 242.131013)


def test_county_without_sample_homes_is_left_out(df_homes):
    df_rate = compute_adoption_rate(
        df_homes, adoption_col='adopter', adopter_tiers=[True])
    assert 'B' not in set(df_rate['county'])


def test_missing_sample_flag_raises(df_homes):
    with pytest.raises(KeyError, match='include_sample'):
        compute_adoption_rate(
            df_homes.drop(columns=['include_sample']),
            adoption_col='adopter', adopter_tiers=[True])
