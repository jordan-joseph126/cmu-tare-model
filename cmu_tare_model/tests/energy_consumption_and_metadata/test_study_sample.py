"""Tests for build_tare_sample_ids and build_sample_funnel (study_sample.py)."""

import pandas as pd
import pytest

from cmu_tare_model.energy_consumption_and_metadata.study_sample import (
    build_sample_funnel,
    build_tare_sample_ids,
)


@pytest.fixture
def df_enduse():
    """Five homes in two counties. Home 104 has valid heating but is outside the
    sample (for example, a package was not applied to it)."""
    df_homes = pd.DataFrame({
        'bldg_id': [105, 101, 102, 103, 104],
        'include_heating': [True, True, False, True, True],
        'include_sample': [True, True, False, True, False],
        'county_fips': ['42003', '06001', '06001', '42003', '42003'],
    })
    return df_homes.set_index('bldg_id')


def test_reads_the_flag_only(df_enduse):
    """The sample follows include_sample, not include_heating."""
    sample_ids = build_tare_sample_ids(df_enduse, release='2022.1.1')
    assert sample_ids['release'] == '2022.1.1'
    assert list(sample_ids['all']) == [101, 103, 105]
    assert 104 not in sample_ids['all']


def test_by_county_groups_and_covers_all(df_enduse):
    """County lists are sorted and together hold exactly the sample."""
    sample_ids = build_tare_sample_ids(df_enduse, release='2022.1.1')
    assert sample_ids['by_county'] == {'06001': [101], '42003': [103, 105]}
    in_counties = sorted(
        bldg_id for county_ids in sample_ids['by_county'].values()
        for bldg_id in county_ids)
    assert in_counties == list(sample_ids['all'])


def test_by_county_keeps_leading_zero(df_enduse):
    """A saved run reads county_fips back as a number; keys stay 5 digits."""
    df_enduse['county_fips'] = df_enduse['county_fips'].astype(int)
    sample_ids = build_tare_sample_ids(df_enduse, release='2022.1.1')
    assert set(sample_ids['by_county']) == {'06001', '42003'}


def test_missing_column_raises(df_enduse):
    with pytest.raises(KeyError, match='include_sample'):
        build_tare_sample_ids(
            df_enduse.drop(columns=['include_sample']), release='2022.1.1')


def test_empty_sample_raises(df_enduse):
    df_enduse['include_sample'] = False
    with pytest.raises(ValueError, match='empty'):
        build_tare_sample_ids(df_enduse, release='2022.1.1')


# =============================================================================
# build_sample_funnel
# =============================================================================

@pytest.fixture
def funnel_inputs():
    """Six in-scope homes, each removed at a known step:
    2 other fuel, 3 existing heat pump, 4 wall furnace, 6 package not applied.
    Homes 1 and 5 are the sample; 1 has AC, 5 does not."""
    df_enduse = pd.DataFrame({
        'bldg_id': [1, 2, 3, 4, 5, 6],
        'weight': [2.0] * 6,
        'base_heating_fuel': ['Natural Gas', 'Other Fuel', 'Electricity',
                              'Natural Gas', 'Natural Gas', 'Natural Gas'],
        'heating_type': ['Natural Gas Fuel Furnace', 'Other Fuel Furnace',
                         'Electricity ASHP', 'Natural Gas Fuel Wall/Floor Furnace',
                         'Natural Gas Fuel Furnace', 'Natural Gas Fuel Furnace'],
        'valid_fuel_heating': [True, False, True, True, True, True],
        'include_heating': [True, False, False, False, True, True],
        'include_sample': [True, False, False, False, True, False],
        'include_cooling': [True, False, True, False, False, True],
    }).set_index('bldg_id')
    applicable_bldg_ids = [pd.Index([1, 2, 3, 4, 5])]
    df_package = pd.DataFrame({
        'stage': ['load', 'applicability', 'housing_type'],
        'rdu_count': [9, 8, 5],
        'weighted_count': [18.0, 16.0, 10.0],
    })
    return df_package, applicable_bldg_ids, df_enduse


def test_funnel_steps_in_order(funnel_inputs):
    df_package, applicable_bldg_ids, df_enduse = funnel_inputs
    df_funnel = build_sample_funnel(
        [df_package, df_package.copy()], applicable_bldg_ids, df_enduse)
    assert list(df_funnel['stage']) == [
        'load', 'applicability', 'housing_type', 'heating_fuel',
        'no_existing_heat_pump', 'replaceable_heating_system',
        'sample_with_ac', 'sample_no_ac']
    assert list(df_funnel['rdu_count']) == [9, 8, 5, 4, 3, 2, 1, 1]
    assert list(df_funnel['removed_rdu'].iloc[1:6]) == [1, 3, 1, 1, 1]


def test_funnel_packages_must_agree(funnel_inputs):
    df_package, applicable_bldg_ids, df_enduse = funnel_inputs
    df_package_mismatched = df_package.copy()
    df_package_mismatched.loc[1, 'rdu_count'] = 7
    with pytest.raises(ValueError, match='disagree'):
        build_sample_funnel(
            [df_package, df_package_mismatched], applicable_bldg_ids, df_enduse)


def test_funnel_must_end_at_sample(funnel_inputs):
    df_package, applicable_bldg_ids, df_enduse = funnel_inputs
    df_enduse.loc[5, 'include_sample'] = False
    with pytest.raises(ValueError, match='include_sample'):
        build_sample_funnel([df_package], applicable_bldg_ids, df_enduse)
