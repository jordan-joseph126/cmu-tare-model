"""Tests for adoption_reconciliation_table and the in-scope population used by
build_econ_plot_df (adoption_potential/data_processing/visuals_adoption_dotplot.py).
"""

import numpy as np
import pandas as pd
import pytest

from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    REBATE_POLICY_SCENARIO_ORDER,
    REPLACEMENT_CREDIT_SCOPES,
    adoption_reconciliation_table,
    build_econ_plot_df,
)

WEIGHT = 242.131013


@pytest.fixture
def adoption_df() -> pd.DataFrame:
    """Six homes: five in heating scope (two fuels plus electricity), one out of
    scope. Adopter values are NaN out of scope, as in the model output."""
    df = pd.DataFrame({
        'weight': WEIGHT,
        'include_heating': [True, True, True, True, True, False],
        'include_sample': [True, True, True, True, True, False],
        'base_heating_fuel': ['Electricity', 'Electricity', 'Natural Gas',
                              'Natural Gas', 'Propane', 'Electricity'],
        'lmi_or_mui': ['LMI', 'MUI', 'LMI', 'MUI', 'LMI', 'MUI'],
    }, index=pd.Index([10, 11, 12, 13, 14, 15], name='bldg_id'))
    unsub = [0.0, 0.0, 1.0, 0.0, 1.0, np.nan]
    sub = [1.0, 1.0, 1.0, 1.0, 1.0, np.nan]
    # June 2026: electric homes may gain; fossil homes keep their unsub status.
    june = [1.0, 0.0, 1.0, 0.0, 1.0, np.nan]
    by_token = {'unsub': unsub, 'sub': sub, 'sub_june2026': june}
    for scope, _label in REPLACEMENT_CREDIT_SCOPES:
        for token in REBATE_POLICY_SCENARIO_ORDER:
            df[f'ref2025_mp3_{scope}_{token}_econ_adopter_fixed_base'] = by_token[token]
    return df


def test_national_equals_fuel_rows_and_excludes_out_of_scope(adoption_df):
    """National adopters equal the fuel rows, and out-of-scope homes are not counted."""
    table = adoption_reconciliation_table(adoption_df, mp=3)
    case = table[table['npv_case'] == 'heatingLCC_coolingLCC_sub_june2026'].set_index('group')

    in_scope_m = 5 * WEIGHT / 1e6
    assert case.loc['National', 'homes_in_scope_m'] == pytest.approx(in_scope_m)
    assert case.loc['National', 'adopters_m'] == pytest.approx(3 * WEIGHT / 1e6)
    assert case.loc['National', 'share_pct'] == pytest.approx(60.0)
    fuel_rows = case.drop(index='National')
    assert fuel_rows['adopters_m'].sum() == pytest.approx(case.loc['National', 'adopters_m'])
    assert (case['nan_adopter_homes_m'] == 0).all()


def test_fossil_home_changing_under_june2026_raises(adoption_df):
    """A fossil-baseline home whose June 2026 status differs from unsubsidized fails the check."""
    col = 'ref2025_mp3_heatingLCC_coolingLCC_sub_june2026_econ_adopter_fixed_base'
    adoption_df.loc[13, col] = 1.0   # natural gas home, unsub 0 -> June 2026 1
    with pytest.raises(ValueError, match="fossil-baseline"):
        adoption_reconciliation_table(adoption_df, mp=3)
    # The check can be switched off for a package that funds fossil baselines.
    adoption_reconciliation_table(adoption_df, mp=3, check_june2026_fossil_rule=False)


def test_plot_df_homes_counts_cover_in_scope_homes_only(adoption_df):
    """Marker homes labels (rate x homes) equal adopters counted directly."""
    plot_df = build_econ_plot_df(adoption_df, mp=3, rebate_vintage='sub_june2026')
    rows = plot_df[plot_df['tier_label'] == 'Heating + Cooling Replacement Cost Offset'].set_index('grouping')

    national = rows.loc['National -- Overall']
    assert national['weighted_homes_millions'] == pytest.approx(5 * WEIGHT / 1e6)
    label_m = national['case_b_pct'] / 100 * national['weighted_homes_millions']
    assert label_m == pytest.approx(3 * WEIGHT / 1e6)

    fuel_labels = sum(
        rows.loc[g, 'case_b_pct'] / 100 * rows.loc[g, 'weighted_homes_millions']
        for g in rows.index if g.endswith('-- Overall') and not g.startswith('National'))
    assert fuel_labels == pytest.approx(label_m)
