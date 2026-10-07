"""Tests for adoption_reconciliation_table and the in-scope population used by
build_econ_plot_df (adoption_potential/data_processing/visuals_adoption_dotplot.py).
"""

import numpy as np
import pandas as pd
import pytest

from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot import (
    MI_TIER_NAMES,
    REBATE_POLICY_SCENARIO_ORDER,
    REPLACEMENT_CREDIT_SCOPES,
    adoption_reconciliation_table,
    build_econ_plot_df,
    prepare_plot_data,
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


def test_fossil_rule_is_skipped_for_a_dual_fuel_package(adoption_df, monkeypatch):
    """A dual-fuel package funds fossil baselines under June 2026 (G8), so the
    table skips rule 2 for it on its own; other packages keep the rule."""
    col = 'ref2025_mp3_heatingLCC_coolingLCC_sub_june2026_econ_adopter_fixed_base'
    adoption_df.loc[13, col] = 1.0   # natural gas home gains under June 2026
    monkeypatch.setattr(
        'cmu_tare_model.adoption_potential.data_processing.'
        'visuals_adoption_dotplot.is_dual_fuel_package', lambda mp: mp == 3)
    adoption_reconciliation_table(adoption_df, mp=3)
    # An explicit True still applies the rule.
    with pytest.raises(ValueError, match="fossil-baseline"):
        adoption_reconciliation_table(
            adoption_df, mp=3, check_june2026_fossil_rule=True)


def test_plot_df_needs_a_weight_column(adoption_df):
    """A frame with no weight column stops in the reconciliation table, with a
    KeyError naming 'weight'; no release's weight is assumed in its place."""
    with pytest.raises(KeyError, match='weight'):
        build_econ_plot_df(adoption_df.drop(columns=['weight']), mp=3)


def test_prepare_plot_data_reads_the_weight(adoption_df):
    """With no scaling_factor given, homes counts use the frame's own weight; a
    frame with no weight column, or with two weights, stops."""
    adopter_col = 'ref2025_mp3_heatingLCC_coolingLCC_unsub_econ_adopter_fixed_base'
    # A made-up weight that is neither release's (242.131013 or
    # 253.90367272727272), so a typed-in release weight would fail this test.
    frame_weight = 100.0
    source_df = adoption_df.dropna(subset=[adopter_col]).assign(weight=frame_weight)
    # The adoption-share frame as prepare_plot_data reads it: one row per
    # (fuel, income group) and one column per (adopter column, tier).
    share_index = pd.MultiIndex.from_tuples(
        [('Electricity', 'LMI')], names=['base_heating_fuel', 'lmi_or_mui'])
    share_columns = pd.MultiIndex.from_product([[adopter_col], MI_TIER_NAMES])
    df_shares = pd.DataFrame(50.0, index=share_index, columns=share_columns)

    plot_df = prepare_plot_data(df_shares, source_df, adopter_col, adopter_col)
    assert plot_df['sample_n'].tolist() == [1, 1, 2, 2, 5, 5]
    assert plot_df['weighted_homes_millions'].tolist() == pytest.approx(
        (plot_df['sample_n'] * frame_weight / 1e6).tolist())

    with pytest.raises(ValueError, match='weight'):
        prepare_plot_data(
            df_shares, source_df.drop(columns=['weight']), adopter_col, adopter_col)
    two_weights = source_df.assign(weight=[frame_weight] * 4 + [2 * frame_weight])
    with pytest.raises(ValueError, match='exactly one value'):
        prepare_plot_data(df_shares, two_weights, adopter_col, adopter_col)
