"""Tests for cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs module.

Verifies enclosure parameter retrieval for all retrofit types and
cost calculation with the validation framework.
"""

import pytest
import pandas as pd
import numpy as np


# ── get_enclosure_parameters ─────────────────────────────────────────────────

def test_attic_floor_insulation_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'upgrade_insulation_atticFloor': ['R-30', 'R-49', 'R-60'],
        'base_insulation_atticFloor': ['R-13', 'R-19', 'R-38'],
    })
    result = get_enclosure_parameters(df, 'insulation_atticFloor_upgradeCost')
    assert 'conditions' in result
    assert 'tech_eff_pairs' in result
    assert len(result['conditions']) == 14  # All R-30/R-49/R-60 combinations
    assert len(result['tech_eff_pairs']) == 14


def test_infiltration_reduction_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'upgrade_infiltration_reduction': ['30%'],
    })
    result = get_enclosure_parameters(df, 'infiltration_reduction_upgradeCost')
    assert len(result['conditions']) == 1
    assert result['tech_eff_pairs'][0][0] == 'Air Leakage Reduction: 30% Reduction'


def test_duct_sealing_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'upgrade_duct_sealing': ['10% Leakage, R-8'],
        'base_ducts': ['20% Leakage, R-4'],
    })
    result = get_enclosure_parameters(df, 'duct_sealing_upgradeCost')
    assert len(result['conditions']) == 3  # 10%, 20%, 30% leakage baseline


def test_wall_insulation_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'upgrade_insulation_wall': ['Wood Stud, R-13'],
    })
    result = get_enclosure_parameters(df, 'insulation_wall_upgradeCost')
    assert len(result['conditions']) == 1


def test_foundation_wall_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'upgrade_insulation_foundation_wall': ['Wall R-10, Interior'],
    })
    result = get_enclosure_parameters(df, 'insulation_foundation_wall_upgradeCost')
    assert len(result['conditions']) == 1


def test_rim_joist_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'base_insulation_foundation_wall': ['Uninsulated'],
        'base_foundation_type': ['Unvented Crawlspace'],
    })
    result = get_enclosure_parameters(df, 'insulation_rim_joist_upgradeCost')
    assert len(result['conditions']) == 1


def test_seal_crawlspace_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'upgrade_seal_crawlspace': ['Unvented Crawlspace'],
    })
    result = get_enclosure_parameters(df, 'seal_crawlspace_upgradeCost')
    assert len(result['conditions']) == 1


def test_roof_insulation_parameters():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({
        'upgrade_insulation_roof': ['Finished, R-30'],
    })
    result = get_enclosure_parameters(df, 'insulation_roof_upgradeCost')
    assert len(result['conditions']) == 1


def test_invalid_retrofit_col_raises():
    from cmu_tare_model.private_impact.calculations.calculate_enclosure_upgrade_costs import (
        get_enclosure_parameters,
    )
    df = pd.DataFrame({'col': [1]})
    with pytest.raises(ValueError, match="Invalid retrofit_col"):
        get_enclosure_parameters(df, 'invalid_col')
