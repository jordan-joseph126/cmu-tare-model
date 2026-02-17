"""
Test Suite: Efficiency Floor Refactoring (Steps 1–3)
=====================================================
Validates that:
  1. _apply_efficiency_floor() preserves original pm2 in a '_original' column
  2. add_remdb_metrics() includes '_original' in both df_main and df_detailed
  3. Module-level imports resolve correctly (no lazy imports inside functions)
  4. Cost calculations still use the FLOORED pm2, not the original
  5. Upgrade metrics do NOT create an '_original' column

Run from the project root:
    python -m pytest test_efficiency_floor_refactoring.py -v
    
Or paste individual test functions into a notebook cell and call them directly.

Requirements:
  - The REMDB v4 CSV must be accessible via load_remdb_v4_data()
  - Steps 1–3 of the refactoring must be applied to:
      remdb_v4_installed_cost_utils.py
      constants.py
"""

import pandas as pd
import numpy as np
import traceback
from typing import Dict, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Test utilities
# ─────────────────────────────────────────────────────────────────────────────

PASS = "\u2705"  # ✅
FAIL = "\u274C"  # ❌
SKIP = "\u26A0\uFE0F"   # ⚠️

def _print_result(test_name: str, passed: bool, detail: str = ""):
    """Print a formatted test result."""
    icon = PASS if passed else FAIL
    print(f"  {icon} {test_name}")
    if detail:
        print(f"       {detail}")


def _build_mock_heating_df(n: int = 100) -> pd.DataFrame:
    """Build a minimal DataFrame mimicking EUSS heating data.
    
    Creates homes with a range of heating types and efficiencies,
    including values below the efficiency floors (SEER 8, 10 for ASHPs;
    AFUE 60%, 68%, 76% for gas furnaces).
    
    Args:
        n: Number of homes (will be rounded to nearest multiple of 10).
        
    Returns:
        DataFrame with columns matching add_remdb_metrics() expectations.
    """
    rng = np.random.default_rng(42)
    n_per_group = max(n // 10, 1)
    
    rows = []
    
    # --- Gas furnace homes (various AFUE levels) ---
    for afue_str, count in [
        ('60% AFUE', n_per_group * 2),     # Below floor (0.80)
        ('68% AFUE', n_per_group),          # Below floor
        ('76% AFUE', n_per_group),          # Below floor
        ('80% AFUE', n_per_group),          # At floor
        ('92.5% AFUE', n_per_group),        # Above floor
    ]:
        for _ in range(count):
            rows.append({
                'base_heating_fuel': 'Natural Gas',
                'heating_type': 'Natural Gas Fuel Furnace',
                'hvac_has_ducts': 'Yes',
                'hvac_cooling_type': 'Central AC',
                'hvac_heating_efficiency': afue_str,
                'hvac_cooling_efficiency': f'SEER 13, 11.7 EER',
                'size_heating_system_primary_k_btu_h': rng.uniform(36, 120),
                'size_cooling_system_primary_k_btu_h': rng.uniform(18, 60),
            })
    
    # --- ASHP homes (various SEER levels) ---
    for seer_str, count in [
        ('SEER 8, 6.8 HSPF', n_per_group),     # Far below floor (15.0)
        ('SEER 10, 7.7 HSPF', n_per_group),     # Below floor
        ('SEER 13, 8.2 HSPF', n_per_group),     # Below floor
        ('SEER 15, 8.8 HSPF', n_per_group),     # At floor
    ]:
        for _ in range(count):
            rows.append({
                'base_heating_fuel': 'Electricity',
                'heating_type': 'Electricity ASHP',
                'hvac_has_ducts': 'Yes',
                'hvac_cooling_type': 'Heat Pump',
                'hvac_heating_efficiency': seer_str,
                'hvac_cooling_efficiency': seer_str,
                'size_heating_system_primary_k_btu_h': rng.uniform(18, 60),
                'size_cooling_system_primary_k_btu_h': rng.uniform(18, 60),
            })
    
    # return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    
    # Upgrade efficiency column required by add_remdb_metrics(metric_type='upgrade').
    # In the real pipeline this comes from the measure package definition
    # (e.g., SEER 18 for MP3, SEER 21 for MP8). Use a fixed value for testing.
    df['upgrade_hvac_heating_efficiency'] = 'SEER 18, 10 HSPF'
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Import validation (Step 3)
# ─────────────────────────────────────────────────────────────────────────────

def test_imports_resolve():
    """Verify that EFFICIENCY_FLOORS_PM2 is importable at module level."""
    results = []
    
    # Test 1a: Constants import
    try:
        from cmu_tare_model.constants import EFFICIENCY_FLOORS_PM2
        _print_result(
            "EFFICIENCY_FLOORS_PM2 imports from constants",
            True,
            f"Contains {len(EFFICIENCY_FLOORS_PM2)} equipment types"
        )
        results.append(True)
    except ImportError as e:
        _print_result("EFFICIENCY_FLOORS_PM2 imports from constants", False, str(e))
        results.append(False)
    
    # Test 1b: Verify expected keys exist
    try:
        from cmu_tare_model.constants import EFFICIENCY_FLOORS_PM2
        expected_keys = [
            'air_source_heat_pump_centrally_ducted',
            'air_conditioner_centrally_ducted',
            'furnaces_gas_furnace',
        ]
        missing = [k for k in expected_keys if k not in EFFICIENCY_FLOORS_PM2]
        passed = len(missing) == 0
        detail = "" if passed else f"Missing keys: {missing}"
        _print_result("EFFICIENCY_FLOORS_PM2 contains expected equipment types", passed, detail)
        results.append(passed)
    except Exception as e:
        _print_result("EFFICIENCY_FLOORS_PM2 contains expected equipment types", False, str(e))
        results.append(False)
    
    # Test 1c: No lazy import inside add_remdb_metrics
    try:
        import inspect
        from cmu_tare_model.utils.remdb_v4_installed_cost_utils import add_remdb_metrics
        source = inspect.getsource(add_remdb_metrics)
        has_lazy_import = 'from cmu_tare_model.constants import EFFICIENCY_FLOORS_PM2' in source
        passed = not has_lazy_import
        detail = "Lazy import still present inside function body" if not passed else ""
        _print_result("No lazy import of EFFICIENCY_FLOORS_PM2 inside add_remdb_metrics()", passed, detail)
        results.append(passed)
    except Exception as e:
        _print_result("No lazy import of EFFICIENCY_FLOORS_PM2 inside add_remdb_metrics()", False, str(e))
        results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: _apply_efficiency_floor() preserves original (Step 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_efficiency_floor_preserves_original():
    """Verify that _apply_efficiency_floor() creates a '_original' column
    and that original values are preserved while pm2 is clamped."""
    results = []
    
    try:
        from cmu_tare_model.utils.remdb_v4_installed_cost_utils import _apply_efficiency_floor
        from cmu_tare_model.constants import EFFICIENCY_FLOORS_PM2
    except ImportError as e:
        _print_result("Import _apply_efficiency_floor", False, str(e))
        return False
    
    # Build a small test DataFrame with known values
    df_test = pd.DataFrame({
        'row_id': [
            'furnaces_gas_furnace',
            'furnaces_gas_furnace',
            'furnaces_gas_furnace',
            'air_conditioner_centrally_ducted',
            'air_conditioner_centrally_ducted',
            'air_source_heat_pump_centrally_ducted',
        ],
        'pm2': [
            0.60,   # Gas furnace, AFUE 60% → should clamp to 0.80
            0.80,   # Gas furnace, AFUE 80% → at floor, no change
            0.925,  # Gas furnace, AFUE 92.5% → above floor, no change
            8.0,    # Central AC, SEER 8 → should clamp to 15.0
            15.0,   # Central AC, SEER 15 → at floor, no change
            10.0,   # ASHP, SEER 10 → should clamp to 15.0
        ]
    })
    
    # Store expected original values
    expected_originals = df_test['pm2'].copy()
    
    # Apply efficiency floor
    df_result = _apply_efficiency_floor(
        df=df_test,
        row_id_col='row_id',
        pm2_col='pm2',
        efficiency_floors=EFFICIENCY_FLOORS_PM2,
        verbose=False
    )
    
    # Test 2a: '_original' column exists
    original_col = 'pm2_original'
    has_original = original_col in df_result.columns
    _print_result(
        f"'{original_col}' column created by _apply_efficiency_floor()",
        has_original,
        f"Columns in result: {list(df_result.columns)}" if not has_original else ""
    )
    results.append(has_original)
    
    if not has_original:
        print("       Skipping remaining tests (dependent on _original column)")
        return False
    
    # Test 2b: Original values preserved exactly
    originals_match = df_result[original_col].equals(expected_originals)
    _print_result(
        "Original pm2 values preserved in '_original' column",
        originals_match,
        "" if originals_match else (
            f"Expected: {expected_originals.tolist()}, "
            f"Got: {df_result[original_col].tolist()}"
        )
    )
    results.append(originals_match)
    
    # Test 2c: Clamped values are correct
    expected_clamped = [0.80, 0.80, 0.925, 15.0, 15.0, 15.0]
    clamped_match = np.allclose(df_result['pm2'].values, expected_clamped, rtol=1e-9)
    _print_result(
        "Clamped pm2 values match expected floors",
        clamped_match,
        "" if clamped_match else (
            f"Expected: {expected_clamped}, "
            f"Got: {df_result['pm2'].tolist()}"
        )
    )
    results.append(clamped_match)
    
    # Test 2d: Values AT or ABOVE floor are unchanged
    at_or_above = [1, 2, 4]  # indices where original >= floor
    unchanged = all(
        df_result.loc[i, 'pm2'] == df_result.loc[i, original_col]
        for i in at_or_above
    )
    _print_result(
        "Values at or above floor are NOT modified",
        unchanged
    )
    results.append(unchanged)
    
    # Test 2e: NaN handling — NaN pm2 should remain NaN, not get clamped
    df_with_nan = df_test.copy()
    df_with_nan.loc[0, 'pm2'] = np.nan
    df_nan_result = _apply_efficiency_floor(
        df=df_with_nan,
        row_id_col='row_id',
        pm2_col='pm2',
        efficiency_floors=EFFICIENCY_FLOORS_PM2,
        verbose=False
    )
    nan_preserved = pd.isna(df_nan_result.loc[0, 'pm2'])
    nan_original_preserved = pd.isna(df_nan_result.loc[0, original_col])
    _print_result(
        "NaN pm2 values remain NaN (not clamped to floor)",
        nan_preserved and nan_original_preserved,
        "" if (nan_preserved and nan_original_preserved) else
        f"pm2={df_nan_result.loc[0, 'pm2']}, original={df_nan_result.loc[0, original_col]}"
    )
    results.append(nan_preserved and nan_original_preserved)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: add_remdb_metrics() passes '_original' through (Step 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_add_remdb_metrics_outputs_original():
    """Verify that add_remdb_metrics() includes '_original' in both
    df_main and df_detailed for replacement metrics, but NOT for upgrade."""
    results = []
    
    try:
        from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
            add_remdb_metrics, load_remdb_v4_data
        )
    except ImportError as e:
        _print_result("Import add_remdb_metrics", False, str(e))
        return False
    
    # Load real REMDB data and build mock homes
    try:
        remdb_costs = load_remdb_v4_data()
        df_mock = _build_mock_heating_df(n=100)
    except Exception as e:
        _print_result("Load REMDB data + build mock DataFrame", False, str(e))
        return False
    
    # --- Test 3a: Replacement metrics include '_original' ---
    try:
        df_main, df_detailed = add_remdb_metrics(
            df=df_mock,
            remdb_v4_costs=remdb_costs,
            end_use='heating',
            metric_type='replacement',
            percentile='mid',
            verbose=False
        )
        
        pm2_col = 'heating_replacement_pm2_euss'
        original_col = f'{pm2_col}_original'
        
        # Check df_main
        in_main = original_col in df_main.columns
        _print_result(
            f"'{original_col}' present in df_main (replacement)",
            in_main,
            f"df_main columns with 'pm2': {[c for c in df_main.columns if 'pm2' in c]}" if not in_main else ""
        )
        results.append(in_main)
        
        # Check df_detailed
        in_detailed = original_col in df_detailed.columns
        _print_result(
            f"'{original_col}' present in df_detailed (replacement)",
            in_detailed,
            f"df_detailed columns with 'pm2': {[c for c in df_detailed.columns if 'pm2' in c]}" if not in_detailed else ""
        )
        results.append(in_detailed)
        
    except Exception as e:
        _print_result("Replacement metrics include '_original'", False, f"Exception: {e}")
        traceback.print_exc()
        results.append(False)
        results.append(False)
    
    # --- Test 3b: Upgrade metrics do NOT have '_original' ---
    try:
        df_main_upg, df_detailed_upg = add_remdb_metrics(
            df=df_mock,
            remdb_v4_costs=remdb_costs,
            end_use='heating',
            metric_type='upgrade',
            percentile='mid',
            verbose=False
        )
        
        pm2_upgrade_col = 'heating_upgrade_pm2_euss'
        original_upgrade_col = f'{pm2_upgrade_col}_original'
        
        not_in_main = original_upgrade_col not in df_main_upg.columns
        _print_result(
            f"'{original_upgrade_col}' NOT present in df_main (upgrade)",
            not_in_main,
            "Column unexpectedly present — floor should only apply to replacement" if not not_in_main else ""
        )
        results.append(not_in_main)
        
    except Exception as e:
        _print_result("Upgrade metrics do NOT have '_original'", False, f"Exception: {e}")
        results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Original vs clamped values are different where expected
# ─────────────────────────────────────────────────────────────────────────────

def test_clamped_vs_original_diverge():
    """Verify that for homes with below-floor efficiency, the clamped pm2
    differs from the original, and for above-floor homes they are equal."""
    results = []
    
    try:
        from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
            add_remdb_metrics, load_remdb_v4_data
        )
        from cmu_tare_model.constants import EFFICIENCY_FLOORS_PM2
    except ImportError as e:
        _print_result("Imports", False, str(e))
        return False
    
    remdb_costs = load_remdb_v4_data()
    df_mock = _build_mock_heating_df(n=100)
    
    try:
        df_main, df_detailed = add_remdb_metrics(
            df=df_mock,
            remdb_v4_costs=remdb_costs,
            end_use='heating',
            metric_type='replacement',
            percentile='mid',
            verbose=False
        )
    except Exception as e:
        _print_result("Run add_remdb_metrics (replacement)", False, str(e))
        return False
    
    pm2_col = 'heating_replacement_pm2_euss'
    original_col = f'{pm2_col}_original'
    row_id_col = 'row_id_heating_replacement'
    
    if original_col not in df_main.columns:
        _print_result("'_original' column exists for comparison", False,
                       "Column missing — cannot run divergence tests")
        return False
    
    # Test 4a: Some homes should have clamped != original (below-floor homes exist)
    differs = (df_main[pm2_col] != df_main[original_col])
    # Only compare where both are not NaN
    valid = df_main[pm2_col].notna() & df_main[original_col].notna()
    n_clamped = (differs & valid).sum()
    has_clamped = n_clamped > 0
    _print_result(
        f"Some homes have clamped pm2 != original ({n_clamped:,} homes)",
        has_clamped,
        "No divergence found — either no below-floor homes or floor not applied" if not has_clamped else ""
    )
    results.append(has_clamped)
    
    # Test 4b: All clamped values >= floor for their equipment type
    for row_id, floor in EFFICIENCY_FLOORS_PM2.items():
        mask = (df_main[row_id_col] == row_id) & df_main[pm2_col].notna()
        if mask.sum() == 0:
            continue
        min_pm2 = df_main.loc[mask, pm2_col].min()
        at_or_above = min_pm2 >= floor - 1e-9  # floating point tolerance
        _print_result(
            f"  {row_id}: min pm2 ({min_pm2:.2f}) >= floor ({floor})",
            at_or_above
        )
        results.append(at_or_above)
    
    # Test 4c: Original values include sub-floor values (confirms mock data worked)
    for row_id, floor in EFFICIENCY_FLOORS_PM2.items():
        mask = (df_main[row_id_col] == row_id) & df_main[original_col].notna()
        if mask.sum() == 0:
            continue
        min_original = df_main.loc[mask, original_col].min()
        has_sub_floor = min_original < floor
        _print_result(
            f"  {row_id}: original includes sub-floor values (min={min_original:.2f})",
            has_sub_floor,
            f"No sub-floor originals found — mock data may not have below-floor homes for this type" if not has_sub_floor else ""
        )
        results.append(has_sub_floor)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Cost calculation uses CLAMPED pm2 (not original)
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_uses_clamped_pm2():
    """Verify that the cost regression formula receives the floored pm2,
    not the original EUSS value, by checking that a SEER 8 home does NOT
    produce a cost consistent with SEER 8 in the regression."""
    results = []
    
    try:
        from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
            add_remdb_metrics, load_remdb_v4_data
        )
        from cmu_tare_model.constants import EFFICIENCY_FLOORS_PM2
    except ImportError as e:
        _print_result("Imports", False, str(e))
        return False
    
    remdb_costs = load_remdb_v4_data()
    
    # Build two identical homes — one at SEER 8 (will be floored to 15),
    # one already at SEER 15. After flooring, their pm2 should be identical
    # and therefore their v4 costs should be identical.
    df_pair = pd.DataFrame({
        'base_heating_fuel': ['Electricity', 'Electricity'],
        'heating_type': ['Electricity ASHP', 'Electricity ASHP'],
        'hvac_has_ducts': ['Yes', 'Yes'],
        'hvac_cooling_type': ['Heat Pump', 'Heat Pump'],
        'hvac_heating_efficiency': ['SEER 8, 6.8 HSPF', 'SEER 15, 8.8 HSPF'],
        'hvac_cooling_efficiency': ['SEER 8, 6.8 HSPF', 'SEER 15, 8.8 HSPF'],
        # Same capacity so costs should match exactly
        'size_heating_system_primary_k_btu_h': [36.0, 36.0],
        'size_cooling_system_primary_k_btu_h': [36.0, 36.0],
    })
    
    try:
        df_main, df_detailed = add_remdb_metrics(
            df=df_pair,
            remdb_v4_costs=remdb_costs,
            end_use='heating',
            metric_type='replacement',
            percentile='mid',
            verbose=False
        )
    except Exception as e:
        _print_result("Run add_remdb_metrics on pair", False, str(e))
        return False
    
    pm2_col = 'heating_replacement_pm2_euss'
    original_col = f'{pm2_col}_original'
    
    # Test 5a: Clamped pm2 values should be identical (both 15.0)
    pm2_vals = df_detailed[pm2_col].values
    pm2_match = np.allclose(pm2_vals[0], pm2_vals[1], rtol=1e-9)
    _print_result(
        f"SEER 8 home and SEER 15 home have same clamped pm2 ({pm2_vals[0]:.1f}, {pm2_vals[1]:.1f})",
        pm2_match
    )
    results.append(pm2_match)
    
    # Test 5b: Original values should differ (8.0 vs 15.0)
    if original_col in df_detailed.columns:
        orig_vals = df_detailed[original_col].values
        originals_differ = not np.isclose(orig_vals[0], orig_vals[1])
        _print_result(
            f"Original pm2 values differ ({orig_vals[0]:.1f} vs {orig_vals[1]:.1f})",
            originals_differ
        )
        results.append(originals_differ)
    else:
        _print_result("Original column exists in df_detailed", False)
        results.append(False)
    
    # Test 5c: Manually compute cost and verify it matches the SEER 15 cost,
    # NOT the SEER 8 cost.
    # Material_Price = (pm1 * pm1_coef) + (pm2 * pm2_coef) + intercept
    prefix = 'heating_replacement_'
    try:
        pm1 = df_detailed[f'{prefix}pm1_euss']
        pm1_coef = df_detailed[f'{prefix}pm1_coef_mid']
        pm2 = df_detailed[pm2_col]          # Should be [15.0, 15.0] after floor
        pm2_coef = df_detailed[f'{prefix}pm2_coef_mid']
        intercept = df_detailed[f'{prefix}intercept_mid']
        
        material_price = (pm1 * pm1_coef) + (pm2 * pm2_coef) + intercept
        
        costs_equal = np.allclose(material_price.iloc[0], material_price.iloc[1], rtol=1e-9)
        _print_result(
            f"Material price identical for both homes (${material_price.iloc[0]:,.0f}, ${material_price.iloc[1]:,.0f})",
            costs_equal,
            "If unequal, the SEER 8 home is NOT using the floored pm2" if not costs_equal else ""
        )
        results.append(costs_equal)
        
        # Sanity check: what WOULD the cost be with SEER 8?
        seer8_material = (pm1.iloc[0] * pm1_coef.iloc[0]) + (8.0 * pm2_coef.iloc[0]) + intercept.iloc[0]
        cost_not_seer8 = not np.isclose(material_price.iloc[0], seer8_material, rtol=1e-9)
        _print_result(
            f"Material price does NOT match SEER 8 regression (${seer8_material:,.0f} vs ${material_price.iloc[0]:,.0f})",
            cost_not_seer8,
            "Cost matches SEER 8 — floor clamping may not be applied" if not cost_not_seer8 else ""
        )
        results.append(cost_not_seer8)
        
    except KeyError as e:
        _print_result("Manual cost verification", False, f"Missing column: {e}")
        results.append(False)
    
    return all(results)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests() -> None:
    """Run all Step 1–3 refactoring tests and print summary."""
    
    print("=" * 70)
    print("  EFFICIENCY FLOOR REFACTORING — TEST SUITE (Steps 1–3)")
    print("=" * 70)
    
    test_functions = [
        ("Test 1: Module-level imports", test_imports_resolve),
        ("Test 2: _apply_efficiency_floor preserves original", test_efficiency_floor_preserves_original),
        ("Test 3: add_remdb_metrics outputs include '_original'", test_add_remdb_metrics_outputs_original),
        ("Test 4: Clamped vs original values diverge correctly", test_clamped_vs_original_diverge),
        ("Test 5: Cost regression uses clamped pm2", test_cost_uses_clamped_pm2),
    ]
    
    outcomes = {}
    for name, fn in test_functions:
        print(f"\n{'─' * 70}")
        print(f"  {name}")
        print(f"{'─' * 70}")
        try:
            passed = fn()
            outcomes[name] = passed
        except Exception as e:
            print(f"  {FAIL} UNHANDLED EXCEPTION: {e}")
            traceback.print_exc()
            outcomes[name] = False
    
    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    n_passed = sum(1 for v in outcomes.values() if v)
    n_total = len(outcomes)
    for name, passed in outcomes.items():
        icon = PASS if passed else FAIL
        print(f"  {icon} {name}")
    
    print(f"\n  Result: {n_passed}/{n_total} test groups passed")
    
    if n_passed == n_total:
        print(f"\n  {PASS} All tests passed — safe to proceed to Steps 4–5")
    else:
        print(f"\n  {FAIL} Some tests failed — review before proceeding")
    
    print(f"{'=' * 70}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    run_all_tests()
