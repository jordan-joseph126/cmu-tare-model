"""
Targeted county smoke test for Phase 2 validation.
Tests compute_thermal_cop(aggregation='county') against PA_COP_RANGES.
Run from project root: python cmu_tare_model/adoption_kpis/_smoke_test_county.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np

from config import PROJECT_ROOT
from cmu_tare_model.constants import PA_COP_RANGES
from cmu_tare_model.adoption_kpis.kpi_functions import (
    compute_thermal_cop,
    load_euss_baseline,
    load_euss_upgrade,
    HEATING_LOAD_COL,
    HP_BACKUP_ELEC_COL,
    HP_FANS_PUMPS_COL,
    COUNTY_COL,
    CLIMATE_ZONE_COL,
)

ALLEGHENY_GISJOIN = 'G4200030'
TEST_MPS = [3, 4]   # both have PA_COP_RANGES entries

print("=" * 60)
print("County smoke test — Allegheny County PA (G4200030)")
print("=" * 60)

print("\nLoading baseline (required columns only)...")
df_baseline = load_euss_baseline()

all_passed = True

for mp_num in TEST_MPS:
    upgrade_name = f'upgrade{mp_num:02d}'
    print(f"\nLoading {upgrade_name}...")
    df_upgrade = load_euss_upgrade(upgrade_name)

    print(f"Computing county-level thermal COP for MP{mp_num}...")
    cop_county = compute_thermal_cop(
        df_baseline, df_upgrade,
        fuel_filter='Natural Gas',
        aggregation='county',
        verbose=False,
    )

    # Golden state-level reference check (must not have drifted)
    cop_state = compute_thermal_cop(
        df_baseline, df_upgrade,
        fuel_filter='Natural Gas',
        aggregation='state',
        verbose=False,
    )
    pa_state_row = cop_state[cop_state['state'] == 'PA']
    if not pa_state_row.empty:
        pa_state_cop = pa_state_row['thermal_cop'].iloc[0]
        print(f"  State-level PA COP (MP{mp_num}): {pa_state_cop:.3f}  "
              f"(golden ref: {'2.020' if mp_num == 3 else '?'})")

    # Allegheny county check
    allegheny_row = cop_county[cop_county['county'] == ALLEGHENY_GISJOIN]
    mp_key = f'mp{mp_num}'

    if allegheny_row.empty:
        print(f"  [FAIL] Allegheny County not found in results "
              f"(total county rows: {len(cop_county)})")
        all_passed = False
        continue

    cop_val = allegheny_row['thermal_cop'].iloc[0]
    home_count = allegheny_row['home_count'].iloc[0]

    if mp_key in PA_COP_RANGES:
        lo, hi = PA_COP_RANGES[mp_key]
        ok = lo <= cop_val <= hi
        status = '[OK] ' if ok else '[FAIL]'
        if not ok:
            all_passed = False
        print(f"  {status} Allegheny PA COP (MP{mp_num}) = {cop_val:.3f}  "
              f"(expected [{lo}, {hi}], homes={home_count})")
    else:
        print(f"  [OK]  Allegheny PA COP (MP{mp_num}) = {cop_val:.3f}  "
              f"(no reference range, homes={home_count})")

print("\n" + "=" * 60)
if all_passed:
    print("[OK] All county smoke tests PASSED")
else:
    print("[FAIL] One or more county smoke tests FAILED")
print("=" * 60)
