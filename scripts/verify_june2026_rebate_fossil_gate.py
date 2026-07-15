"""Copy-paste verification cell: rebate fuel gates by program (MP4).

NOT run automatically and NOT part of the .ipynb. Paste the CELL block below into
the main notebook AFTER both rebate vintages and their downstream
economic-adopter steps have run, so the frame carries, for MP4:
  - mp4_heating_rebate_amount_v4MID              (2024, guidance-less)
  - mp4_rebate_eligibility_ira2024               (2024 program label)
  - mp4_heating_rebate_amount_june2026_v4MID     (June 2026)
  - mp4_rebate_eligibility_june2026              (June 2026 program label)
  - base_heating_fuel, weight

What changed (2026-07-14 consolidation): HOMES is fuel-neutral -- the
fossil-removal restriction is a HEEHR-only rule. The enduring correctness check
is therefore on HEEHR, not on "all non-electric fuels":

  - HEEHR must fund $0 for any fossil baseline UNDER JUNE 2026 (the HEEHR fuel
    gate). This is the hard assertion.
  - HOMES MAY fund fossil baselines above 150% AMI (fuel-neutral). Under 2024
    this is now active, so 2024 shows fossil HOMES dollars. Under June 2026 the
    HOMES pathway is still electric-gated THIS SESSION (a deferred byte-identity
    choice: making 2026 HOMES fuel-neutral would move the '_sub_june2026'
    golden), so June 2026 fossil HOMES is currently $0 -- reported, not asserted,
    because it is expected to become nonzero when that fix lands.
  - 2024 HEEHR funds fossil baselines by design (2024 has no HEEHR fuel gate).
"""

# ===== CELL (paste into the notebook) =====
import pandas as pd

_MP = 4
_COST = 'v4MID'
_WEIGHT_COL = 'weight'  # adjust if the frame's household-weight column differs
_FUEL_COL = 'base_heating_fuel'

_df = DATAFRAMES_BY_MP[_MP]['fixed_base']  # the canonical MP4 frame in the notebook


def _program_by_fuel(amount_col, elig_col):
    """Weighted rebate dollars pivoted program (rows) x baseline fuel (cols)."""
    w = _df[_WEIGHT_COL].fillna(0.0)
    tidy = pd.DataFrame({
        'program': _df[elig_col],
        'fuel': _df[_FUEL_COL],
        'dollars': _df[amount_col].fillna(0.0) * w,
    })
    tidy = tidy[tidy['program'].isin(['HEEHR', 'HOMES'])]
    return tidy.pivot_table(
        index='program', columns='fuel', values='dollars',
        aggfunc='sum', fill_value=0.0)


# June 2026 pivot.
_j = _program_by_fuel(
    f'mp{_MP}_heating_rebate_amount_june2026_{_COST}',
    f'mp{_MP}_rebate_eligibility_june2026')
print('--- June 2026 rebate dollars: program x baseline fuel ---')
print(_j.round(0))

# 2024 pivot (guidance-less amount column, ira2024 label).
_i = _program_by_fuel(
    f'mp{_MP}_heating_rebate_amount_{_COST}',
    f'mp{_MP}_rebate_eligibility_ira2024')
print('\n--- December 2024 rebate dollars: program x baseline fuel ---')
print(_i.round(0))

_fossil_cols = [c for c in _j.columns if c != 'Electricity']

# HARD CHECK 1 -- June 2026 HEEHR funds $0 for every fossil baseline (fuel gate).
if 'HEEHR' in _j.index:
    _heehr_fossil = _j.loc['HEEHR', _fossil_cols]
    if (_heehr_fossil != 0).any():
        print('\n[FAIL] June 2026 HEEHR funded a fossil baseline (fuel gate bug):')
        print(_heehr_fossil[_heehr_fossil != 0].round(2))
    else:
        print('\n[PASS] June 2026 HEEHR is $0 for every fossil baseline.')

# HARD CHECK 2 -- 2024 HOMES is fuel-neutral: fossil baselines above 150% AMI
# receive HOMES dollars now that 2024 HOMES is enabled.
_i_fossil_cols = [c for c in _i.columns if c != 'Electricity']
if 'HOMES' in _i.index and _i_fossil_cols:
    _homes_fossil_2024 = _i.loc['HOMES', _i_fossil_cols].sum()
    if _homes_fossil_2024 > 0:
        print('[PASS] 2024 HOMES funds fossil baselines (fuel-neutral): '
              f'${_homes_fossil_2024:,.0f}.')
    else:
        print('[WARN] 2024 HOMES shows no fossil dollars -- expected some once '
              '2024 HOMES is enabled (check percent_AMI / savings coverage).')

# REPORT (not asserted) -- June 2026 HOMES fossil dollars. Expected $0 this
# session (2026 HOMES still electric-gated); becomes nonzero when the deferred
# fuel-neutral fix lands.
if 'HOMES' in _j.index and _fossil_cols:
    _homes_fossil_2026 = _j.loc['HOMES', _fossil_cols].sum()
    print(f'[INFO] June 2026 HOMES fossil dollars = ${_homes_fossil_2026:,.0f} '
          '(expected $0 this session; deferred fuel-neutral fix will raise it).')
