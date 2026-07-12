"""Copy-paste verification cell: June 2026 rebate fossil gate (MP4).

NOT run automatically and NOT part of the .ipynb. Paste the CELL block below
into the main notebook AFTER the June 2026 rebate path and its downstream
economic-adopter step have run, so the frame carries:
  - mp4_heating_rebate_amount_june2026_v4MID
  - mp4_rebate_eligibility_june2026
  - ref2025_mp4_heatingLCC_coolingSavings_sub_june2026_econ_adopter_fixed_base

Pass rule:
  - by_fuel: every NON-electric baseline row must be $0 in BOTH 'total_eligible'
    and 'adopters_only'. A nonzero fossil row means the fuel gate has a real bug.
  - 'adopters_only' national total is the figure to compare against the ~$8-9B
    appropriation; 'total_eligible' is the uncapped potential (no funding cap
    is modeled).
"""

# ===== CELL (paste into the notebook) =====
from cmu_tare_model.private_impact.data_processing.determine_rebate_eligibility_and_amount import (
    summarize_rebate_funding,
)
from cmu_tare_model.utils.column_names import create_adoption_col
from cmu_tare_model.utils.modeling_params import define_scenario_params

_MP = 4
_COST = 'v4MID'
_METHOD_SUFFIX = '_fixed_base'
_WEIGHT_COL = 'weight'  # adjust if the frame's household-weight column differs

# Hold the heating-replacement-credit scenario fixed; use its June 2026 adopter.
_prefix = define_scenario_params(_MP)[0]
_adopter_col = create_adoption_col(
    _prefix, 'heatingLCC_coolingSavings_sub_june2026', _METHOD_SUFFIX)

_df = DATAFRAMES_BY_MP[_MP]['fixed_base']  # the canonical MP4 frame in the notebook

by_program, by_fuel = summarize_rebate_funding(
    _df,
    menu_mp=_MP,
    cost_scenario=_COST,
    guidance='june2026',
    weight_col=_WEIGHT_COL,
    adopter_col=_adopter_col,
)

print('Adopter column:', _adopter_col)
print('\n--- June 2026 rebate funding by program (weighted $) ---')
print(by_program.round(0))
print('\n--- June 2026 rebate funding by baseline fuel (weighted $) ---')
print(by_fuel.round(0))

# Fossil-gate assertion: non-electric fuels must be exactly $0 under June 2026.
_fossil = by_fuel.drop(index='Electricity', errors='ignore')
_nonzero = _fossil[(_fossil != 0).any(axis=1)]
if len(_nonzero):
    print('\n[FAIL] Non-electric fuels received June 2026 rebate dollars:')
    print(_nonzero.round(2))
else:
    print('\n[PASS] Every non-electric baseline fuel is $0 under June 2026.')
