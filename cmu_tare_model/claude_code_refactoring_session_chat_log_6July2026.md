I'll start with Task 1 — a read-only pipeline audit. Let me locate and read the relevant files.

Globpattern: "**/column_names.py"
Found 1 file
Globpattern: "**/calculate_lifetime_private_impact.py"
Found 1 file
Globpattern: "**/determine_economic_adoption_potential.py"
Found 1 file
Globpattern: "**/*.py"
Found 101 files
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\utils\column_names.py
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\private_impact\calculate_lifetime_private_impact.py
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_potential\determine_economic_adoption_potential.py
This is the core. Now let me read the remaining pipeline pieces and inventory the downstream consumers.

Grep "heating_only|heating_and_cooling|NPV_CASE_CATEGORIES|VALID_HVAC_REPLACEMENT_SCENARIOS|moreWTP|lessWTP|v4MID|iraRef|preIRA" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model, glob: **/*.py)
30 lines of output
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\grid_impact\peak_load_functions.py
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_kpis\compute_adoption_rate.py
Grep "heating_only|heating_and_cooling|npv_case|econ_adopter|create_adoption_col|create_npv|iraRef|preIRA|moreWTP|lessWTP|v4MID" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_potential\data_processing\visuals_adoption_potential.py)
20 lines of output
Grep "heating_only|heating_and_cooling|npv_case|econ_adopter|create_adoption_col|create_npv|iraRef|preIRA|moreWTP|lessWTP|v4MID" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_potential\data_processing\visuals_adoption_dotplot.py)
2 lines of output
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_potential\data_processing\visuals_adoption_potential.py (lines 2-96)
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_potential\data_processing\visuals_adoption_dotplot.py (lines 2-161)
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_kpis\bill_savings.py
Grep "heating_only|heating_and_cooling|VALID_HVAC_REPLACEMENT_SCENARIOS|NPV_CASE" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\constants.py)
3 lines of output
Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\constants.py (lines 206-235)
Grep "heating_only|heating_and_cooling|npv_case|econ_adopter|iraRef|preIRA|moreWTP|lessWTP|find_adoption_column|create_adoption_col" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_kpis\calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_5July2026.py)
27 lines of output
Grep "heating_only|heating_and_cooling|npv_case|econ_adopter|iraRef|preIRA|find_adoption_column|create_adoption_col|NPV_CASE" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\adoption_kpis\calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_28June2026.py)
25 lines of output
Bash Compare bill savings export lineage and git history
IN
cd "c:/Users/jorda/Desktop/Projects/cmu-tare-model" && git log --oneline -5 -- cmu_tare_model/adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_5July2026.py cmu_tare_model/adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_28June2026.py 2>/dev/null; echo "---sizes---"; wc -l cmu_tare_model/adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_*.py

OUT
a7718c1 Exported the notebook files that will be updated in the next refactoring session. Committing now so that changes can be tracked and used to updated the .ipynb files.
ec651cd Refactor model run output export and loading functions. Updated the run simulation and main notebook.
b6be433 Clean up data folder and update file paths. Data folder uploaded to Zenodo: https://doi.org/10.5281/zenodo.17509166
---sizes---
 1018 cmu_tare_model/adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_28June2026.py
  933 cmu_tare_model/adoption_kpis/calculate_postTARE_am_kpis_demand_bill_savings_EXPORT_5July2026.py
 1951 total
Grep "heating_only|heating_and_cooling|econ_adopter|private_npv|moreWTP|lessWTP|NPV_CASE|create_adoption_col|create_npv_case_col" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\tests, glob: **/*.py)
46 lines of output
I now have the full pipeline picture. Here is the Task 1 audit report. No edits made.

1. Term-by-term composition of the three CURRENT cases
From calculate_lifetime_private_impact.py:271-306, with moreWTP NPV = case_savings − case_net_capital, and:

total_capital = HP install − rebate
net_capital_heating = total_capital − heating_replacement_credit
net_capital_heating_and_cooling = net_capital_heating − cooling_replacement_credit (credit only when include_cooling=True)
Expanding NPV term-by-term (base = HP install cost as a negative, +rebate as a credit):

Current case	heating op. savings	cooling op. savings	HP capital	heating repl. credit	cooling repl. credit
heating_only	✓	✗	−	+	✗
heating_and_cooling_savings	✓	✓	−	+	✗
heating_and_cooling_full	✓	✓	−	+	+
2. Confirmed old → new mapping (derived from code, not assumed)
Reading the new tokens as: {end-use}LCC = credit that end use's avoided incumbent-equipment replacement (its lifecycle capital); {end-use}Savings = operating savings only, no replacement credit. All three new cases carry both heating and cooling operating savings.

New case	heating sav	cooling sav	HP cap	heat repl credit	cool repl credit	= which old case?	Value impact
heatingLCC_coolingSavings	✓	✓	−	+	✗	== heating_and_cooling_savings	Pure rename, value-preserving
heatingLCC_coolingLCC	✓	✓	−	+	+	== heating_and_cooling_full	Pure rename, value-preserving
heatingSavings_coolingLCC	✓	✓	−	✗	+	NO equivalent (new composition)	New; must be built
heating_only is retired outright — every new case includes cooling operating savings, which heating_only excluded. It maps to no new case.

3. NPV / adoption ordering for the NEW cases (re-derived — do NOT carry old inequalities over)
Let base = (heating_sav + cooling_sav) − HP + rebate. Then:

heatingLCC_coolingLCC = base + heat_repl + cool_repl → always the maximum
heatingLCC_coolingSavings = base + heat_repl
heatingSavings_coolingLCC = base + cool_repl
Consequences:

heatingLCC_coolingLCC ≥ heatingLCC_coolingSavings (adds cool_repl ≥ 0) ✓
heatingLCC_coolingLCC ≥ heatingSavings_coolingLCC (adds heat_repl ≥ 0) ✓
heatingLCC_coolingSavings vs heatingSavings_coolingLCC are NOT ordered in general (sign of heat_repl − cool_repl varies per home). The old strict NPV3≥NPV2≥NPV1 chain no longer holds.
For no-AC homes (include_cooling=False, cool_sav=cool_repl=0): heatingLCC_coolingLCC == heatingLCC_coolingSavings, and heatingSavings_coolingLCC == base (the lowest).
4. Golden-value impact
Golden value	Fate
Mean econ adoption 20.8%/20.5% (heating_only)	SUPERSEDED — defined on the retired case; no 1:1 new equivalent. Add superseded row per CLAUDE.md; new headline case is a researcher choice (likely heatingLCC_coolingLCC).
Operating-cost % (−38.5/−60.6), demand GWh, median demand %, symmetric norms, LMI shares	Independent of NPV naming → must stay byte-identical.
heating_and_cooling_savings / _full adoption values	Carried forward unchanged under new names (value-preserving).
5. Blocking finding — lessWTP collision
The target names drop the moreWTP token: ref2025_mp{mp}_{case}_private_npv_{discount_rate}. But lessWTP NPV columns are still produced at calculate_lifetime_private_impact.py:292,297-299,305 (and the legacy calculate_and_update_npv at line 701). If moreWTP is dropped while lessWTP remains, the two collide on one name. To drop the token, Task 3 must stop emitting lessWTP columns. v4MID is safe to drop — REMDB_COST_SCENARIO_KEYS = ['v4MID'] only (constants.py:205-207). No lessWTP/non-v4MID columns exist anywhere else in live code.

6. Old-style / hardcoded name inventory
Live code the rename must touch:

column_names.py:148-152 NPV_CASE_CATEGORIES + comment block 138-146; create_npv_case_col, create_adoption_col (bakes moreWTP, embeds cost_scenario)
calculate_lifetime_private_impact.py:271-308 npv_case_inputs keys + composition; lessWTP emission; legacy calculate_and_update_npv (likely orphaned — uses create_npv_col; flag)
determine_economic_adoption_potential.py:130-133 string-replace private_npv_moreWTP→econ_adopter_moreWTP (must become private_npv→econ_adopter)
peak_load_functions.py:45,69-74 find_adoption_column default npv_case="heating_only"
visuals_adoption_potential.py:34-36,77-80,603,618 hardcodes the ..._econ_adopter_moreWTP_{cost}_{disc} pattern (bypasses create_adoption_col) + default heating_only
visuals_adoption_dotplot.py two-marker Case A/Case B "pre-IRA vs IRA" structure — needs re-map decision (Task 4)
Tests: test_column_names.py:159-179, test_calculate_lifetime_private_impact.py:386-388,416-419,467-468, test_peak_load_functions.py:70,97
Flag, out of this session's scope (deferred prefix migration):

constants.py:226 VALID_HVAC_REPLACEMENT_SCENARIOS=['heating','heating_and_cooling'] — this is the internal capital-cost mechanic (hvac_replacement_scenario), a different axis from the public case tokens. Stays.
bill_savings.py:37 POLICY_SCENARIOS=("iraRef","preIRA") on fuel-cost columns — the rename doesn't touch fuel-cost names. Deferred iraRef/preIRA migration.
Lineage ambiguity: both ..._demand_bill_savings_EXPORT_28June2026.py and _5July2026.py carry iraRef/preIRA + two-scope drift; the 28June version calls the refactored find_adoption_column (line 440) while the 5July version uses a raw f-string. These are notebook exports (backport by hand). I'll resolve which is canonical at Task 4.
Before-snapshot captured
Golden values recorded from CLAUDE.md as the pre-edit baseline (adoption 20.8/20.5 on heating_only; operating-cost/demand/LMI values as listed).

Two load-bearing confirmations before any edit:

AskUserQuestion
OUT
Your questions have been answered: "Does my derived composition for the NEW third case `heatingSavings_coolingLCC` match your intent? (savings = heating + cooling operating savings; net capital = HP install − rebate − COOLING replacement credit ONLY; heating replacement NOT credited.) The other two are confirmed value-preserving renames of the current _savings and _full cases."="Yes, as derived", "The `moreWTP` name token can only be dropped if `lessWTP` columns stop being produced (otherwise they collide). Drop lessWTP entirely?"="Drop lessWTP columns". You can now continue with these answers in mind.
Both confirmed. Proceeding to Task 2 — column-name builders in column_names.py. Here are the proposed diffs (three cohesive, interdependent changes in one file). I'll apply only after your approval.

Diff 2a — NPV_CASE_CATEGORIES tuple + its comment block (lines 138-152):


# OLD
# The private NPV is computed under three distinct cost/savings cases. Each case
# label is carried as the "category" segment of the NPV and economic-adopter
# column names, so the existing builders and the econ-adopter name swap
# (private_npv_moreWTP -> econ_adopter_moreWTP) work without special-casing.
#
#   heating_only                 -> heating capital;          heating savings only
#   heating_and_cooling_savings  -> heating capital;          heating + cooling savings
#   heating_and_cooling_full     -> heating + cooling capital; heating + cooling savings
#
# Defined here as the single source of truth so no module hardcodes the strings.
NPV_CASE_CATEGORIES = (
    "heating_only",
    "heating_and_cooling_savings",
    "heating_and_cooling_full",
)

# NEW
# The private NPV is computed under three cases. All three count BOTH the
# heating and cooling operating (energy-bill) savings from the heat pump; the
# cases differ only in which capital costs enter the lifecycle calculation.
# In each case label, "LCC" means that end use's capital is included -- the
# avoided replacement of that end use's incumbent equipment is credited --
# while "Savings" means that end use contributes operating savings only.
#
#   heatingSavings_coolingLCC  -> credit avoided cooling replacement only
#   heatingLCC_coolingSavings  -> credit avoided heating replacement only
#   heatingLCC_coolingLCC      -> credit both avoided replacements
#
# The case label is carried as the "category" segment of the NPV and
# economic-adopter column names. Defined here as the single source of truth so
# no module hardcodes the strings.
NPV_CASE_CATEGORIES = (
    "heatingSavings_coolingLCC",
    "heatingLCC_coolingSavings",
    "heatingLCC_coolingLCC",
)
Diff 2b — create_npv_case_col drops wtp and cost_scenario params (lines 155-196):


# OLD signature + body
def create_npv_case_col(
    scenario_prefix: str,
    npv_case: str,
    wtp: str,
    cost_scenario: str,
    method_suffix: str) -> str:
    ...
    return create_npv_col(
        scenario_prefix=scenario_prefix, category=npv_case, wtp=wtp,
        cost_scenario=cost_scenario, method_suffix=method_suffix)

# NEW signature + body
def create_npv_case_col(
    scenario_prefix: str,
    npv_case: str,
    method_suffix: str) -> str:
    """Build a private NPV column name for one of the three NPV cases.

    The heat pump is the only heating technology, and the adoption decision uses
    a single willingness-to-pay framing and a single capital-cost scenario, so
    the NPV column name carries neither a WTP token nor a cost-scenario token.
    ... (Args: scenario_prefix, npv_case, method_suffix; Raises ValueError)
    Example -> 'ref2025_mp3_heatingLCC_coolingLCC_private_npv_fixed_base'
    """
    if npv_case not in NPV_CASE_CATEGORIES:
        raise ValueError(
            f"Invalid npv_case: '{npv_case}'. "
            f"Must be one of {NPV_CASE_CATEGORIES}")
    return f'{scenario_prefix}{npv_case}_private_npv{method_suffix}'
Diff 2c — create_adoption_col drops wtp and cost_scenario params (lines 374-417):


# OLD -> returns
#   f'{scenario_prefix}{npv_case}_econ_adopter_{wtp}_{cost_scenario}{method_suffix}'
# NEW signature + body
def create_adoption_col(
    scenario_prefix: str,
    npv_case: str,
    method_suffix: str,
) -> str:
    """...mirrors the private NPV column with private_npv -> econ_adopter.
    Example -> 'ref2025_mp3_heatingLCC_coolingLCC_econ_adopter_fixed_base'"""
    if npv_case not in NPV_CASE_CATEGORIES:
        raise ValueError(
            f"Invalid npv_case: '{npv_case}'. "
            f"Must be one of {NPV_CASE_CATEGORIES}")
    return f'{scenario_prefix}{npv_case}_econ_adopter{method_suffix}'
Note: create_npv_col (the low-level builder, still moreWTP/cost_scenario-aware) is intentionally left untouched — its only remaining live caller is the legacy calculate_and_update_npv, which I'll address in Task 3 (it's part of the lessWTP removal). Climate builders untouched.

Approve applying diffs 2a–2c to column_names.py?

Approve
Edit c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\utils\column_names.py
Added 4 lines
Edit c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\utils\column_names.py
Removed 7 lines
Edit c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\utils\column_names.py
Removed 5 lines
Diffs 2a–2c applied. Verifying the builders emit the exact target strings for mp3/mp4, all three cases, both families:

Bash Verify new column-name builders
IN
cd "c:/Users/jorda/Desktop/Projects/cmu-tare-model" && python -c "
from cmu_tare_model.utils.column_names import (
    NPV_CASE_CATEGORIES, create_npv_case_col, create_adoption_col)
print('CASES:', NPV_CASE_CATEGORIES)
for mp in (3, 4):
    for case in NPV_CASE_CATEGORIES:
        npv = create_npv_case_col(f'ref2025_mp{mp}_', case, method_suffix='_fixed_base')
        adopt = create_adoption_col(f'ref2025_mp{mp}_', case, method_suffix='_fixed_base')
        print('NPV  ', npv)
        print('ADOPT', adopt)
# invalid case still raises
try:
    create_npv_case_col('ref2025_mp3_', 'heating_only', method_suffix='_fixed_base')
    print('ERROR: old token accepted')
except ValueError:
    print('OK: old token heating_only rejected')
"

OUT
CASES: ('heatingSavings_coolingLCC', 'heatingLCC_coolingSavings', 'heatingLCC_coolingLCC')
NPV   ref2025_mp3_heatingSavings_coolingLCC_private_npv_fixed_base
ADOPT ref2025_mp3_heatingSavings_coolingLCC_econ_adopter_fixed_base
NPV   ref2025_mp3_heatingLCC_coolingSavings_private_npv_fixed_base
ADOPT ref2025_mp3_heatingLCC_coolingSavings_econ_adopter_fixed_base
NPV   ref2025_mp3_heatingLCC_coolingLCC_private_npv_fixed_base
ADOPT ref2025_mp3_heatingLCC_coolingLCC_econ_adopter_fixed_base
NPV   ref2025_mp4_heatingSavings_coolingLCC_private_npv_fixed_base
ADOPT ref2025_mp4_heatingSavings_coolingLCC_econ_adopter_fixed_base
NPV   ref2025_mp4_heatingLCC_coolingSavings_private_npv_fixed_base
ADOPT ref2025_mp4_heatingLCC_coolingSavings_econ_adopter_fixed_base
NPV   ref2025_mp4_heatingLCC_coolingLCC_private_npv_fixed_base
ADOPT ref2025_mp4_heatingLCC_coolingLCC_econ_adopter_fixed_base
OK: old token heating_only rejected
Task 2 verified. Builders emit the exact target strings for both MPs, all three cases, both families; old tokens are rejected.


NPV:    ref2025_mp{mp}_{case}_private_npv_fixed_base
ADOPT:  ref2025_mp{mp}_{case}_econ_adopter_fixed_base
cases:  heatingSavings_coolingLCC | heatingLCC_coolingSavings | heatingLCC_coolingLCC
Stop gate — Task 2 complete.

Next is Task 3 (NPV + adoption logic in calculate_lifetime_private_impact.py and determine_economic_adoption_potential.py): implement the confirmed term composition per case, drop lessWTP emission, and update the create_npv_case_col/create_adoption_col call sites to the new (no-wtp, no-cost_scenario) signatures. Shall I proceed to draft the Task 3 diffs?

Yes proceed.
Proceeding to Task 3. First, checking whether the legacy calculate_and_update_npv (which still emits lessWTP) is called anywhere in live code:

Grep "calculate_and_update_npv" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model, glob: **/*.py)
9 lines of output
Grep "create_npv_case_col|create_adoption_col|create_npv_col" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model, glob: **/*.py)
55 lines of output
Key finding: calculate_and_update_npv is test-only legacy (not in the live pipeline), and create_npv_col is still needed by the DEPRECATED module + EXPORT files, so both stay. But tare_run_simulation_..._EXPORT calls create_npv_case_col at 5 sites — let me see how, since my signature change affects them:

Read c:\Users\jorda\Desktop\Projects\cmu-tare-model\cmu_tare_model\model_scenarios\tare_run_simulation_v2_3_EXPORT_28June2026.py (lines 251-280)
The 5 create_npv_case_col sites in tare_run_simulation_..._EXPORT use the old signature — those are notebook-export files (Session B territory, backport-by-hand), so I'll flag them for backport rather than edit here. Let me confirm who calls economic_adoption_decision (to decide whether its now-vestigial cost_scenario param must stay for caller compatibility):

Grep "economic_adoption_decision" (in c:\Users\jorda\Desktop\Projects\cmu-tare-model, glob: **/*.py)
13 lines of output
Callers of economic_adoption_decision are all EXPORT/notebook files that pass cost_scenario=..., so I'll keep that param (now only documenting which cost-scenario NPV feeds the decision, no longer in the column name) to avoid breaking them.

Here are the Task 3 diffs. Starting with file A — calculate_lifetime_private_impact.py.

Diff 3A-1 — docstring (the case list + WTP paragraph + Returns, lines ~93-130):


# OLD
    A single heat pump replaces the home's heating system and also serves the
    cooling load, so all three cases share the same heating retrofit and differ
    only in whether cooling savings and the avoided cooling-system replacement
    are counted:

      - heating_only:                heating capital;           heating savings
      - heating_and_cooling_savings: heating capital;           heating + cooling savings
      - heating_and_cooling_full:    heating + cooling capital; heating + cooling savings

    Cooling savings and the cooling replacement credit are zero for homes with
    no AC (include_cooling = False), so for those homes Case 2 == Case 1 and
    Case 3 == Case 1.

    The adoption-relevant output is the moreWTP NPV (savings minus the net
    capital cost, i.e. the incremental cost over a like-for-like replacement).
    A lessWTP NPV (savings minus the full gross capital cost) is also retained
    for backward compatibility but is not used by the adoption decision.
# ...
    Returns:
        DataFrame with, per measure package, moreWTP and lessWTP NPV columns and
        capital cost columns for each of the three NPV cases (see
        NPV_CASE_CATEGORIES).

# NEW
    A single heat pump replaces the home's heating system and also serves the
    cooling load. All three cases count both the heating and cooling operating
    (energy-bill) savings; they differ only in which avoided-replacement credit
    reduces the net capital cost (see NPV_CASE_CATEGORIES):

      - heatingSavings_coolingLCC: credit avoided cooling replacement only
      - heatingLCC_coolingSavings: credit avoided heating replacement only
      - heatingLCC_coolingLCC:     credit both avoided replacements

    Cooling savings and the cooling replacement credit are zero for homes with
    no AC (include_cooling = False). For those homes heatingLCC_coolingLCC ==
    heatingLCC_coolingSavings, and heatingSavings_coolingLCC carries no credit.

    The NPV is the lifetime savings minus the incremental (net) capital cost of
    the heat pump over a like-for-like replacement. A single willingness-to-pay
    framing is modeled, so the NPV column name carries no WTP token, and the
    economic adoption decision adopts when this NPV >= 0.
# ...
    Returns:
        DataFrame with, per measure package, one private NPV column and a net
        capital cost column for each of the three NPV cases, plus the shared
        gross capital cost column (see NPV_CASE_CATEGORIES).
Diff 3A-2 — case assembly + loop (lines ~260-308):


# OLD
    # Case 3 also credits the avoided cooling-system replacement, but only for
    # homes that actually have AC (include_cooling = True).
    cooling_replacement_col = create_cost_col(
        menu_mp=menu_mp, category='cooling',
        cost_type='replacement', cost_scenario=cost_scenario)
    cooling_replacement_cost = (
        df_copy[cooling_replacement_col].fillna(0).where(include_cooling, other=0.0))
    net_capital_heating_and_cooling = net_capital_heating - cooling_replacement_cost

    # ===== Assemble the three NPV cases =====
    # Each entry maps an NPV case to its (savings, net capital cost) pair.
    npv_case_inputs = {
        'heating_only': (heating_savings, net_capital_heating),
        'heating_and_cooling_savings': (
            heating_and_cooling_savings, net_capital_heating),
        'heating_and_cooling_full': (
            heating_and_cooling_savings, net_capital_heating_and_cooling),
    }

    # The shared gross capital is stored once under the heating category.
    total_capital_col = create_capital_col(
        scenario_prefix=scenario_prefix, category='heating',
        net=False, cost_scenario=cost_scenario)
    df_new_columns[total_capital_col] = total_capital
    all_columns_to_mask['heating'].append(total_capital_col)

    for npv_case, (case_savings, case_net_capital) in npv_case_inputs.items():
        # moreWTP NPV: savings minus the incremental (net) capital cost. This is
        # the value the economic adoption decision uses (moreWTP >= 0).
        npv_more_wtp = round(case_savings - case_net_capital, 2)
        # lessWTP NPV: savings minus the full (gross) capital cost. Retained for
        # backward compatibility; not used by the economic adoption decision.
        npv_less_wtp = round(case_savings - total_capital, 2)

        more_wtp_col = create_npv_case_col(
            scenario_prefix=scenario_prefix, npv_case=npv_case, wtp='moreWTP',
            cost_scenario=cost_scenario, method_suffix=method_suffix)
        less_wtp_col = create_npv_case_col(
            scenario_prefix=scenario_prefix, npv_case=npv_case, wtp='lessWTP',
            cost_scenario=cost_scenario, method_suffix=method_suffix)
        net_capital_col = create_capital_col(
            scenario_prefix=scenario_prefix, category=npv_case,
            net=True, cost_scenario=cost_scenario)

        df_new_columns[more_wtp_col] = npv_more_wtp
        df_new_columns[less_wtp_col] = npv_less_wtp
        df_new_columns[net_capital_col] = case_net_capital
        all_columns_to_mask['heating'].extend(
            [more_wtp_col, less_wtp_col, net_capital_col])

# NEW
    # Two of the three cases credit the avoided cooling-system replacement, but
    # only for homes that actually have AC (include_cooling = True).
    cooling_replacement_col = create_cost_col(
        menu_mp=menu_mp, category='cooling',
        cost_type='replacement', cost_scenario=cost_scenario)
    cooling_replacement_cost = (
        df_copy[cooling_replacement_col].fillna(0).where(include_cooling, other=0.0))
    net_capital_heating_and_cooling = net_capital_heating - cooling_replacement_cost
    # Cooling-only credit: heat-pump capital (net of rebate) credited against
    # the avoided AC replacement but NOT against the heating system it replaces.
    net_capital_cooling_only = total_capital - cooling_replacement_cost

    # ===== Assemble the three NPV cases =====
    # Every case counts both heating and cooling operating savings; the cases
    # differ only in which avoided-replacement credit reduces the net capital
    # cost (see NPV_CASE_CATEGORIES). Each entry maps a case to its
    # (savings, net capital cost) pair.
    npv_case_inputs = {
        # Credit avoided cooling replacement only (heating treated as savings).
        'heatingSavings_coolingLCC': (
            heating_and_cooling_savings, net_capital_cooling_only),
        # Credit avoided heating replacement only (cooling treated as savings).
        'heatingLCC_coolingSavings': (
            heating_and_cooling_savings, net_capital_heating),
        # Credit both avoided replacements.
        'heatingLCC_coolingLCC': (
            heating_and_cooling_savings, net_capital_heating_and_cooling),
    }

    # The shared gross capital is stored once under the heating category.
    total_capital_col = create_capital_col(
        scenario_prefix=scenario_prefix, category='heating',
        net=False, cost_scenario=cost_scenario)
    df_new_columns[total_capital_col] = total_capital
    all_columns_to_mask['heating'].append(total_capital_col)

    for npv_case, (case_savings, case_net_capital) in npv_case_inputs.items():
        # Private NPV: lifetime energy-bill savings minus the incremental (net)
        # capital cost of the heat pump over a like-for-like baseline
        # replacement. This is the value the economic adoption decision uses
        # (NPV >= 0). A single willingness-to-pay framing is modeled, so the
        # column name carries no WTP token.
        npv_case_value = round(case_savings - case_net_capital, 2)

        npv_col = create_npv_case_col(
            scenario_prefix=scenario_prefix, npv_case=npv_case,
            method_suffix=method_suffix)
        net_capital_col = create_capital_col(
            scenario_prefix=scenario_prefix, category=npv_case,
            net=True, cost_scenario=cost_scenario)

        df_new_columns[npv_col] = npv_case_value
        df_new_columns[net_capital_col] = case_net_capital
        all_columns_to_mask['heating'].extend([npv_col, net_capital_col])
Approve applying diffs 3A-1 and 3A-2 to calculate_lifetime_private_impact.py? (File B — determine_economic_adoption_potential.py — will follow as the next gate.)

Approve
You've hit your session limit · resets 9:10pm (America/New_York)