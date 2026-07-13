# Continuation Session Hand-off (2026-07-12)

Two independent pieces of work for a new refactoring session, with a read-only
codebase audit of the relevant files and exact locations. This document is
context for building the session prompt (via the `vscode-refactoring-session-prompt`
skill). Nothing in the codebase was edited to produce it.

**Recommended split:** these are two separate concerns and are best run as two
sessions.
- **Issue 2 (cooling savings)** is a correctness/methodology question that can
  move results -- do it first, on its own.
- **Issue 1 (capital-cost sensitivity + ENERGY STAR override + v3 cleanup)** is a
  larger, mostly value-moving effort; the dot-plot overlap is downstream of it and
  should be fixed only after the capital-cost work lands.

Within Issue 1, keep the three sub-tasks separated by blast radius:
- **A (ENERGY STAR override)** and re-enabling **low/high cost scenarios** are
  value-moving (golden values change).
- **C (v3 cleanup)** is pure code removal (no output change).
- The **dot-plot fix** is presentation-only.

---

## Hard constraints (from CLAUDE.md -- carry into the prompt)

- **Never edit:** `utils/validation_framework.py`; any `.ipynb` (backport by hand);
  the EUSS/TARE load cells and demand-computation cells (preserved regions);
  `fetch_aeo_data_and_project_EXPORT_24June2026.py`.
- One diff per stop gate; audit the real file state before each edit; ASCII only.
- Degree-day CSV year columns **must** be int-cast on read (already handled in
  `degree_day_consumption_utils.py`; preserve it).
- Never silently overwrite a golden value -- add a new row marked "supersedes" and
  keep the old row.
- Column-name tokens are load-bearing; do not change real DataFrame column strings
  unless the task is explicitly a value-moving modeling change.

---

## Issue 2 -- Cooling cost *increases* after retrofit (negative cooling savings)

### Symptom
For some homes, `ref2025_mp4_cooling_lifetime_savings_fuel_cost` is **negative**
(cooling cost rises after the heat-pump retrofit), which should never happen.
Observed examples:
- bldg 549993: baseline cooling lifetime = 2410.38, MP4 = 4300.43, savings = -1890.05
- bldg 549999: baseline cooling lifetime = 6716.26, MP4 = 7556.73, savings = -840.47

User's hypothesis: future-consumption CDD growth (climate change) was applied to
the heat pump but **not** to the baseline counterfactual, so the baseline stays
flat while the heat pump grows.

### Audit findings (these partly reframe the hypothesis -- verify first)
- **Both** baseline and retrofit cooling consumption are CDD-adjusted by the
  **same** factor (same census division, same year) in
  `degree_day_consumption_utils.py`:
  - baseline via `get_total_baseline_consumption` (L282-344), which calls
    `get_cdd_factor_for_year` for `category == 'cooling'`.
  - retrofit via `get_electricity_consumption_for_year` (L194-240), which also
    calls `get_cdd_factor_for_year`.
  So the baseline **is** CDD-adjusted in the current code -- the specific "baseline
  not adjusted" hypothesis does not match what is there. **Step 1 of the session is
  to confirm/refute this on the real data.**
- Because the same CDD factor scales both sides each year, the negative savings is
  driven by the **base-year ResStock consumption**: the MP4 upgrade run's cooling
  kWh exceeds the baseline run's cooling kWh for those homes. Source columns in
  `process_euss_data.py`:
  - `base_electricity_cooling_consumption = out.electricity.cooling.energy_consumption.kwh`
    (baseline run, L270)
  - `mp{mp}_cooling_consumption = out.electricity.cooling.energy_consumption.kwh`
    (upgrade run, L480)
  The MP/baseline ratio is roughly constant across years (consistent with
  symmetric CDD scaling), which points at the underlying EUSS delta, not a
  projection asymmetry.
- **Real methodological question for the session:** is the intended counterfactual
  that the existing equipment's cooling grows with CDD identically to the heat pump
  (current behavior), or something else? And how should homes where the EUSS
  upgrade increases cooling kWh be handled (flagged, floored at zero savings, or
  accepted as a real ResStock result)?
- **Year-offset in the printed columns** (e.g. `baseline_2035_cooling_fuel_cost`
  shown next to `ref2025_mp4_2036_cooling_fuel_cost`): both cost loops use
  `year_label = year + 2023` over the same `EQUIPMENT_SPECS[category]` lifetime, so
  this looks like a cosmetic column-interleaving artifact in the printout, not a
  true off-by-one. Confirm there is no real misalignment, but it is likely
  display-only.
- Savings sign convention: `savings = baseline - measure` (via
  `calculate_avoided_values`), confirmed by the numbers above.

### Relevant files
| File | Role |
|---|---|
| `cmu_tare_model/private_impact/calculate_lifetime_fuel_costs.py` | Annual + lifetime fuel cost; per-year loop; savings via `calculate_avoided_values` |
| `cmu_tare_model/utils/degree_day_consumption_utils.py` | HDD/CDD projection factors applied to baseline **and** retrofit |
| `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | Origin of `base_electricity_cooling_consumption` and `mp{mp}_cooling_consumption` |
| `cmu_tare_model/data/projections/aeo2026_degree_day_factors_2025_2050.csv` | CDD/HDD factors (year columns must be int-cast) |
| `cmu_tare_model/constants.py` | `EQUIPMENT_SPECS` lifetimes (year-range check) |
| `cmu_tare_model/private_impact/data_processing/create_lookup_fuel_prices.py` | Electricity price (identical on both sides for cooling -- rules price out as the cause) |
| `cmu_tare_model/tests/utils/test_degree_day_consumption_utils.py` | Existing projection tests to extend |
| `cmu_tare_model/utils/validation_framework.py` | `calculate_avoided_values` -- **read-only reference; NEVER edit** |

---

## Issue 1 -- Capital-cost sensitivity + ENERGY STAR override + v3 cleanup (MP3)

Scope is the capital-cost section of the scenarios export
(`tare_scenarios_v2_3_EXPORT_12July2026.py`, ~L353-618) and its upstream cost
modules. The dot-plot overlap is expected to shift once this lands, so treat the
plot as downstream.

### Current state found in the audit
- The cost "sensitivity" is presently a single point:
  `REMDB_COST_SCENARIO_KEYS = ['v4MID']` -- `v4LOW` and `v4HIGH` are **commented
  out** in `constants.py` (L259-262). A real low/mid/high capital-cost sensitivity
  means re-enabling those and confirming the loop plus downstream NPV/adopter
  columns carry all three suffixes.
- Cost column name pattern: `mp{mp}_{category}_{cost_type}_installed_cost_{scenario}`
  (see `create_cost_col` in `utils/column_names.py`, L55-71).

### Sub-task A -- ENERGY STAR spec override column (make MP3 rebate-eligible + re-cost)
**Override values (confirmed):** ENERGY STAR requires **>= 15.2 SEER2 and
>= 8.1 HSPF2**, which in the SEER1/HSPF1 units the REMDB v4 cost regression uses is
approximately **>= 16.0 SEER1 and >= 9.5 HSPF1**. MP3's modeled spec
(15 SEER1 / 9 HSPF1) sits just below both thresholds, so the override is a modest
efficiency bump.

The session should:
1. Create an ENERGY-STAR-spec efficiency override for MP3's **upgrade** efficiency
   (>= 16.0 SEER1 / >= 9.5 HSPF1) in `process_euss_data.py` alongside the upgrade
   efficiency columns:
   - baseline: `base_heating_efficiency` (L259), `base_cooling_efficiency` (L269)
   - upgrade: `hvac_heating_efficiency` / `upgrade_hvac_heating_efficiency`
     (L389, L393); `hvac_cooling_efficiency` / `upgrade_hvac_cooling_efficiency`
     (L398, L400)
2. Route that override into the v4 **pm2 (efficiency) metric** in
   `remdb_v4_installed_cost_utils.py` (`add_remdb_metrics` -> pm2 conversion at
   L303-353) so MP3's capital cost reflects ENERGY-STAR-level equipment. The
   existing `_apply_efficiency_floor` (L358-402) is the precedent for
   clamping/overriding pm2 before costing.
3. Add MP3 to `REBATE_ELIGIBLE_HEATING_MPS` in `constants.py` (currently
   `[4, 8, 9, 10]`) so it passes the rebate MP gate under the override.

This is **value-moving** -- golden values will change; keep old rows with a
"supersedes" note. Design the counterfactual explicitly (MP3-at-ENERGY-STAR vs
true MP3) and keep it separate from Sub-task C.

### Sub-task B -- Where cooling capital cost is saved (answered)
Only **cooling REPLACEMENT** is computed, not cooling upgrade -- by design, because
the heat pump (the *heating* upgrade) provides cooling, so there is no separate
cooling upgrade cost. In the loop, cooling is done via
`add_remdb_metrics(end_use='cooling', metric_type='replacement')` +
`calculate_replacement_installed_cost(end_use='cooling')` and merged through the
`('cooling', 'replacement')` tuple.

Saved column: **`mp{mp}_cooling_replacement_installed_cost_{scenario}`** (e.g.
`mp4_cooling_replacement_installed_cost_v4MID`). It lives on the
`CAPITAL_COSTS_MPX['heating']['upgrade'][scenario]` DataFrame (cooling has no
top-level key in `CAPITAL_COSTS_MPX`) and is merged onto `df_euss_am_mpX_home`.
This cooling replacement cost is the avoided-AC-replacement credit consumed by the
`heatingLCC_coolingLCC` / cooling-LCC NPV scopes in
`calculate_lifetime_private_impact.py`. The session should confirm this is
intended (no cooling upgrade cost) and document it.

### Sub-task C -- Remove deprecated REMDB v3 code
v3 is no longer in `REMDB_COST_SCENARIO_KEYS`, so these remnants in the export are
dead and should be stripped:
- The pre-loop v3 stores:
  `CAPITAL_COSTS_MPX['heating']['replacement']['v3'] = df_euss_am_mpX_home.copy()`
  and the `['upgrade']['v3']` line.
- The `if scenario_key == 'v3': method, percentile = 'v3', None` branch and the
  `if method == 'v3': continue` skip inside the loop (unreachable now).
- The `if scenario_key == 'v3': continue` guard in the merge block.
- v3-referencing comments ("Store v3 results...", "DELETED V3...", "REMDB v3:
  Excel-based...").
- The enclosure-cost path is flagged in-code as "still uses v3 calculations path"
  (`calculate_enclosure_upgrade_costs.py`) -- decide whether it stays or is
  migrated.
- Before deleting, confirm no remaining consumer reads a `_v3` cost column.

### Relevant files
| File | Role |
|---|---|
| `cmu_tare_model/model_scenarios/tare_scenarios_v2_3_EXPORT_12July2026.py` | Capital-cost section to refactor; v3 cleanup; **backport to `.ipynb` by hand** |
| `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | Efficiency columns; home of the new ENERGY STAR override |
| `cmu_tare_model/utils/remdb_v4_installed_cost_utils.py` | `add_remdb_metrics`, pm2 efficiency conversion, `_apply_efficiency_floor` precedent |
| `cmu_tare_model/private_impact/calculations/calculate_equipment_installation_costs.py` | Upgrade-cost regression formula |
| `cmu_tare_model/private_impact/calculations/calculate_equipment_replacement_costs.py` | Replacement-cost regression formula (heating and cooling) |
| `cmu_tare_model/private_impact/calculations/calculate_enclosure_upgrade_costs.py` | Still on a v3 path -- evaluate during cleanup |
| `cmu_tare_model/constants.py` | `REMDB_COST_SCENARIO_KEYS` (re-enable low/high); `REBATE_ELIGIBLE_HEATING_MPS` (MP3 gate) |
| `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` | Consumes cooling replacement cost in the LCC net-capital credit |
| `cmu_tare_model/utils/validate_capital_costs.py` | Capital-cost validation oracle |
| `cmu_tare_model/utils/column_names.py` | `create_cost_col` -- cost column naming |

---

## Issue 1 (downstream) -- Dot-plot marker overlap (especially MP3)

### Symptom
In the economic adoption dot plot, markers/labels overlap noticeably, worst for
MP3. Likely to shift once the capital-cost sensitivity above changes adoption
rates, so tackle it after that work rather than as a standalone plotting tweak.

### Audit note
The label-placement logic in `plot_adoption_panel` cleanly splits **2-marker**
clusters left/right, but a **3+ marker** cluster that is not fully inside
x in [10, 90] falls through to default positions (no vertical stagger) -> overlap.
See `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py`
around L493-532. The new 3-marker rebate-policy-scenario mode (added 2026-07-12)
makes this fall-through more visible. Fix = better placement for 3+ markers per row
(vertical stagger or an offset ladder).

### Relevant files
| File | Role |
|---|---|
| `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py` | Dot plot; `build_econ_plot_df`, `plot_adoption_panel`, cluster/label placement |
| `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` | Net capital / NPV -- where the capital-cost sensitivity changes adoption inputs |
| `tare_model_main_v2_3.ipynb` (econ dotplot cell) | Figure driver -- **backport by hand, never edit the .ipynb** |

---

## Notes carried from the 2026-07-12 session (context only)

- The dot plot gained a `shape_by` option: `'replacement_credit_scenario'`
  (default, two markers) and `'rebate_policy_scenario'` (three markers: unsub /
  2024 / June 2026) for one fixed replacement-credit scope. Module symbols:
  `REBATE_POLICY_SCENARIO_MARKERS`, `build_rebate_policy_scenario_legend_handles`.
- Terminology: "regime" was renamed to **rebate_policy_scenario** everywhere; the
  replacement-cost-credit axis is named **replacement_credit_scenario**. The
  constant `REBATE_GUIDANCE_REGIMES` was renamed to `REBATE_POLICY_SCENARIOS`.
- June 2026 rebate fossil gate verified PASS on the real MP4 frame (every
  non-electric fuel $0).
