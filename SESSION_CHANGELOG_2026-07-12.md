# Session Changelog -- 2026-07-12 (Rename for clarity, June 2026 rebate check, dotplot shape option)

## Session Summary

Three pieces of work, in order:

1. **Rename for clarity.** Renamed the ambiguous term "regime"/"rebate_regime" to
   **rebate_policy_scenario** (which rebate policy applies: unsubsidized / 2024 /
   June 2026) and introduced **replacement_credit_scenario** for the previously
   unnamed axis (which replacement cost the NPV credits:
   `heatingLCC_coolingSavings` = heating replacement only vs
   `heatingLCC_coolingLCC` = heating + cooling replacement). These are code-shape
   changes only -- comments, docstrings, and one constant. No DataFrame column
   string changed.

2. **June 2026 rebate correctness check** (verification only, no logic change).

3. **Dotplot shape option** -- the economic adoption dotplot can now encode the
   rebate policy scenario as the marker shape (three markers) for one fixed
   replacement_credit_scenario, in addition to the existing two-shape
   replacement-credit view.

## Renames (no column strings touched)

- Constant `REBATE_GUIDANCE_REGIMES` -> `REBATE_POLICY_SCENARIOS` in
  `constants.py`. The list VALUES ("ira2024"/"june2026") are unchanged; grep
  confirmed the constant is not imported in any `.py`.
- "regime"/"rebate-regime" -> "rebate policy scenario" in comments/docstrings
  across `column_names.py`, `calculate_lifetime_private_impact.py`,
  `determine_economic_adoption_potential.py`, `export_tepper_csv.py`,
  `determine_rebate_eligibility_and_amount.py`, `test_rebate_june2026.py`,
  `tare_scenarios_v2_3_EXPORT_28June2026.py`, `tepper_export_data_dictionary.md`,
  and `CLAUDE.md`.
- `hvac_replacement_scenario`: NOT renamed. It lives only in the deprecated
  sensitivity module and the pre-Session-A export (retired
  `heating`/`heating_and_cooling` category tokens), not the live axis. Added a
  one-line deprecation note pointing to the live `replacement_credit_scenario`
  (expressed through the NPV case tokens). Its string values were left untouched
  because changing them would move (retired) column names.
- "credit scenario" as a term did not exist in the codebase; nothing to rename.

Verification: full suite 287 passed / 11 pre-existing failures (unchanged
baseline; unrelated to this rename) + 1 pre-existing collection error
(`test_kpi_functions.py`). Rebate tests: 11 passed. No column name changed.

## June 2026 rebate check (fossil gate)

- Confirmed `calculate_rebate_june2026` fuel gate limits June 2026 rebates to
  `base_heating_fuel == 'Electricity'`, applied to BOTH HEEHR and HOMES. 2024
  guidance still allows fossil HEEHR by design (not a bug).
- `test_rebate_june2026.py`: 11 passed, including the fossil-gate, 2024-fossil-
  allowed, and adopters-only tests.
- On the real frame (MP4, `guidance='june2026'`), `summarize_rebate_funding`
  by-fuel table showed $0 for every non-electric fuel (Fuel Oil, Natural Gas,
  Other Fuel, Propane) in both total_eligible and adopters_only. PASS.
- Weighted (MP4): HEEHR total_eligible ~$89.9B / adopters_only ~$41.5B; HOMES
  total_eligible ~$17.2B / adopters_only ~$8.2B. total_eligible is uncapped
  potential (no funding cap modeled); adopters_only is the figure to compare
  against the ~$8-9B appropriation.

## Dotplot shape option

`visuals_adoption_dotplot.py`:
- `build_econ_plot_df` gained `shape_by` (default
  `'replacement_credit_scenario'`, unchanged behavior) and
  `fixed_replacement_credit_scenario` (default `'heatingLCC_coolingSavings'`).
  New mode `'rebate_policy_scenario'` returns three rows per grouping (unsub /
  2024 / June 2026) for the fixed scope, each plotting that scenario's own
  adoption rate. All column names built via `create_adoption_col`.
- Added module constants `REBATE_POLICY_SCENARIO_ORDER`,
  `REBATE_POLICY_SCENARIO_LABELS`, `REBATE_POLICY_SCENARIO_MARKERS`
  (circle / square / triangle) and helper
  `build_rebate_policy_scenario_legend_handles()`.
- Module docstring updated to describe both modes.

Verification: default mode returns 2 rows/grouping with unchanged labels and
columns; new mode returns 3 rows/grouping labeled
['Unsubsidized', '2024 HEEHR', 'June 2026 Guidance']; June 2026 marker value
equals the raw adopter mean; both invalid-input paths raise ValueError. Synthetic
render confirmed three markers/row with shapes matching the legend.

Known limitation (deferred): the label-placement logic handles 2-marker clusters
cleanly but falls through for a 3-marker cluster that is not fully inside
x in [10, 90]; such rows may show overlapping labels. Flag on the real figure.

## Backport (researcher's environment; .ipynb not edited here)

- No public function or parameter was renamed, so the Phase 1 rename has no
  notebook call sites to backport.
- The three-shape dotplot figure is produced by a new notebook cell (provided in
  chat) placed just after the existing econ dotplot cell. The existing two-shape
  cell is unchanged.

## Corrected column naming (for refreshing the CLAUDE.md mirror)

Live economic-adopter columns, confirmed from a real column:
`ref2025_mp{mp}_{scope}_{policy}_econ_adopter_{discount}` where
`scope` in {heatingSavings_coolingLCC, heatingLCC_coolingSavings,
heatingLCC_coolingLCC}, `policy` in {unsub, sub, sub_june2026}, e.g.
`ref2025_mp4_heatingLCC_coolingSavings_sub_june2026_econ_adopter_fixed_base`.
The chat-project CLAUDE.md mirror's older tokens
(`heating_only` / `heating_and_cooling_savings` / `heating_and_cooling_full`)
are stale and should be replaced with the above.

## Golden values

No golden value changed. The renames and the plot option leave all outputs
identical.
