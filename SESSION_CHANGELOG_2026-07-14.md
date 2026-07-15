# Session Changelog -- 2026-07-14 (Rebate Consolidation + 2024 HOMES + Dotplot Relabel)

## Session summary

Consolidated the two parallel rebate functions into ONE central function plus a
per-vintage rule config (DRY), reproducing the existing numbers exactly, then
filled the gap where the 2024 vintage had no HOMES program, and relabeled /
deduplicated the adoption dotplots.

Autonomous mode (researcher pre-approved): the one-edit-per-stop-gate rule was
suspended; work ran through all seven tasks with a halt-on-failed-verification
checkpoint. Exactly ONE intended value move (Task 4, add 2024 HOMES). All edits
are to `.py` sources and the constants/module only. Per CLAUDE.md the `.ipynb`
files are never edited; the researcher backports by hand. Full-run numeric
verification (331k homes) and golden re-derivation are deferred.

See `SESSION_NOTES_rebate_consolidation_2026-07-14.md` for the per-task
diff-and-verification log.

## Task 1 -- Audit (no edits)

Mapped both rebate functions and every call site. Reconciled the two HEEHR cap
mechanisms (2024 `REBATE_MAPPING` tech-string cap vs June 2026 flat
`HEEHR_CAP_HEAT_PUMP`) -- equal ($8,000) for every modeled heat-pump retrofit.
Confirmed 2024 HOMES is genuinely absent. Resolved the central ambiguity: the
hard byte-identity gates require June 2026 to stay unchanged (so its HOMES stays
electric-gated), so only 2024 HOMES becomes fuel-neutral this session.

## Task 2 -- Guidance rule config (additions only)

Added `REBATE_RULE_CONFIG` to `constants.py` (per-vintage keys: column_guidance,
eligibility_col, heehr_fuel_gate, homes_enabled, homes_fuel_gate,
heehr_python_round). No existing value changed.

## Task 3 -- Central function + helpers (byte-identical)

Added `calculate_rebate_program(guidance)` + `_heehr_rebate_amount` /
`_homes_rebate_amount`; converted `calculate_rebateIRA` and
`calculate_rebate_june2026` to DEPRECATED thin wrappers. Preserved a per-vintage
rounding flag (`heehr_python_round`) because the original 2024 path used Python
`round()` and June 2026 used numpy `.round()` -- they differ a cent on exact
half-cent products. Verified on a 217-home grid: 0 mismatches for MP3 and MP4 on
the 2024 amount, June 2026 amount, and June 2026 eligibility.

## Task 4 -- Add 2024 HOMES (THE ONE VALUE MOVE)

Flipped `homes_enabled` True for 2024 (fuel-neutral; `homes_fuel_gate=False`).
Repointed `summarize_rebate_funding` to read the explicit
`mp{mp}_rebate_eligibility_ira2024`/`_june2026` label instead of inferring HEEHR
from a positive amount. Verified: 2024 HEEHR unchanged (fossil <=80% AMI still
funded), 2024 HOMES now credits fossil + electric homes above 150% AMI, June 2026
unchanged (fossil HOMES still $0).

## Task 5 -- Migrate callers

Migrated `model_scenarios/tare_scenarios_v2_3_EXPORT_12July2026.py` to loop
`for guidance in REBATE_POLICY_SCENARIOS:` on the central function. Old names kept
as deprecated wrappers (tests still use them). Added a value-neutral guard so
HOMES inputs are read only when a home routes to HOMES.

## Task 6 -- Relabel dotplots + consolidate viz constants

`visuals_adoption_dotplot.py`: `REBATE_POLICY_SCENARIO_LABELS` ->
`Unsubsidized` / `December 2024 Rebate Eligibility` / `June 2026 Rebate
Eligibility` (markers rekeyed); added `REPLACEMENT_CREDIT_CASES/MARKERS`,
`NATIONAL_FUEL_GROUPING_ORDER`, `build_replacement_credit_legend_handles()`; folded
`scaling_factor=242` to a weight-derived homes count. Notebook export: deleted the
floating `_ECON_*` / `_RPS_GROUPING_ORDER` locals and the inline legend, fixing the
"Heating Repl. Credit Only" vs "Heating Repl. Credit" mismatch. Both plot modes
build; every legend label maps to a marker key.

## Task 7 -- Spec, docs, memory, tests

`docs/rebate_guidance_reference.md` is in the repo as the rebate spec. Updated
CLAUDE.md (fuel-gate bullet superseding "adjudicated KEEP"; HOMES fuel-neutral,
HEEHR-only gate; "2024 = HEEHR only" retired; limitations #2/#5/#6; savings-frac
note; `_sub`/`_sub_june2026` golden rows PENDING; new Session Log row; header
date). Superseded the `project_june2026_rebate_gate_electric_only` memory + its
MEMORY.md entry and added `project_rebate_consolidation_central_function`. Rewrote
the two stale tests (`test_june2026_mp3_is_eligible`,
`test_funding_2024_heehr_and_homes_fuel_neutral`), added
`test_2024_homes_is_fuel_neutral` and `test_june2026_homes_still_electric_gated`
(13 rebate tests pass). Rewrote `scripts/verify_june2026_rebate_fossil_gate.py` to
a program x fuel crosstab for both vintages: HEEHR fossil = $0 (hard), HOMES fossil
MAY be > $0 (fuel-neutral), plus a 2024-HOMES fossil-funding check.

## Deferred (not started)

- 2026 HOMES fuel-neutral fix (moves `_sub_june2026`); HOMES low-income tier +
  routing fallback for LMI fossil <=150% AMI.
- Insulation-before-HVAC (heehr_2026) and dual-fuel retention pathway.
- Full 331k-home run: re-derive `_sub` (and eventually `_sub_june2026`) golden
  rows, refresh the movement cross-tab, hand-backport all edited cells (rebate +
  both dotplots) to the `.ipynb`.
