# Session Changelog -- 2026-07-13 (Rebate-Gate Adjudication, Tepper Alignment, Reproducibility)

## Session summary

Fixed the issues found auditing the 12 July 2026 run. Six gated tasks: audit the
June 2026 rebate fuel gate against its spec, adjudicate it, consolidate its
self-confirming test, fix the Tepper five-county home_count WARN, restore fresh-run
reproducibility of the main notebook, and clean up stale strings / dead cells.

All edits are to `.py` exports only. Per CLAUDE.md the `.ipynb` files are never edited
directly; the researcher backports by hand. Full-run numeric verification is deferred
to the researcher's 331k-home environment.

## Task 1 -- Audit (no edits)

- **The June 2026 fuel gate has no primary spec in the repo.** The
  "rebates may not fund fossil-system removal" rule exists only as the researcher's
  own restatement in `constants.py`, `calculate_rebate_june2026`,
  `summarize_rebate_funding`, CLAUDE.md, and the 11/12 July changelogs. No DOE notice,
  statute excerpt, or citation is stored anywhere. The `[PASS]` verification asserted
  the implementation ("fossil = $0"), so it could not detect an inversion.
- **$34 fingerprint mechanism confirmed:** June 2026 HEEHR reuses the 2024 HEEHR
  formula on the electric-baseline subset, so its total equals the 2024
  electric-baseline subtotal to within per-home rounding -- i.e. June 2026 eligibility
  IS the electric-baseline home set.
- **Five-county drift is a home-set difference, not a bucketing artifact.** The
  scenario frame's `county` is a verbatim copy of `in.county`
  (`process_euss_data.py`), so both paths key on the same codes. The adoption and
  demand tables come from two independent EUSS loads (scenario frame vs a fresh
  `load_euss_baseline()` inner-joined to the upgrade frame), so edge-case FIPS can
  drift by one home.
- **Both paths read the same baseline file at full precision;** no `.round(2)` on
  weight survives in source. The WARN fired for all 3,098 counties on sub-home
  precision noise.
- **Confirmed fresh-'N'-run NameErrors:** bare `df` in the funding cell, the inventory
  cell's leaked `df_euss_am_*` names, undefined `FIGURE_DPI`, and the `importlib.reload`
  scaffolding.

## Task 2 -- Adjudication (researcher decision)

**Decision: KEEP the electric-resistance-only fuel gate.** Per the June 2026 DOE
guidance the researcher holds, a rebate may not fund removing a fossil heating system,
so only existing electric-resistance baselines qualify under both HEEHR and HOMES; any
fossil baseline gets $0. This is intended behavior, not an inversion of the 2024
HEEHR/HOMES statute. The ~10-14 pp gap below 2024 adoption is the expected consequence
(fossil homes lose the rebate). Recorded in CLAUDE.md.

## Task 3 -- Spec-driven test (test-only, zero value move)

`model_scenarios/tare_scenarios_v2_3_EXPORT_12July2026.py`:
- Consolidated the two near-duplicate verification cells (they differed only in the
  adopter NPV scope, which does not affect the fuel gate) into ONE cell.
- Removed the false "NOT run automatically and NOT part of the .ipynb" docstring (both
  executed in the run).
- Replaced the print-only `[FAIL]` with hard `assert`s so the cell fails loudly, and
  added a positive check: fossil baselines must be $0 AND electric-resistance baselines
  must be funded (a bug that zeroed every rebate would otherwise pass the fossil-only
  check).

## Task 4 -- Tepper home_count WARN (diagnostic only, no exported value moves)

- `utils/export_tepper_csv.py`: removed the in-function exact-match WARN (fired for all
  3,098 counties) and renumbered the remaining steps.
- `tare_model_main_v2_3_EXPORT_12July2026.py`: moved the per-MP reconciliation into the
  notebook, next to the data, with a tolerance of one home read from
  `df_baseline['weight'].median()` (never hardcoded, so it adapts to future ResStock
  releases). Warns only when two counts differ by MORE than one home; the five
  one-home edge cases and all precision noise now fall within tolerance, so the WARN
  lists zero counties. Exported `home_count` (from the adoption table) is unchanged, so
  no golden value moves. The exact five buildings were not chased down (needs a data
  run); the tolerance makes that unnecessary for a clean WARN.

## Task 5 -- Fresh-run reproducibility (no output-value change)

`tare_model_main_v2_3_EXPORT_12July2026.py`:
- Rebuilt the funding-summary cell on `DATAFRAMES_BY_MP[mp]['fixed_base']`, looping
  `selected_mps` and deriving the adopter column via `create_adoption_col` /
  `define_scenario_params` (no bare `df`, no hardcoded MP or column string). MP4 output
  matches the Task 3 spec cell.
- Deleted the `importlib.reload` interactive-scaffolding cell.
- Guarded the column-inventory cell so it runs only when the scenario-run intermediate
  frames are in memory (`%run -i`); otherwise it prints `[SKIP]`.
- Defined `FIGURE_DPI = 600` with the other figure constants (matches the literal
  `dpi=600` used elsewhere).

## Task 6 -- Cleanup (no output-value change, except one display annotation)

`tare_model_main_v2_3_EXPORT_12July2026.py`:
- MP3 header now matches the `.ipynb`: appended `--> (16 SEER1, 9.5 HSPF1) for ENERGY
  STAR`.
- "SIX economic-adopter columns" comment -> NINE, listing the three rebate policy
  scenarios (`unsub`, `sub`, `sub_june2026`).
- Deduplicated the adoption choropleth cell (kept the `vmax=100` variant that was
  overwriting the `vmax=50` one, so the final PNG is unchanged).
- De-hardcoded both `fuel_counts_millions` blocks: `int(n) * 242` ->
  `groupby(...)['weight'].sum()`. This nudges the dotplot's fuel-count ANNOTATION from
  n x 242 to n x 242.131013 (~0.05%, display only, more accurate).
- Removed the four dead `.columns` inspection cells.

`model_scenarios/tare_scenarios_v2_3_EXPORT_12July2026.py`:
- Stale "less WTP and more WTP" -> "nine cases: three scopes x three rebate policy
  scenarios".
- Removed the mid-notebook `VERBOSE = True` hardcode (reverts to the imported default;
  prints only, no computed values change).

Left as-is: the three PLACEHOLDER cells (intentional WIP stubs) and the
`HEATING_MP_SUBTITLES` / `mp_labels` figure labels (already `.py`<->`.ipynb`
consistent; whether to surface the ENERGY STAR cost respec in figure labels is a
content call, given MP3's load profile is still the 15-SEER profile).

## CLAUDE.md updates

- June 2026 rules header: "MP4 in, MP3 out" -> "MP3 and MP4 in" (12 Jul ENERGY STAR
  override made MP3 eligible).
- Fuel gate bullet: added the "adjudicated KEEP (13 Jul 2026)" rationale.
- Limitation #5: MP3 now qualifies for both programs (was "qualifies for neither").
- Session Log: added the 13 July 2026 row.

## Deferred to the researcher's environment

- `.ipynb` hand-backport of the Task 3 consolidated test cell, the Task 4 notebook
  reconciliation block, and all Task 5/6 main-notebook edits.
- Full-run verification: fresh 'N'-path runs top-to-bottom with no NameError; the Task
  3 test asserts hold; the Tepper WARN lists zero counties; the `_sub_june2026` mean
  adoption golden rows get derived (still PENDING in CLAUDE.md).
- Out of scope this session (carried): re-enable v4LOW/v4HIGH, MP8-10 activation,
  negative-cooling-savings handling, FIPS 46102 Cambium crosswalk, data_loading dtype
  specs, and the exact five-county building identification.
