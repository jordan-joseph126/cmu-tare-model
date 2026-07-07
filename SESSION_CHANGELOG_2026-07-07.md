# Session Changelog -- 2026-07-07 (Session B, autonomous)

## Session Summary

Capital-cost refactor and main-notebook restructure for the TARE model, run
autonomously. Delivered a functional-equivalence oracle, a value-identical
compliance refactor of the REMDB v4 cost utilities, an isolated and quantified
removal of capacity-bound clamping, CLAUDE.md alignment to the six-case NPV
scheme, and a latent-bug fix in the main notebook export. Model-level token
migration and notebook cell transport that cannot be run or verified offline
were documented and flagged for hand-implementation rather than guessed. Full
deliverable: `REFACTORING_GUIDE_07July2026.md`.

Branch: `claude/tare-capital-cost-refactor-754hua` (same base commit as
`update-data-and-projections-aeo2026-cambium2024`; see the guide's branch note).

## Detailed Changelog (newest first)

### Task 6 -- Final verification + deliverables
- Full suite reproduces the Task 1 bar: `33 failed, 236 passed, 24 errors`
  (no new failures). Efficiency-floor test 5/5.
- Added `REFACTORING_GUIDE_07July2026.md` and this changelog.

### Task 5 -- `tare_model_main_v2_3_EXPORT_5July2026.py`
- Added an `ANALYSIS RUN CONTROLS` cell defining `GRID_IMPACT_ANALYSIS = False`;
  the flag was referenced in two grid-impact gates but never defined (latent
  `NameError`). Definition precedes both uses.
- Deferred (documented, needs a live run / research judgment): consolidating the
  two MP8 climate-SCC histogram cells, migrating their `moreWTP` `create_npv_col`
  strings to `create_npv_case_col`, and transporting the adoption/dotplot cells
  from the 5 July bill_savings EXPORT.
- Backport to `tare_model_main_v2_3.ipynb` required (recorded in the guide).

### Task 4 -- Remove capacity-bound clamping (value-moving, quantified)
- Removed `_apply_capacity_clamping`, `_log_capacity_clamp`, Step 4.5b, and the
  `CAPACITY_BOUND_CLAMPING_TOLERANCE` import from
  `utils/remdb_v4_installed_cost_utils.py`; removed the constant from
  `constants.py` (replaced with a rationale note); updated the capture harness
  import.
- Impact (from the frozen baseline): replacement metrics only, 3 homes each,
  upgrade unaffected; e.g. a 5.5-ton home's mid heating-replacement cost moves
  $25,069.90 -> $27,417.46. pm2 and `pm2_..._original` byte-identical.
- `validate_capital_costs.py` left unchanged: its `_build_clamping_summary`
  reports the preserved efficiency floor, not capacity clamping.

### Task 3 -- fillna(0) audit
- `private_impact/calculate_lifetime_private_impact.py`: added a WHY comment on
  the pre-mask `.fillna(0)` documenting that valid homes should never carry NaN
  by construction, that this could not be confirmed offline, and a TODO to
  convert to a fail-loud check after a zero-count run. Behavior unchanged.

### Task 2 -- Value-identical REMDB v4 cost-utils refactor
- `utils/remdb_v4_installed_cost_utils.py`: rewrote the stale module header;
  deleted the two dead "DELETE FOR NOW - NON-HVAC" commented blocks; ASCII-fixed
  conversion docstrings/comments and the percentile-filter print; vectorized
  `_apply_efficiency_floor` (row_id->floor map + single `Series.clip`) with a
  `Dict[str, float]` type hint.
- ASCII-only docstring/comment fixes in
  `calculate_lifetime_private_impact.py` and
  `determine_rebate_eligibility_and_amount.py`.
- Verified byte-identical to `baseline_capture/` (df_main, df_detailed, v4
  regression costs at all percentiles); `test_efficiency_floor_refactoring.py`
  5/5 unmodified. Clamping untouched (reserved for Task 4).

### Task 1 -- Close 6 July loose ends
- `CLAUDE.md`: naming section updated to six NPV/adopter cases via
  `create_npv_case_col`; sensitivity table gained a subsidy split; golden-values
  table gained a `PENDING` six-case adoption row (old rows preserved); session
  log updated.
- Propagation verified PASS for `compute_adoption_rate`,
  `visuals_adoption_potential`, `visuals_adoption_dotplot`.
- Old-token sweep documented: `moreWTP`/`lessWTP`/`iraRef`/`preIRA` and
  half-migrated `create_npv_case_col(..., wtp=..., cost_scenario=...)` calls
  (which raise `TypeError`) remain in the orchestration/export layer; flagged
  for hand-migration because the old->new case mapping is non-mechanical and the
  files cannot be run offline.
- Recorded the pre-existing test bar as the "no worse than" line.

### Task 0 -- Baseline capture harness
- Added `scripts/capture_capital_cost_baseline.py` and the `baseline_capture/`
  snapshot: a deterministic path-covering synthetic home set run through the
  real REMDB v4 table, capturing `add_remdb_metrics` outputs and the v4
  regression cost. This is the functional-equivalence oracle for Tasks 2-4.

## Notes

- No `.ipynb` files were edited. One backport is pending (the
  `GRID_IMPACT_ANALYSIS` run-control cell) -- see the guide.
- The REMDB v4 cost CSV was placed under the git-ignored
  `cmu_tare_model/data/retrofit_costs/` for local verification only; it is not
  committed.
- The real EUSS stock is offline (Zenodo), so verification used the synthetic
  oracle; model-level changes needing the full stock were deferred with
  documentation.
