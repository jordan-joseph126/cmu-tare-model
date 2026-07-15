# Session Notes -- Rebate Function Consolidation (2024 + June 2026)

Date: 2026-07-14
Mode: AUTONOMOUS (researcher pre-approved; one-edit-per-stop-gate suspended).
No commits, no .ipynb edits. Everything staged for morning review.

The ONE intended value move this session: Task 4 (add 2024 HOMES). Everything
else is structure-only and must stay byte-identical.

---

## Task 1 -- Audit (no edits) -- COMPLETE

### Files read
- `private_impact/data_processing/determine_rebate_eligibility_and_amount.py`
  (both `calculate_rebateIRA` 2024 + `calculate_rebate_june2026`, plus
  `summarize_rebate_funding`, `summarize_june2026_rebate_totals`).
- `constants.py` (rebate constants block, lines ~180-253).
- `utils/column_names.py` (`create_rebate_col`, `create_npv_case_col`,
  `NPV_CASE_CATEGORIES`).
- `private_impact/calculate_lifetime_private_impact.py` (rebate consumption,
  lines ~274-324) -- confirms `_sub` reads the guidance-LESS 2024 column and
  `_sub_june2026` reads the `june2026` column.
- `model_scenarios/tare_scenarios_v2_3_EXPORT_12July2026.py` (call sites, lines
  ~604-649).
- `docs/rebate_guidance_reference.md` (four-scenario spec).
- `tests/private_impact/test_rebate_june2026.py`.

### Call sites (all guidance passes through `create_rebate_col`)
1. Scenarios export `.py` line 640: `calculate_rebateIRA(...)` per end_use x
   cost_scenario.
2. Scenarios export `.py` line 646: `calculate_rebate_june2026(...)` same loop.
3. `calculate_lifetime_private_impact.py` line 275 reads guidance-less 2024
   rebate col; line 293 reads `june2026` col.
4. `summarize_rebate_funding` / `summarize_june2026_rebate_totals` read the
   amount + eligibility cols.
5. Mirror `.ipynb` cells exist (NOT edited this session).

### Shared vs vintage-specific logic
Shared (all vintages): cooling no-op; category-in-REBATE_MAPPING check;
`initialize_validation_tracking` + `create_retrofit_only_series` +
`apply_final_masking`; MP efficiency gate (`REBATE_ELIGIBLE_HEATING_MPS`, now
MP3+MP4); state-participation gate (`NON_PARTICIPATING_REBATE_STATES = {'SD'}`);
HEEHR income coverage 100% (<=80% AMI) / 50% (80-150%); $8,000 heat-pump cap.

Vintage-specific:
- 2024 (`calculate_rebateIRA`): row-wise `.apply`; cap via
  `get_max_rebate_amount` -> `REBATE_MAPPING['heating']` (tech-string check on
  `upgrade_hvac_heating_efficiency` containing 'ASHP'/'MSHP', amount 8000);
  MP9/10 weatherization branch; NO fuel gate; NO HOMES; NO eligibility label;
  writes guidance-less column `mp{mp}_heating_rebate_amount_{cost}`.
- June 2026 (`calculate_rebate_june2026`): vectorized; flat `HEEHR_CAP_HEAT_PUMP`
  (8000); HEEHR electric-resistance fuel gate; HOMES savings-tier branch
  (currently ALSO electric-gated); writes `..._june2026_{cost}` +
  `mp{mp}_rebate_eligibility_june2026` label.

### HEEHR cap reconciliation (the two mechanisms)
`REBATE_MAPPING['heating'][2] == 8000 == HEEHR_CAP_HEAT_PUMP`. The 2024 mechanism
additionally zeroes the cap when the upgrade string is not a heat pump
('ASHP'/'MSHP'). For every modeled MP3/MP4 heating retrofit the upgrade IS a heat
pump, so both mechanisms yield 8000. The merged HEEHR helper applies the
tech-string gate vectorized (cap = 8000 where upgrade contains ASHP/MSHP else 0);
this reproduces 2024 exactly AND leaves 2026 unchanged for all modeled homes.
Rounding order differs (2024: `min(round(cov*cost,2), 8000)`; 2026:
`round(min(8000, cov*cost),2)`) but is equal to the penny because the cap 8000 is
exact. Merged helper uses `round(min(cap, cov*cost), 2)`.

### 2024 HOMES genuinely absent -- CONFIRMED
`calculate_rebateIRA` has no HOMES branch; >150% AMI ('Middle-to-Upper-Income')
homes are set to 0.0 (line 437). This is the gap Task 4 fills.

### HOMES fuel-neutrality -- the central ambiguity, RESOLVED
Reference doc + adjudication: HOMES is fuel-neutral in BOTH vintages (the
fossil-removal restriction is HEEHR-only). BUT current `calculate_rebate_june2026`
applies `electric_mask` to HOMES too (2026 HOMES is electric-gated today).

Two instructions conflict:
- (A) Tasks 3 & 5 HARD HALT GATES: "2026 HEEHR+HOMES MUST stay byte-identical";
  homes_2026 is explicitly one of "the three unchanged scenarios".
- (B) Reference doc / Task 7: "HOMES fuel-neutral in both vintages".

Resolution (honoring the hard halt-gates, which override a doc directive):
- 2024 HOMES -> fuel-neutral (the Task 4 value move; affects `_sub` only).
- 2026 HOMES -> UNCHANGED (electric-gated) to preserve byte-identity; the
  fuel-neutral fix for 2026 is DEFERRED (it would move the `_sub_june2026`
  golden, which the deferred-work list explicitly defers to the full-run
  session). Supported by: CLAUDE.md already marks `_sub_june2026` PENDING and the
  deferred list defers the `_sub_june2026` re-derivation; this session moves ONLY
  `_sub`. Documented as a limitation in Task 7 rather than falsely claiming 2026
  is already fuel-neutral.

### Proposed central-function design
`calculate_rebate_program(df, category, menu_mp, cost_scenario, guidance,
verbose)` dispatched by a per-vintage rule config in `constants.py`
(`REBATE_RULE_CONFIG`, keyed by `REBATE_GUIDANCE_IRA2024` / `_JUNE2026`):
  - `column_guidance`: None for 2024 (byte-identical guidance-less names),
    'june2026' for 2026.
  - `eligibility_col`: `mp{mp}_rebate_eligibility_ira2024` / `_june2026`.
  - `heehr_fuel_gate`: 2024 False, 2026 True.
  - `homes_enabled`: 2024 False (Task 2/3) -> True (Task 4 value move);
    2026 True.
  - `homes_fuel_gate`: 2024 False (fuel-neutral); 2026 True (byte-identity;
    deferred fix).
Shared helpers: `_heehr_amount(...)` (income coverage + tech-gated cap),
`_homes_amount(...)` (savings tiers + 50% coverage), plus the existing
validation-framework calls. Old names `calculate_rebateIRA` /
`calculate_rebate_june2026` become thin wrappers -> central fn.

Weatherization (MP9/10) is out of scope and never runs (VALID_MENU_MPS=[0,3,4]);
the byte-identity sample is MP3/MP4, so the merged heat-pump path need not carry
the weatherization branch. The 2024 wrapper preserves it by delegating MP9/10 to
the legacy `calculate_rebate` helper (kept) -- documented, low-risk, untested
(no MP9/10 run).

### Change classification
- STRUCTURE-ONLY (byte-identical): central fn + helpers; 2024 HEEHR; 2026
  HEEHR+HOMES; caller migration; dotplot relabel/consolidation.
- VALUE-MOVING (Task 4 only): enable 2024 HOMES (fuel-neutral); add 2024
  eligibility label; repoint `summarize_rebate_funding` guidance=None branch to
  read the label.

No ambiguity remaining that risks silent value corruption. CONTINUE.

---

## Task 2 -- Guidance rule config -- COMPLETE

Added `REBATE_RULE_CONFIG` to `constants.py` (additions only; no existing value
changed). Keys: column_guidance, eligibility_col, heehr_fuel_gate, homes_enabled,
homes_fuel_gate, heehr_python_round. 2024 starts homes_enabled=False (flipped in
Task 4). Verify: config imports; existing rebate tests unaffected by the addition
(10 pass, 1 pre-existing failure -- see Task 3). CONTINUE.

## Task 3 -- Central function + shared helpers (byte-identical) -- COMPLETE

Added `calculate_rebate_program(df, category, menu_mp, cost_scenario, guidance,
verbose)` plus helpers `_heehr_rebate_amount` and `_homes_rebate_amount`.
Converted `calculate_rebateIRA` and `calculate_rebate_june2026` to thin wrappers
that delegate to it. Kept legacy `get_max_rebate_amount` / `calculate_rebate`
(now unused; harmless).

ROUNDING QUIRK FOUND + PRESERVED (important): the original 2024 path rounded with
Python builtin round(); the original June 2026 path used numpy array .round().
They disagree by ONE CENT on exact half-cent products (a two-decimal cost with an
odd final cent, halved by the 50% moderate-income coverage). This is real and
frequent in the moderate-income HEEHR homes, so each vintage keeps its own
rounding via `heehr_python_round` in the config. Without this the `_sub` 2024
column would have moved a cent on many homes.

REGRESSION FIXED: the consolidated function reads `base_heating_fuel` only when a
fuel gate is active (the 2024 fuel-neutral path no longer requires it) and keys
HEEHR routing off `percent_AMI` (the pipeline always sets it in
calculate_percent_AMI, consistent with income_level). Updated
`test_2024_south_dakota_excluded` to supply `percent_AMI` (new contract).

VERIFICATION:
- `scratchpad/verify_task3.py` compared the central fn against the ORIGINAL
  bodies (before wrapper conversion) on a 217-home grid (fuel x AMI x savings x
  state x validity, non-round costs incl. exact half-cents): OVERALL PASS -- 0
  mismatches for MP3 and MP4 on the 2024 amount, June 2026 amount, and June 2026
  eligibility. homes_2026 stays electric-gated (byte-identical).
- Unit suite `test_rebate_june2026.py`: 10 pass, 1 fail
  (`test_june2026_mp3_never_eligible`) -- PRE-EXISTING and stale (MP3 became
  rebate-eligible in the 12-Jul ENERGY STAR override; the fixture also lacks MP3
  cost columns). To be rewritten in Task 7. Not a regression from this session.

No value moved. CONTINUE.

---

## Task 4 -- Add 2024 HOMES (THE ONE VALUE MOVE) -- COMPLETE

Flipped `REBATE_RULE_CONFIG[ira2024]['homes_enabled']` False -> True (one-line
value move; homes_fuel_gate stays False = fuel-neutral). Repointed
`summarize_rebate_funding` to read the explicit eligibility label for BOTH
vintages (via REBATE_RULE_CONFIG) instead of inferring HEEHR from a positive
amount -- the old inference would mislabel the new 2024 HOMES dollars as HEEHR.
The 2024 amount column stays guidance-less; its label uses the 'ira2024' token.

VERIFICATION (`scratchpad/verify_task4.py`):
- 2024 HEEHR unchanged: electric <=80 -> $8,000; electric 120% -> $5,000; FOSSIL
  <=80 -> $8,000 (2024 allows fuel switching, no HEEHR fuel gate).
- 2024 HOMES now credited to >150% AMI homes, FUEL-NEUTRAL: electric 200%/40% ->
  $4,000; Natural Gas 200%/40% -> $4,000; Fuel Oil 200%/25% -> $2,000; <20%
  savings -> $0. by_fuel shows Natural Gas HOMES = $12,000 and Fuel Oil = $2,000
  (fossil HOMES now funded under 2024).
- June 2026 UNCHANGED: HEEHR still electric-gated (fossil <=80 -> $0); HOMES still
  electric-gated (fossil 200% -> $0; electric 200% -> $4,000). '_sub_june2026'
  does not move this session.

Consequence: the '_sub' NPV/adopter (2024) adoption rows will rise on the full
run (>150% AMI homes gain HOMES). Golden rows marked PENDING in Task 7.

Known follow-on: `test_funding_2024_is_heehr_only_and_allows_fossil` is now stale
(2024 is no longer HEEHR-only; summarize needs the ira2024 label). Rewritten in
Task 7. CONTINUE.

---

## Task 5 -- Migrate callers -- COMPLETE

Migrated the active export
`model_scenarios/tare_scenarios_v2_3_EXPORT_12July2026.py`:
- import now pulls `calculate_rebate_program` (+ `REBATE_POLICY_SCENARIOS`)
  instead of the two old names.
- the rebate cell loops `for guidance in REBATE_POLICY_SCENARIOS:` calling
  `calculate_rebate_program(...)` (same functions, same order 2024 -> June 2026,
  so byte-identical for the three unchanged scenarios by construction).
- fixed the stale "calculate_rebateIRA() can find them" comment.
Old names kept as DEPRECATED thin wrappers (docstrings updated) because the test
suite and any un-migrated caller still use them. The superseded 28-June export was
left as-is (historical). The `.ipynb` is NOT edited (researcher backports).

VERIFICATION:
- Export compiles (`py_compile`).
- No `calculate_rebate_june2026`/`calculate_rebateIRA` calls remain in the active
  export; only `calculate_rebate_program` is called.
- Rebate tests: 9 pass, 2 fail -- both intended stale tests for Task 7.
  `test_2024_south_dakota_excluded` now passes.
- Added a value-neutral robustness guard: HOMES inputs (modeled savings, cooling
  cost) are read only when at least one home routes to HOMES (verify_task4 still
  PASS).

CONTINUE.

---

## Task 6 -- Relabel dotplots + consolidate viz constants -- COMPLETE

Module `adoption_potential/data_processing/visuals_adoption_dotplot.py`:
- `REBATE_POLICY_SCENARIO_LABELS`: unsub -> 'Unsubsidized', sub -> 'December 2024
  Rebate Eligibility', sub_june2026 -> 'June 2026 Rebate Eligibility'.
- `REBATE_POLICY_SCENARIO_MARKERS` keys updated to the new labels (o / s / ^).
- Module docstring "2024 HEEHR / June 2026 Guidance" -> vintage labels, with a
  note that the labels are vintage-based so they stay correct now both vintages
  model HEEHR + HOMES.
- Added `REPLACEMENT_CREDIT_CASES`, `REPLACEMENT_CREDIT_MARKERS`,
  `NATIONAL_FUEL_GROUPING_ORDER`, and `build_replacement_credit_legend_handles()`
  (legend labels == marker keys exactly).
- Folded the hardcoded `scaling_factor = 242.0`: `weighted_homes_millions` is now
  derived from the actual `weight` column sum per group (falls back to
  n*scaling_factor only when no weight column), consistent with the y-axis
  fuel_counts_millions.

Notebook export `tare_model_main_v2_3_EXPORT_12July2026.py`:
- Plot 1 (replacement-credit): deleted floating `_ECON_CASE_MARKERS`,
  `_ECON_CASES`, `_ECON_GROUPING_ORDER` and the inline legend; imports and uses
  `REPLACEMENT_CREDIT_CASES` / `REPLACEMENT_CREDIT_MARKERS` /
  `NATIONAL_FUEL_GROUPING_ORDER` / `build_replacement_credit_legend_handles()`.
  This fixes the legend/marker mismatch ("Heating Repl. Credit Only" ->
  "Heating Repl. Credit").
- Plot 2 (rebate-policy): deleted floating `_RPS_GROUPING_ORDER`; uses the shared
  `NATIONAL_FUEL_GROUPING_ORDER`.

VERIFICATION:
- Module imports; every REBATE_POLICY / REPLACEMENT_CREDIT marker key equals its
  label, and both legend helpers emit labels that match their marker keys.
- Export: no `_ECON_*` / `_RPS_GROUPING_ORDER` / "Heating Repl. Credit Only"
  strings remain; the new module names are referenced.
- End-to-end smoke test (Agg backend): both `build_econ_plot_df` modes build and
  `plot_adoption_panel` + both legend helpers render; weighted_homes is
  weight-derived (not n*242). The `.ipynb` mirror cells are NOT edited
  (researcher backports).

CONTINUE.

---

## Task 7 -- Spec, docs, memory, tests -- COMPLETE

- `docs/rebate_guidance_reference.md` already in the repo as the rebate spec.
- CLAUDE.md: rewrote the fuel-gate bullet (HOMES fuel-neutral, HEEHR-only gate;
  SUPERSEDES the 13 Jul "adjudicated KEEP electric-only"); documented the central
  function + `REBATE_RULE_CONFIG` + deprecated wrappers + `heehr_python_round`;
  retired "2024 = HEEHR only"; updated limitations #2/#5/#6; savings-fraction note
  (now consumed for fossil HOMES homes); sensitivity table row; by-fuel helper
  note; marked `_sub`/`_sub_june2026` golden rows PENDING + added a 2024-HOMES
  movement row; new Session Log row; header date.
- Memory: superseded `project_june2026_rebate_gate_electric_only` (+ MEMORY.md
  entry); added `project_rebate_consolidation_central_function`.
- Tests (`test_rebate_june2026.py`): rewrote the two stale tests
  (`test_june2026_mp3_is_eligible`,
  `test_funding_2024_heehr_and_homes_fuel_neutral`), added
  `test_2024_homes_is_fuel_neutral` and `test_june2026_homes_still_electric_gated`;
  added MP3 cost columns to the fixture. 13/13 pass.
- `scripts/verify_june2026_rebate_fossil_gate.py`: rewritten to a program x fuel
  crosstab for BOTH vintages -- HEEHR fossil = $0 (hard), HOMES fossil MAY be > $0
  (fuel-neutral), 2024-HOMES fossil-funding check. Smoke-tested PASS.
- SESSION_CHANGELOG_2026-07-14.md added.

VERIFICATION: rebate suite 13/13. Broader suite: 254 passed, 14 failed -- all 14
CONFIRMED PRE-EXISTING (identical with my constants + rebate edits stashed): 6
fuel-cost + 3 private-impact (MP3 now requires a rebate column, a 12-Jul
consequence) + 4 climate + 1 validation-framework (passes in isolation). ASCII
check: my added lines are ASCII-clean; the only non-ASCII in edited files is
pre-existing. geopandas collection error is an environment issue (not installed).

DONE.

---

## Final state for morning review

- ONE value move landed (2024 HOMES, fuel-neutral) -> `_sub` will rise on the
  full run. Everything else structure-only and byte-identical (verified).
- 2026 HOMES intentionally left electric-gated (deferred; documented in a code
  comment, CLAUDE.md, the memory, and a lock test).
- NO commits, NO `.ipynb` edits (per autonomous-mode instruction). Edited `.py`
  sources: constants.py, determine_rebate_eligibility_and_amount.py,
  visuals_adoption_dotplot.py, the scenarios + main-notebook `.py` exports, the
  rebate test, the verify script; docs (CLAUDE.md, changelog, this log); memory.
- Backport of the edited cells (rebate loop + both dotplot cells) to the `.ipynb`
  and the full 331k-home golden re-derivation are deferred to the researcher.
- HEADS-UP: editing `tare_scenarios_v2_3_EXPORT_12July2026.py` (Task 5) caused a
  jupytext/VSCode pairing to AUTO-SYNC the same change into the paired
  `model_scenarios/tare_scenarios_v2_3.ipynb`. I did NOT edit the notebook
  directly; I reverted it with `git checkout` to honor the no-`.ipynb` rule, so
  the working tree carries NO `.ipynb` change. When you backport, be aware the
  pairing may re-sync `.py` edits into that notebook on open/save -- confirm the
  synced cells match the intended edits (rebate guidance loop + import).
