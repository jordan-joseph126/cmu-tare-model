# Session Changelog -- 2026-08-19

## Closing the documentation and contract gaps found by an independent reconciliation

> Branch `joseph-2026-nature-comms-submission`. Follows the 18 August Tepper
> export rebuild and the full end-to-end pipeline run `2026-08-19_13-19`
> (National and Allegheny scopes, MP3 and MP4, `fixed_base`). One cell-value
> fix (the rebate eligibility token), the rest documentation. No existing
> exported value moved except the one named fix. Nothing was committed.

---

## 1. Why this session existed

The 18 August session made the Tepper household export reconcilable from its
own shipped columns. Once the full pipeline actually ran end to end
(`2026-08-19_13-19`), an independent reconciliation of the shipped Allegheny
CSVs against the vendored source data found the arithmetic holds -- but
surfaced one value-visible defect, one column-group description that was
actively misleading, and several data-dictionary claims that were either
stale (citing an earlier run) or wrong (an Allegheny denominator).

---

## 2. What changed, task by task

### Task 1 -- Audit the rebate-eligibility blank-cell defect (no edits)

Traced why `mp{mp}_rebate_eligibility_june2026` shipped blank for 1,284 of
1,356 Allegheny rdu (MP3) while the paired amount column shipped an explicit
`0.0` for the same rows -- breaking the export's "blank never means zero"
promise.

**Root cause, confirmed against actual file bytes, not assumed:**
`REBATE_NONE = "None"` (`constants.py`) is written correctly by
`calculate_rebate_program` -- verified directly, reading the National retrofit
results CSV with `keep_default_na=False` shows the literal text `None` on
disk for the affected rows. The loss happens one step later: every run of
`tare_model_main_v2_3.ipynb` reloads that same CSV back into memory via
`load_model_run_output` (`pd.read_csv(..., index_col=0, low_memory=False)`,
both the chunked and non-chunked paths), and neither call disables pandas'
default NA-string list, which includes the literal token `"None"`. The string
survives the first write and is swallowed on the very next read, before the
Tepper export ever runs. Confirmed the exact count: 197,402 literal-`"None"` +
71,320 genuinely-blank (excluded) rows = 268,722, matching the National CSV's
NaN count read with default `pd.read_csv` settings exactly.

Also confirmed in the same audit: the golden `heatingLCC_coolingLCC_unsub`
NPV and adoption-rate figures for both MP3 and MP4 reproduce exactly on this
run (see section 3 below), and the two `size_*_system_primary_k_btu_h`
columns hold the retrofit heat pump's ResStock-autosized capacity, not the
baseline system's -- see section 5.

### Task 2 -- Fix the rebate eligibility token

`REBATE_NONE` changed from `"None"` to `"Not Eligible"` (`constants.py`) --
not in pandas' default NA-string list, so it survives the same read that was
swallowing `"None"`. Updated the three docstrings/comments in
`determine_rebate_eligibility_and_amount.py` that named the old value, and
six hardcoded `'None'` assertions across two test functions in
`test_rebate_june2026.py` (two were caught by the Task 1 audit; two more were
list literals my initial grep pattern missed, and were only caught when the
suite actually failed on them). One unrelated `'None'` hit in
`test_calculate_equipment_installation_costs.py` (a ResStock
`hvac_cooling_type` value meaning "no cooling system") was confirmed out of
scope and left untouched.

**Verified:** 301 tests pass (one pre-existing collection error, confirmed
via `git stash` to fail identically at HEAD before this session's changes).
Root-cause mechanism closed: `'Not Eligible'` round-trips through the exact
same `pd.read_csv` call cleanly, zero nulls introduced. No other cell value
changed -- confirmed by inspection that nothing else reads or branches on the
literal string.

**Verified in Task 8** (section 7) once the researcher's re-run finished:
zero blank cells confirmed on the fresh files. The HEEHR/HOMES counts moved
by a small amount, traced to an unrelated, pre-existing issue -- see section
7 for the full account.

### Task 3 -- Document the system-size columns in the codebase

Added inline comments (no logic changes) at the two sites that assign and
consume `size_heating_system_primary_k_btu_h` /
`size_cooling_system_primary_k_btu_h`:

- `process_euss_data.py`, `df_enduse_compare` -- states plainly that these are
  the retrofit heat pump's capacity for that measure package, not the
  baseline system's, and that heating and cooling are equal because one heat
  pump serves both loads.
- `remdb_v4_installed_cost_utils.py`, `add_remdb_metrics` -- states that the
  same capacity column feeds both the `'replacement'` and `'upgrade'` cost
  lookups, because no separate baseline-capacity column exists in this
  pipeline.

**Verified:** 301 tests pass (comment-only diff).

### Task 4 -- Data dictionary corrections

Five corrections to `docs/tare_tepper_exports_data_dictionary.md`, all
recomputed from the shipped CSVs of the `2026-08-19_13-19` run rather than
carried over:

1. **Run timestamp** (section 1) updated to `2026-08-19_13-19`; the
   "update once re-run" instruction removed.
2. **Group 4 (household income):** `lmi_or_mui` spelled out on first use --
   Low-to-Moderate Income (LMI) / Middle-to-Upper Income (MUI), per the
   wording in `determine_rebate_eligibility_and_amount.py`'s own module
   docstring. Stored values unchanged (`LMI`/`MUI`).
3. **Group 5 (existing HVAC):** added a paragraph identifying
   `size_heating_system_primary_k_btu_h` / `size_cooling_system_primary_k_btu_h`
   as the retrofit heat pump's capacity, not a value read from the home's
   existing equipment. This paragraph went through a correction mid-session
   -- an earlier draft said the value was unrelated to the existing system,
   which the researcher flagged as unverified. A follow-up audit (see section
   5) showed the real picture is more specific than either "equal to the
   baseline" or "disconnected from it," so the data-dictionary text now
   states only what is directly confirmed (source, and the heating/cooling
   equality) and points to this changelog for the rest, instead of asserting
   a relationship that wasn't fully checked.
4. **Section 5 (blank-cell denominators):** corrected the claim that the
   Allegheny file's denominators are "just its row count." The pre-filter
   drops on `include_heating` alone, so 209 of the 1,356 exported rdu have
   `include_cooling = False`; the correct cooling denominator is **1,147**,
   not 1,356. Confirmed directly from the shipped CSV.
5. **Section 7 (rebate columns):** `mp{mp}_rebate_eligibility_june2026`'s
   documented value set updated to `'HEEHR'`, `'HOMES'`, `'Not Eligible'`.
6. **Group 11 (negative cooling savings):** added Allegheny shares beside the
   national ones, split by measure package (Room AC / Central AC), all four
   recomputed from the shipped household CSVs:

   | Scope | MP | Room AC | Central AC |
   |---|---|---|---|
   | National | MP3 | 90.68% | 10.84% |
   | National | MP4 | 61.97% | 3.46% |
   | Allegheny | MP3 | 95.3% | 16.5% |
   | Allegheny | MP4 | 74.5% | 4.5% |

   **This replaces a stale, MP-unsplit "about 54% / 2.5%" figure** that was
   in the data dictionary (and is also recorded in CLAUDE.md's 12 Jul 2026
   business-logic note on negative cooling savings). That older figure does
   not match either MP on the current run -- it is roughly halfway between
   the two MPs' Central AC shares and well below both MPs' Room AC shares.
   It most likely predates several sessions of fuel-price, degree-day, and
   capital-cost changes between 12 Jul and this run. **The data dictionary
   is corrected; CLAUDE.md's older business-logic note was not touched** --
   that edit wasn't in this session's scope, and the researcher should decide
   whether to update or annotate it as superseded.

### Task 7 -- Correct CLAUDE.md's stale national figure

Confirmed first, per the task's own branching instruction: the "12 Jul 2026
session" negative-cooling-savings note sits under "Masking and Validation
Rules" as prose, not as a row in the Golden Values table. That makes this a
direct-correction case, not a superseded-row case -- CLAUDE.md's
never-silently-overwrite rule governs the golden-value table specifically,
and this note is outside it.

Replaced the single, MP-unsplit "about 54% of Room AC baselines go negative
vs 2.5% of Central AC" claim with the same MP-split national figures Task 4
already computed for the data dictionary (90.68%/10.84% MP3, 61.97%/3.46%
MP4) -- not recomputed a second time, reused as instructed. Added a one-line
note recording what changed and why (`"national share corrected 19 Aug
2026"` in the heading, plus a sentence naming the old figure and that it did
not reproduce). Verified no other CLAUDE.md content changed: confirmed the
old "about 54%" string is gone from everywhere except the one sentence that
now references it historically, and confirmed the two golden-value
annotations added earlier in this session (Task 5) are still intact and
untouched.

---

## 3. Golden-value reproduction

Recomputed directly from `mp{3,4}_results_National_2026-08-19_13-19.csv`:

| | MP3 | MP4 |
|---|---|---|
| Mean `heatingLCC_coolingLCC_unsub` NPV, `fixed_base` | -$4,852.4058 | -$5,838.2317 |
| Adoption rate | 27.7140% (72,115 / 260,211) | 18.4416% (47,987 / 260,211) |

Both match the CLAUDE.md CONFIRMED rows (from the 17 Aug run) exactly. CLAUDE.md
annotated with this independent reproduction rather than adding a new row,
since no value moved.

---

## 4. Notebook backport needed

One cell in `tare_model_main_v2_3.ipynb` needs a manual edit (never edited
directly, per the standing rule). The "VERIFY: peak-load + whole-home
electricity columns" cell globs **every** `tepper_household_mp*.csv` under
`PROJECT_ROOT` and keeps "the newest by mtime" per `(mp, scope)` pair. Two
stale files from a 3 Aug test run
(`tepper_household_mp{3,4}_CO_2026-08-03_16-25.csv`) have no newer `CO`-scope
file to be superseded by, so they show up in every verification report even
though the current run only produces `National` and `Allegheny` scopes.

Replace the file-discovery block (the two lines building `all_files` and the
`for p in all_files:` loop immediately below it) with:

```python
# Only check the files THIS run actually wrote -- pin the glob to
# model_run_date_time so a stale scope from an older run (e.g. the 3 Aug
# Colorado test export) never gets silently included in this report.
# Filenames read tepper_household_mp{mp}_{scope}_{model_run_date_time}.csv.
all_files = sorted(
    search_root.rglob(f"tepper_household_mp*_{model_run_date_time}.csv")
)
by_file = {}
for p in all_files:
    token = p.name.split("_mp", 1)[1]           # e.g. "3_National_2026-08-19_13-19.csv"
    parts = token.split("_")
    mp = int(parts[0])
    scope = parts[1]                             # 'National', 'Allegheny', ...
    by_file[(mp, scope)] = p
```

Everything else in the cell (the `expected_new_columns` helper, the reporting
loop, the consistency checks) is unchanged.

---

## 5. Files changed this session

| File | Change |
|---|---|
| `constants.py` | `REBATE_NONE`: `"None"` -> `"Not Eligible"` |
| `private_impact/data_processing/determine_rebate_eligibility_and_amount.py` | 3 docstring/comment updates |
| `tests/private_impact/test_rebate_june2026.py` | 6 assertions updated |
| `energy_consumption_and_metadata/process_euss_data.py` | 2 comments added (Task 3), extended with a follow-up flag each (Task 6) |
| `utils/remdb_v4_installed_cost_utils.py` | 2 comments/docstring lines added (Task 3), extended with a follow-up flag (Task 6) |
| `private_impact/calculations/calculate_equipment_replacement_costs.py` | follow-up flag added to docstring (Task 6) |
| `private_impact/calculate_lifetime_private_impact.py` | follow-up flag added to docstring and at the replacement-cost subtraction (Task 6) |
| `docs/tare_tepper_exports_data_dictionary.md` | 6 corrections (see Task 4) |
| `CLAUDE.md` | 2 golden-value rows annotated with independent reproduction (Task 5); negative-cooling-savings prose corrected (Task 7) |
| `docs/tare_tepper_exports_data_dictionary.pdf` | deleted, superseded |
| `docs/SESSION_CHANGELOG_2026-08-19.md` | new, this file |
| `tare_model_main_v2_3.ipynb` | 1 cell backport pending (section 4) |

301 tests pass. No `_EXPORT_` snapshot was edited.

---

## 6. Known limitations / follow-up

### The heating replacement-cost credit is priced off the wrong system's capacity

Found while responding to a researcher question about the accuracy of the
Group 5 data-dictionary wording (Task 4, item 3). Audited before any fix was
proposed, per explicit instruction; **nothing about this was changed this
session** beyond the four forward-looking code comments listed below.

**The finding.** `mp{mp}_heating_replacement_installed_cost_{cost_scenario}`
is meant to be the avoided cost of replacing the home's existing heating
system (furnace or boiler) like-for-like -- the counterfactual capital cost a
household does NOT spend by choosing a heat pump instead, and it is credited
against the heat pump's own cost to form `net_capital_cost`, which feeds the
NPV directly. The REMDB v4 regression that prices it is built from
`size_heating_system_primary_k_btu_h`. That column, confirmed in this
session's Task 1 audit and again here, is the **retrofit heat pump's**
ResStock-autosized capacity for that measure package -- not a value read
from the home's existing furnace or boiler.

**Why the existing system's capacity is unavailable at all**, traced with
`inventory_tare_columns.py`'s `diff_stages()` against the repo's existing
column inventory (`tare_column_inventory_OLD.csv`, committed 3 Aug 2026 --
structural, predates this session's other changes, flagged as a caveat, not
re-verified against current code): the raw baseline EUSS file
(`baseline_metadata_and_annual_results.csv`) does carry its own
`out.params.size_heating_system_primary_k_btu_h` for the home's existing
system. But `baseline_home` -- the frame built by `df_enduse_refactored`,
the baseline-side counterpart of `df_enduse_compare` -- carries **zero**
`size_*` columns of any kind (`baseline_home -> loaded_mp3_fixed` diff: 0
dropped, matching). The value is not renamed or overwritten in a later merge;
it is simply never selected into any frame downstream of the raw baseline
file. `size_heating_system_primary_k_btu_h` first appears at
`mp{mp}_home_renamed`, sourced only from the MP-specific ResStock upgrade
output, and that name is carried unchanged all the way to the final loaded
frame and the REMDB v4 replacement-cost regression.

**Confirmed not a rounding/snapping artifact.** `_convert_pm1`
(`remdb_v4_installed_cost_utils.py`) does a pure unit conversion (kBtu/h /
12 for tons, x 1000 for BTU/hr) with no snapping to nominal equipment sizes.
The continuous autosized value feeds the linear regression directly, both for
the replacement lookup and the upgrade lookup.

**A 5-home spot check** (Allegheny, `2026-08-19_13-19` run) shows the gap is
real and can be large on the heating side:

| bldg_id | baseline heating capacity | baseline cooling capacity | MP3 capacity (both) | MP4 capacity (both) |
|---|---|---|---|---|
| 491 | 35.30 | 13.75 | 14.90 | 16.03 |
| 640 | 111.31 | 33.10 | 37.08 | 39.90 |
| 1081 | 94.45 | 39.23 | 39.23 | 42.22 |
| 2377 | 40.14 | 15.50 | 15.50 | 16.69 |
| 3205 | 34.33 | 15.43 | 16.11 | 17.34 |

All values kBtu/h. The retrofit heat pump's capacity lands close to the
baseline **cooling** capacity (bldg 1081: 39.23 matches exactly) but is
consistently far below the baseline **heating** capacity (bldg 491: 35.30 vs
14.90-16.03, roughly 2 to 2.5x smaller) -- consistent with the heat pump
being sized to the home's cooling load while the baseline furnace's stated
capacity reflects whatever a furnace happens to have been sized to, which
commonly exceeds the actual heating load in the field.

**What this session did NOT confirm, and the next session needs to:**
1. **The dollar and NPV magnitude across the full population.** The table
   above is 5 homes, not a distribution. Since the REMDB v4 regression is
   approximately linear in capacity, a smaller capacity likely understates
   the heating replacement credit for many homes, which would understate
   `net_capital_cost` and make the NPV look worse than it should for
   adoption -- but this direction and size were not measured at scale.
2. **Whether cooling shares the defect to the same degree.** The cooling
   replacement-cost column has the identical structural gap (no baseline
   cooling capacity survives either), but the 5-home sample above suggests
   the retrofit and baseline cooling capacities are usually close, unlike
   heating. Not checked at population scale.
3. **Whether MP3 and MP4 are affected differently**, since their retrofit
   capacities differ from each other as well as from baseline.

**Four forward-looking code comments added, no logic changed:**
`process_euss_data.py` (`df_enduse_compare`, both the heating and cooling
capacity assignments), `remdb_v4_installed_cost_utils.py`
(`add_remdb_metrics`, the `capacity_col` assignment), and
`calculate_equipment_replacement_costs.py`
(`calculate_replacement_installed_cost` docstring). The most complete note
is in `calculate_lifetime_private_impact.py`'s `calculate_capital_costs` --
both in the function docstring and at the exact line where the replacement
cost is subtracted to form `net_capital_cost` -- since that is the module a
reader is most likely to open when interpreting NPV output.

**A fix is planned for a separate, value-critical session.** Changing the
replacement-cost capacity input would move `net_capital_cost`, the NPV, and
the adoption rate for a large share of homes -- exactly the kind of change
this session's scope excluded ("without moving a single exported value").
Do not attempt the fix from these notes alone; the population-scale checks
above are prerequisites.

---

## 7. Task 8 -- Final export verification

The researcher's re-run finished as **PA scope**, not National (no fresh
National-scope files exist; every fresh file under `output_results/` is
`_PA_2026-08-19_17-41`, including the Allegheny subset pulled from it). The
researcher confirmed this PA-scoped run is the intended final check, so the
golden-value comparison below is against the PA population, not the National
one CLAUDE.md's table cites.

**Rebate-token fix, Allegheny scope (MP3 and MP4, fresh vs pre-fix):** zero
blank cells in `mp{mp}_rebate_eligibility_june2026` for both MPs -- the fix
holds. HEEHR/HOMES counts did **not** come back at the stated 43/29
baseline: MP3 measured 44 HEEHR / 29 HOMES (pre-fix: 43/29), MP4 measured 44
HEEHR / 31 HOMES (pre-fix: 43/32).

**That count difference is real but traced to something unrelated to Task
2.** A full 154-column diff (pre-fix Allegheny vs fresh Allegheny) shows
exactly 6 columns differ: `household_income`, `percent_AMI`, `income_level`,
`lmi_or_mui` (income-related, differing for the large majority of the 1,356
rows), plus `mp{mp}_rebate_eligibility_june2026` and
`mp{mp}_heating_rebate_amount_june2026_v4MID` (each differing for exactly 3
rows -- the same 3 bldg_ids for both MPs, all sitting within a few
percentage points of the 150% AMI HEEHR/HOMES cutoff). No other column
differs at all -- confirmed on the full 154-column set, not a sample.

Root cause: `calculate_percent_AMI` (`determine_rebate_eligibility_and_amount.py`)
samples household income from a normal distribution
(`generate_household_medianIncome_2025`) under a **fixed default seed**
(`random_seed: int = 42`), reset immediately before a single
`.apply(axis=1)` call, with the explicit documented purpose of giving MP3
and MP4 identical income draws **within one run**. There is exactly one call
site (`tare_scenarios_v2_3.ipynb`), applied to whatever rows are already
loaded for that run's `location_id`. A seeded draw sequence is assigned
positionally, row by row -- so the same seed produces the same *sequence* of
draws, but which physical home receives which draw in that sequence depends
on the row order and row count of the input, which differs between a
331,531-row National load and a 15,651-row PA load. The result: income
sampling is fully reproducible for repeated runs at the *same* scope, but not
comparable across a National run and a PA run for the same homes -- this is
a pre-existing characteristic of the pipeline, present before this session
and unrelated to the `REBATE_NONE` token fix. This was not previously
documented anywhere found in this session's audits; flagged here rather than
silently worked around. Confirmed at PA scale too (12,266 applicable homes,
not just Allegheny's 1,356): same 6-column pattern, 13,286-13,295 income-column
differences, 3,455 rebate-eligibility-label differences (proportionally much
larger than Allegheny's 3, consistent with more PA homes sitting near the
150% AMI threshold), and again zero differences in every other column.

**Golden-value NPV/adoption reproduction, PA population (unaffected by the
income-sampling issue, since the shipped case is unsubsidized):** compared
the pre-fix run's PA rows (`mp{mp}_results_National_2026-08-19_13-19.csv`,
filtered to `state == 'PA'`) against the fresh PA-scope run
(`mp{mp}_results_PA_2026-08-19_17-41.csv`), merged 1:1 on `bldg_id` (15,651
rows both sides, 12,266 with a usable NPV both sides). For
`heatingLCC_coolingLCC_unsub`, `fixed_base`, both MP3 and MP4: **maximum
absolute NPV difference across all 15,651 homes was exactly 0.0, and zero
adopter-flag differences.** MP3: mean NPV -$2,984.58, adoption 34.8280%
(4,272/12,266). MP4: mean NPV -$2,924.29, adoption 28.9499%
(3,551/12,266). This is not the National-scope CLAUDE.md figure (a different
population, 12,266 vs 260,211 homes) but it is an exact same-population,
before/after comparison across the code change -- and it confirms the NPV
and adoption computation moved by exactly zero for every home, exactly as
expected since Task 2 only ever touched the rebate eligibility label.

**Verdict:** Task 2's fix is confirmed working and confirmed scoped exactly
as intended -- zero blank cells, zero NPV/adoption movement, and the only
other differences trace entirely to a separate, pre-existing,
scope-dependent behavior in income sampling that this session did not touch
and is not responsible for. The 43/29 baseline in the task prompt does not
reproduce literally, but the reason is fully accounted for and is not a
defect in this session's work. Not resolved or worked around here, per
instruction to report rather than silently resolve; left for the researcher
to decide whether the income-sampling scope-dependence needs its own
follow-up note (similar treatment to section 6's capacity finding).

---

## 8. Outstanding

1. **Notebook backport** (section 4) -- pending the researcher.
2. **The capacity-source finding** -- see section 6 above and the four code
   sites it was flagged at. Planned for a separate, value-critical session;
   not attempted here.
3. **FLAGGED, NOT INVESTIGATED FURTHER -- scope-dependent income seeding
   (found during Task 8 verification, section 7; extended by Task 9).**
   `generate_household_medianIncome_2025`'s fixed seed
   (`calculate_percent_AMI`, `random_seed: int = 42`) is documented to give
   MP3/MP4 identical income draws **within one run**, but that guarantee does
   not extend across differently-scoped population runs (e.g. National vs
   PA) for the same physical homes -- draws are assigned positionally, so a
   different row count/order at the same seed reassigns them to different
   homes. **The 3-home Allegheny flip measured in Task 8 is a known
   instance, not a bounded count -- the full PA-wide extent is
   unquantified.** Task 9 confirmed the consequence is not cosmetic: for
   those 3 bldg_ids, the `_sub_june2026` NPV and net capital cost are NOT
   byte-identical pre- vs post-fix, moving by thousands of dollars per home
   in both MP3 and MP4 (consistent with the HEEHR/HOMES/ineligible
   reclassification, not a new anomaly) -- this does not touch the shipped
   Tepper export (the unsub case, confirmed unmoved in Task 8), but it does
   mean the subsidized NPV cases are exposed to this issue. **A CLAUDE.md
   caveat and full population-scale quantification are deferred to a
   separate future session -- not investigated further here.**
4. **A true National re-run** has not happened since the code changes in
   this session (Task 2's fix and the earlier 18 Aug export rebuild). The PA
   run verifies the fix works and moves nothing else; it does not reproduce
   CLAUDE.md's National-scope golden-value table literally, since that table
   is defined over the full 260,211-home population, not PA's 12,266. Those
   National figures were last measured on `2026-08-19_13-19` (pre-fix), and
   should still hold post-fix given Task 8's population-scale confirmation
   that the fix moves nothing outside the rebate-eligibility columns -- but
   this was not re-measured at National scale.

---

## Session closed

All ten tasks complete. **Verification passed** for everything checked:
Task 2's rebate-eligibility fix (zero blank cells, unchanged unsub
NPV/adoption, scoped to exactly the columns it should touch), Task 4's
data-dictionary corrections, Task 7's CLAUDE.md correction. Two items
surfaced during verification were diagnosed, confirmed unrelated to this
session's changes, and reported rather than silently resolved, per
instruction: the income-sampling scope-dependence (section 7, extended by
Task 9, Outstanding item 3) and the pre-existing 43/29 Allegheny HEEHR/HOMES
baseline not reproducing exactly for that reason. Task 9 additionally
confirmed the `_sub_june2026` NPV and net capital cost are NOT
byte-identical for the 3 affected homes -- a bounded, reported fact, not
chased further, per instruction. One finding (the heating replacement-cost
capacity-source defect, section 6) was deliberately left unfixed, flagged
forward at four code sites plus this document, for a separate value-critical
session. No exported value moved
this session except the one intended fix -- the rebate-eligibility token
from `'None'` to `'Not Eligible'`.
