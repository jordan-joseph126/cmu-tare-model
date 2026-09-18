# Session Changelog -- 2026-09-17

## Grid impact: custom per-building weighting, seasonal peaks, and a documentation pass

> Branch `joseph-2026-nature-comms-submission`. This session is ADDITIVE: the
> existing ResStock-weighted grid-impact path (uniform 242.131013 per dwelling)
> is not being replaced. Custom per-building weighting from the propensity-score-
> matched tax-parcel data is being added alongside it as a second, selectable
> option behind a `custom_weighting` parameter.
>
> This file is the running, plain-English record of the session. One entry per
> task: what the code did before, why it changed, what it does now. Task 8 turns
> these entries into a standalone refactoring guide. Written to be readable by a
> colleague who did not write this codebase.

---

## Task 1 -- Audit (no code edits)

### What the grid-impact code does today, in plain language

The grid-impact section answers one question: if homes in a county swap their
existing heating system for a heat pump, what happens to the county's hourly
electricity demand, and to its peak hour?

It works in four moves, spread across the live notebook
(`cmu_tare_model/tare_model_main_v3_0.ipynb`) and one module
(`cmu_tare_model/grid_impact/peak_load_functions.py`):

1. **Pick the buildings** (notebook cell 25). For each measure package, group
   every modeled home by its 5-digit county code and split each county into two
   sets: `all_filtered` (every home, the 100%-adoption bound) and `constrained`
   (only the homes where the heat pump pays for itself, `econ_adopter == 1.0`).
   Today this is done for exactly one NPV case -- `BASE_CASE_NPV_CASE`, the
   unsubsidized case with both replacement costs credited.
2. **Pull the hourly data** (cell 29). Open a `BuildStockQuery` connection to
   ResStock on AWS and run two kinds of query: one baseline (`upgrade_id="0"`)
   and one per measure package (`upgrade_id="3"`, `"4"`). Each returns 8,760
   hourly values per building for the whole-home electricity column
   `out.electricity.total.energy_consumption`.
3. **Add the buildings up** (cell 31, via `compute_county_scenario_profile`).
   For each hour, sum every building's kWh, giving adopters their retrofit value
   and everyone else their baseline value, then divide by 1,000 to get MW. The
   result is one 8,760-row county profile per (measure package, adoption
   scenario), plus a small `peak_dict` recording each series' peak MW and the
   hour it happened.
4. **Draw it** (`plot_county_demand_grid` and `plot_demand_panel`). A 2x2 grid:
   rows are measure packages, columns are the two adoption scenarios.

### The weighting question, and why it is the foundation of everything else

A ResStock row is not a home. It is a representative dwelling unit standing for
242.131013 real dwellings, and in this data release every row carries that same
weight. BuildStockQuery applies that weight itself, inside the SQL it generates:
it sums `enduse x sample_weight`. Confirmed directly in the installed package at
`buildstock_query/query_core.py`, in `_get_sample_weight` -- a falsy value or the
number `1` both resolve to a literal weight of 1 in the SQL, and anything else
numeric becomes that literal. So the numbers coming back today are already
weighted, which is why `compute_county_scenario_profile` only divides by 1,000
and applies no weight of its own.

One detail matters for how the new option has to be built:
`sample_weight_override` is an argument to the **BuildStockQuery constructor**,
not to an individual `TSQuery`. There is no way to vary the weight query by
query on one connection. So the custom path cannot ask BSQ for custom weights at
all -- it has to ask for raw, unweighted kWh (`sample_weight_override=1`) and
apply the per-building weights afterward in Python.

### Where each thing lives

- **Live module:** `cmu_tare_model/grid_impact/peak_load_functions.py` --
  `gisjoin_to_fips`, `find_adoption_column`, `extract_adopter_ids`,
  `compute_county_scenario_profile`, `plot_demand_panel`,
  `plot_county_demand_grid`.
- **Archived, non-time-aligned path:**
  `cmu_tare_model/grid_impact/archived_files/peak_load_functions_legacy.py` --
  `compute_peak_load_summary` and its helpers, moved out on 2 Sep 2026. It sums
  each home's own annual peak rather than finding one shared peak hour, so it
  overstates the true feeder peak. Not to be revived wholesale.
- **Parcel join:** `cmu_tare_model/grid_impact/build_parcel_frame.py` -- joins
  the household export to the tax-parcel match mapping, one row per matched
  parcel.
- **Matched-subset data:** `cmu_tare_model/tamar_grid_impact/` (gitignored).
- **Notebook cells that matter:** 25 (adopter IDs), 27 (AWS diagnostic), 29
  (AWS and BSQ init plus the two timeseries queries), 31 (profiles and the
  figure), 32 (heating-fuel table).

### The one piece of the legacy module worth borrowing

`compute_peak_load_summary` already solved the exact problem this session is
about, for the other peak-load path: it takes an `already_weighted` flag and
switches between "multiply each home by its sample weight" and "the frame is
already weighted, sum rows directly". It even refuses the dangerous combination
-- a row-duplicated frame with `already_weighted=False` -- rather than silently
returning an answer inflated by a factor of 242. That two-branch shape, and that
refusal, are the model for the `custom_weighting` parameter, even though none of
the code itself transfers (it works on per-home annual maxima, not hourly
profiles).

### What the audit found that the session plan did not anticipate

Five findings, each of which changes a later task:

1. **`query_unload_s3_bucket` is not set anywhere in this repo.** A search across
   every `.py`, `.ipynb`, and `.md` file returns nothing. The BuildStockQuery
   constructed in notebook cell 29 passes `workgroup`, `db_name`, `table_name`,
   `db_schema`, `buildstock_type`, and `skip_reports` -- and nothing else. The
   AWS fix may live outside the repo (an AWS-side workgroup setting, or an
   uncommitted notebook state), but it is not in the tracked code. Flagged, not
   touched.
2. **There is no colleague BSQ subset code in the repo.** The
   `tamar_grid_impact/` folder is gitignored and holds two CSVs and no Python.
   So there is no in-progress query to build on -- only the data it was keyed
   against.
3. **The `bldg_id` to weight dict is not stored anywhere; it has to be derived.**
   Neither CSV contains a per-building custom weight. `PSM_weighted_Resstock.csv`
   is the row-duplicated frame (37,663 rows, only 169 distinct `bldg_id`), and
   its `weight` column is still the uniform ResStock 242.13 carried through
   unapplied. The custom weight is the **number of tax parcels matched to each
   representative building**, obtained by grouping
   `PSM_output_buildYear07_09_2026.csv` on `representative_ID`: 169 buildings,
   weights ranging from 1 to 1,341 (median 86, mean 223, total 37,663 parcels).
   The two files agree exactly -- rows per building in the weighted frame equal
   the parcel counts from the mapping.
4. **The matched subset is in Colorado, not Allegheny County.** Its two counties
   are Boulder (08013) and Larimer (08069); the live figure and `TEST_FIPS` are
   Allegheny County, PA (42003). Of the 169 representative buildings, 145 survive
   into the current National MP3 output (the 24 missing are Mobile Home and
   multi-family types outside the study's allowed housing types -- a known scope
   mismatch, not a defect). So `custom_weighting=True` produces a Colorado
   two-county result and cannot be compared against the Allegheny figure
   directly. The two weighting modes really do cover different building sets.
5. **There is no annual-MWh aggregation anywhere to restore.** Searched the
   archived legacy module, all four archived
   `calculate_postTARE_ts_aws_peak_demand_*` snapshots, all three September
   notebook exports, and the git history for `MWh`, `GWh`, and annual-energy
   sums. Nothing matches. The nearest relatives are (a) `build_parcel_frame.py`,
   which requires `base_total_electricity_consumption` and
   `mp{mp}_total_electricity_consumption` columns but never sums them, and (b)
   `cmu_tare_model/adoption_kpis/demand.py`, which computes county annual GWh
   from TARE's own degree-day columns rather than from BSQ hourly data. Task 5 is
   therefore new code, not a port -- and the honest version is simple: annual MWh
   is the sum of the same hourly profile the peak is a maximum over, which makes
   the internal-consistency check Task 5 asks for immediate by construction.

### Two smaller things worth knowing

- **Exported weight precision.** The household CSVs store `weight` as `242.13`,
  while the Athena metadata column and CLAUDE.md both say `242.131013`. The BSQ
  path uses the Athena value. Any hand-check that mixes a CSV weight with a BSQ
  number will be off by about 0.0004 percent -- small, but enough to look like a
  bug if an exact match is expected.
- **The BSQ cache is empty** (`cmu_tare_model/grid_impact/.bsq_cache/`), and the
  2 Sep session recorded the two timeseries queries taking roughly ten minutes.
  So the `custom_weighting=False` regression target should be captured once and
  saved, not produced by re-querying AWS on every comparison.

### Conventions confirmed, so later tasks follow them instead of inventing

- **Tests:** there is a real pytest suite. `cmu_tare_model/tests/` mirrors the
  package layout, and `tests/adoption_kpis/test_peak_load_functions.py` already
  covers all four public functions on synthetic data with no AWS connection --
  one class per function, `pytest.fixture` inputs, and a `_make_hourly_df`
  helper. The Task 4 season-consistency test belongs there. The inline `[OK]` and
  `[WARN]` print style stays for the notebook cells, which cannot be
  unit-tested.
- **Session notes:** dated `SESSION_CHANGELOG_YYYY-MM-DD.md` files in
  `cmu_tare_model/docs/` (six of them, 12 Aug through 2 Sep), structured as
  "what was wrong" then "what changed, task by task". This file follows that.
  The Task 8 refactoring guide goes in the same directory.
- **Missing referenced docs:** CLAUDE.md points to `docs/REFERENCE_VALUES.md`
  and `docs/SESSION_LOG.md`. Neither exists anywhere in the repo. Flagged for
  the researcher; nothing in this session depends on them.

### Verified facts underpinning the tasks ahead

- The three peaks all come from one column and one mechanism. The absolute peak
  is the maximum over all 8,760 hours of the same profile; the seasonal peaks are
  the maximum over Dec/Jan/Feb and Jun/Jul/Aug of that same profile. Nothing new
  gets queried.
- The month boundaries needed for that filtering already exist inside
  `plot_county_demand_grid` as `days_in_month` and `month_start_hours` (non-leap
  year, cumulative hours). Task 3 factors that into a shared helper rather than
  reimplementing it -- and does not edit the plotting functions.
- `compute_county_scenario_profile` raises if the profile is not exactly 8,760
  rows. Any change to how buildings are queried or filtered must keep one row
  per building-hour.
- `find_adoption_column` is called with every argument by keyword in cell 25, on
  purpose: a positional call once routed a figure to the wrong NPV case. Task 6's
  loop keeps that.
- All nine `econ_adopter` columns per measure package are present in the current
  output (checked against the 2026-09-17_19-38 National MP3 run, 195 columns), so
  Task 6's scenario loop has real data to run on.

**Status:** audit only. No code changed. This changelog file is the session's
only new file so far.

---

## Task 2 -- Custom weighting as an additive option

### What changed and why

Two clarifications from the researcher, made at the start of this task, are
folded in because they turned out to be load-bearing, not cosmetic: the
grid-impact notebook cells hardcoded "Allegheny" into six variable names, and
imported the county FIPS as a fixed constant (`TEST_FIPS`). Task 1's audit had
already found that the custom-weighting matched subset lives in two Colorado
counties, not Allegheny -- so those two things had to change for
`custom_weighting=True` to make sense at all, not as a separate style pass.

**`compute_county_scenario_profile`** (`peak_load_functions.py`) gained two
keyword-only parameters, `custom_weighting` and `weight_dict`. Before, it
assumed every kWh value handed to it was already multiplied by BuildStockQuery's
own sample weight. Now it still assumes that when `custom_weighting=False` (the
default) -- byte-identical to before, confirmed two ways: multiplying a float by
`1.0` introduces no floating-point rounding (a mathematical guarantee, not just
an empirical check), and the project's own 32-test suite for this module passes
unchanged. When `custom_weighting=True`, the function instead trusts nothing is
pre-weighted: it restricts every input (baseline buildings, upgrade buildings,
and the adopter list) to only the buildings present in `weight_dict`, reports
how many were excluded from each, and multiplies each row's kWh by that
building's dict weight before summing. A building missing from the dict is
dropped, never defaulted to the uniform weight or to zero.

**A new function, `build_weight_dict_from_mapping`** (`build_parcel_frame.py`)
turns Tamar's tax-parcel match file into the `{bldg_id: weight}` dict the
function above needs. The weight is the count of real tax parcels matched to
each representative building -- the same number `build_parcel_frame` already
uses to row-duplicate its frame, just packaged as a lookup dict instead of row
duplication, because the new aggregation multiplies rather than duplicates.
Checked against the real data: 169 buildings, weights 1 to 1,341, summing to
the 37,663 matched parcels -- matching Task 1's audit numbers exactly.

**The notebook's BSQ query cell** (handed over as cell text, not a live edit,
per this project's `.ipynb` convention) now threads a `CUSTOM_WEIGHTING` flag
through the building-selection, query, and profile-computation steps:
  - `False`: prompts for a county FIPS interactively (`input()`), instead of
    importing the fixed `TEST_FIPS` constant -- so the section now runs for any
    county, not only the Allegheny County case study used in the paper.
    `TEST_FIPS` itself is untouched in `constants.py` (other code still uses
    it); the grid-impact cells just stopped importing it.
  - `True`: loads Tamar's mapping file, builds the weight dict, and queries
    BuildStockQuery with `sample_weight_override=1` (raw kWh) instead of the
    default (BSQ's own uniform weighting).
  - Six variables renamed from an `allegheny_*` prefix to a `case_study_*`
    prefix, matching wording already used in this codebase's archived
    comments ("Allegheny County case study panel").

Two more things were added at the researcher's request while touching this
cell:
  - `my_run.save_cache()`, called once after the queries succeed. BSQ already
    caches every query result in memory and loads whatever is on disk
    automatically when constructed, but nothing writes that cache back to disk
    on its own -- so without this call, every kernel restart re-runs the
    roughly 10-minute baseline/upgrade queries from scratch even though the
    same queries have been run many times before. This one line makes the next
    run reuse the saved cache instead.
  - A comment at the `BuildStockQuery(...)` call noting that this project is
    pinned to buildstock_query 0.2.0 (confirmed via the installed
    dist-info), which has no `query_unload_s3_bucket` parameter -- that was
    added in the 0.30 release a colleague is using on a related effort. Not a
    bug to fix; a version difference to not accidentally "fix" into a
    `TypeError`.

**A judgment call, flagged rather than made silently:** the fuel-distribution
table cell (cell 32) is keyed to a single county's FIPS, which only exists
when `CUSTOM_WEIGHTING=False` (the matched subset spans two counties, so
there's no single FIPS to key it by). Rather than build new multi-county
aggregation logic for it -- which the original task list explicitly reserves
for Task 7 as documentation-only, no logic change -- it's guarded to run only
in the default weighting mode, printing a one-line skip notice under custom
weighting. Handed to the researcher as a flagged decision, not applied as a
foregone conclusion.

**A pre-existing bug found and fixed in passing:** the project's own test
module (`test_peak_load_functions.py`) imported `BSQ_ELEC_COL` from
`peak_load_functions.py`, but that module never imported it from
`constants.py` in the first place -- confirmed via `git show HEAD` that this
predates this session. All 32 tests in that file failed to even collect
before this fix. Fixed by adding `BSQ_ELEC_COL` to the module's existing
import from `constants.py` (the same pattern already used for `BLDG_ID_COL`),
with a `# noqa: F401` comment explaining it is imported for re-export, not
local use.

### Verification

- The project's own pytest suite for this module: 32 passed, 0 failed (was 0
  collected, 1 error, before the import fix).
- A manual synthetic-data script (four checks, since the pre-existing test
  file did not yet cover `custom_weighting`): `custom_weighting=False`
  reproduces the exact pre-existing aggregation; `custom_weighting=True`
  applies the dict weight and excludes an uncovered building rather than
  defaulting it; excluding a building from `weight_dict` also removes it from
  the adopter set, not just the totals; a missing or empty `weight_dict` with
  `custom_weighting=True` raises `ValueError`.
- `build_weight_dict_from_mapping` run against the real
  `PSM_output_buildYear07_09_2026.csv`: 169 buildings, sum of weights 37,663.0
  (exactly the matched-parcel count), min/max 1.0/1,341.0 -- matches Task 1's
  audit.
- Line length (<=88 chars) and no alignment padding (E221/E241) checked on
  both edited files; no violations in the new code.
- The notebook cell text has not been run against live AWS/BSQ this session
  (that would issue real, billed Athena queries) -- it is handed to the
  researcher to paste in and run.

### Correction: comment style

The first pass of this task's edits put each block's explanation in one large
comment ahead of the code it described -- a two-mode explanation in
`compute_county_scenario_profile`'s docstring, and multi-line header blocks at
the top of the notebook cells. The researcher pointed out this made the
notebook cells hard to follow: a reader has to hold the whole explanation in
mind before reaching the code it refers to, rather than reading each line
next to the sentence that explains it.

Fixed in both places: `compute_county_scenario_profile`'s docstring keeps only
the standard Args/Returns/Raises sections, and the "two weighting modes"
explanation was broken into short comments sitting directly above the specific
lines each part describes (the weight-dict filter, the adopter-set
intersection, the weight-multiplier branch). The notebook cell text handed to
the researcher was rewritten the same way -- no header block longer than two
or three lines, with an explanation next to each non-trivial line instead.
This was a comment-only change; re-ran the project's test suite (32 passed)
and the manual verification script to confirm nothing else moved.

### Correction: a dropped print line, and a pre-existing print-formatting bug

The researcher ran the handed-over cell 29 against live AWS and reported two
problems from the real output.

The first was introduced by the comment-restructuring pass just above: while
rewriting the Step 6 loop's comments, the `Hours/bldg`, `kWh range (wtd)`, and
`Query time (s)` print lines were dropped entirely, so the MP3/MP4 summaries
printed only `Rows` and `Buildings` -- missing the same information the Step 5
baseline summary still printed. Fixed by restoring the three lines to the
handed-over cell text.

The second predates this session: `check_athena_output_location`
(`diagnose_bsq_aws.py`) and the notebook's own AWS-credentials print both
used a triple-quoted f-string starting with a bare newline after the opening
`"""` (no backslash), with each content line indented to match the
surrounding code's indentation level. That produces exactly what the
researcher saw -- a blank line before the message, every line pushed right by
the code's own indentation, and a trailing whitespace-only line before the
closing `"""`. Fixed `diagnose_bsq_aws.py` to follow this project's own
print-statement convention (a backslash right after the opening quotes, and
content starting at column 0) -- confirmed clean with a standalone
reproduction using the researcher's own printed values. Fixed the same
pattern in the handed-over AWS-credentials block. Neither fix touches any
computed value; both are print-formatting only.

### Removed a redundant AWS preflight check, and factored the BSQ query cell

The researcher asked to remove the AWS-credentials print and
`check_athena_output_location` call from the BSQ query cell, since the
`RUN_BSQ_DIAGNOSTIC` cell above it already checks both. Checked
`diagnose_bsq_aws.py` directly before agreeing: its stages 2, 3, and 4
already cover AWS credentials and the Athena bucket-write check, but the
module's own docstring said the query cell was supposed to call
`check_athena_output_location` too, as a second, independent preflight. That
was a real design tradeoff (a kernel restart that skips the diagnostic cell
gets a less friendly BSQ error instead of a fast, named one), not a mistake
to fix silently -- flagged it and asked which way to go. The researcher chose
to remove both checks and rely on the diagnostic cell alone. Removed the
block from the query cell (plus its now-unused `boto3`/`botocore` imports)
and corrected `diagnose_bsq_aws.py`'s docstring so it no longer claims the
query cell calls it directly.

The researcher also pointed out three repeated or under-explained blocks in
the same cell and asked for them to become shared functions instead. Added
to `peak_load_functions.py`:

  - `prepare_bsq_timeseries` -- the rename/downcast/hour-index cleanup that
    both the baseline and each MP's upgrade query needed, previously written
    out twice.
  - `summarize_hourly_timeseries` -- the printed summary (rows, buildings,
    hours per building, kWh range, query time) and the full-year check,
    previously duplicated with two slightly different formats (the upgrade
    version was also missing three of the five lines, from the earlier
    print-fix correction above) and using a bare `assert` for the check.
    Both call sites now share one implementation, one format, and raise
    `ValueError` on an incomplete year instead of `assert` (matching how
    `compute_county_scenario_profile` already handles the same check --
    `assert` can be silently skipped under Python's `-O` flag).
  - `check_upgrade_building_coverage` -- the baseline-vs-upgrade
    building-set comparison and its associated note/error.

All three verified against synthetic data mirroring real BSQ output shape
(rename/downcast/sort/hour-index correctness, the summary printing and
raising on incomplete coverage, the coverage check's note and its error),
plus the project's 32-test suite, unaffected by this change.

### Correction: comment density in the handed-over cell text

The researcher gave a concrete example of the comment density they want --
short, stating what a block does and why in one or two lines, not a
mechanism walkthrough -- using the `save_cache()` comment as the model. Several
comments added earlier this session (in `compute_county_scenario_profile` and
the new functions above) were longer than that. Tightened all of them to
match -- the same one-two-line density already used, unedited, in
`plot_demand_panel`/`plot_county_demand_grid` elsewhere in this module, which
turned out to be the right reference point for this project's existing
voice. Comment-only change; re-ran the 32-test suite to confirm.

### Cleaned up the heating-fuel table cell and moved its call

The heating-fuel distribution cell (the researcher's own name for it:
"GRID IMPACT -- Allegheny County baseline heating-fuel distribution table")
had accumulated exactly the problems flagged this session before: single-
letter-prefixed throwaway variable names (`_mp`, `_df_tare`, `_fuel_results`,
`_col_width`), several temporary variables that existed only to be unpacked
once, and almost no comments explaining what the block of print statements
was building toward.

Moved its logic, unchanged, into a new function,
`print_heating_fuel_distribution_table` (`peak_load_functions.py`) --
real variable names, a docstring, and comments at the density the researcher
asked for. The only two intentional differences from the original: it takes
`case_study_fips` as a parameter instead of the hardcoded `TEST_FIPS`
constant (so it works for whichever county was entered at the prompt, not
only Allegheny), and its column-divider width is computed from
`len(selected_mps)` instead of a hardcoded `4` (a latent assumption of
exactly two measure packages in the original). Neither changes what prints
for today's two-MP case.

Verified against the real 2026-09-17_19-38 National MP3/MP4 output for
Allegheny County: ran the original cell's logic and the new function side
by side and diffed their printed output line by line. The header and footer
wording differ on purpose (no more hardcoded "Allegheny"); the entire table
body -- divider, headers, every fuel row, the TOTAL row -- is identical.

The call moved into the weighting-mode-and-scope cell's default-mode branch,
right after `case_study_fips` and `case_study_bldg_ids` are established and
before the BSQ queries run -- so the fuel mix behind the case study is
visible immediately, without waiting on a ~10-minute AWS query first. The
table only ever covered one county, so it is not called from the
custom-weighting branch (which has no single FIPS); the researcher asked
for a direct cleanup and move, not new logic to make the table span the
matched subset's multiple counties -- flagged as a possible follow-up, not
built. The old standalone cell is now redundant and should be deleted by
the researcher; leaving both in place would print the table twice.

Also saved a memory (`feedback_concise_inline_comments`) recording the
comment-density preference from the correction above, so it carries into
future sessions on this project.

---

## Interim refactoring guide -- reconciliation session (18 Sep 2026)

A separate continuation-session prompt asked for progress to be reconciled and an interim
refactoring guide produced, without executing Task 3 or any later task. Two things were
done first, in order, before writing anything: an independent re-audit of the actual
codebase (not trusting the changelog above, Jordan's status report, or the continuation
prompt's own "Current state" claims), and a summary of this session's real back-and-forth,
capturing every point where execution departed from a task's original contract.

The re-audit found two claims in the continuation prompt that do not match the actual
repository: the AWS/BSQ setup is described there as fully resolved with
`query_unload_s3_bucket` now set wherever `BuildStockQuery` is constructed, but the
installed package is still confirmed `buildstock_query-0.2.0`, which has no such parameter
at all -- this session's own code comment (still live in the notebook) explains exactly
why. And the colleague's in-progress BSQ code, described as already running against a
matched subset, still was not found anywhere in this repository -- same as Task 1's
original finding. Both are called out explicitly rather than carried forward as fact.

The re-audit also found two new TODO comments the researcher added directly to the
notebook outside this session's conversation (commit `438b197`, "Added TODO placeholders
for Tamar to resume"): a note that custom weighting may eventually need a
feeder -> building -> weight structure, not just building -> weight, and a note to
parameterize the hardcoded `BuildStockQuery(...)` constructor arguments as user input.
Neither is implemented or was previously scoped; both are flagged for whoever resumes
Task 3 onward to raise with the researcher.

Confirmed unchanged and still valid: Task 5's contract remains wrong as originally
written (no legacy annual-MWh code exists to port -- unchanged from Task 1's original
finding), Task 3's contract is fully intact (no season-peak keys exist yet in
`peak_dict`), Task 6's contract is fully intact (only a comment scaffold exists, no
functional loop), and the 32-test suite still passes.

The guide itself lives at `cmu_tare_model/docs/GRID_IMPACT_REFACTORING_GUIDE.md` --
the same file Task 8 will later finalize, not a second document. It marks Objective 1
(Task 2) complete with a full what-changed walkthrough; Objectives 2-4 (Tasks 3, 4, 5,
6) as not started, each with next steps and an explicit note on whether that task's
contract still holds; Objective 5 (Task 7) as complete only for code that currently
exists, since it cannot be more complete than the code it documents; a "Where the plan
changed" section capturing every deviation from this session (the Allegheny-naming and
TEST_FIPS generalization turning out to be load-bearing, the two pre-existing bugs found
and fixed, the regression introduced and caught, the AWS-check removal, the three new
helper functions, the heating-fuel-table rewrite and relocation, the two rounds of
comment-density correction, and the pre-commit git-hygiene catch); and states plainly
that Task 4's test results are not available, rather than omitting that section.
