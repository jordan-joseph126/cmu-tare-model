# TARE model -- Stage 2: data validation and the data share with Chris -- session prompt

CLAUDE.md is the source of truth for conventions, file rules, reference values, coding
standards and every non-negotiable; it is auto-loaded, so this prompt adds only
session-specific state and the task list. Audit before editing and never stage or commit,
per CLAUDE.md. Its stop-gate rule is changed for this session only as "Auto-approve"
below says.

## Context

**Priority: show Chris, with numbers he can check, what changed between the files he has
and the files he will get, and give him a README and a Data Dictionary for the new ones.**

Read `cmu_tare_model/docs/DF_COST_FIXES_AND_DATA_SHARE_PLAN.md` before Task 1: section 5
is this stage's plan, section 6 the list of changes since 19 Aug 2026, sections 2 and 3
the decisions (D-9 to D-12; Q10 names the packages Chris receives, Q11 the shape of the
comparison), section 7.1 the runs and section 7.2 the results after Stage 1. Its
decisions and values win over anything below that differs, except where "Current state"
says the plan is wrong about a file's location.

This session analyses and writes; it changes no model code, runs no model and moves no
value. Keep the `s9_` scripts short: reuse the model's own helpers and the Stage 1
scripts named below rather than re-deriving them. Out of scope: any fix to the model (a
problem found is reported, not fixed), and the email itself, which is drafted elsewhere
from Task 5's file.

## Auto-approve: on until the next commit point

The researcher turns it on with this start message, with the scope "until the next
commit point".

- While it is on, do not wait at a task gate: report each task's result in the chat and
  go on to the next task. Show each repo diff, apply it without waiting, and confirm the
  file's line endings did not change.
- It ends at the first commit point. Stop there, give the git commands (Git Bash) and the
  commit message in the researcher's bracketed-bullet format, and wait for "committed".
  After that, every gate is a stop again.
- It ends at once, and you stop and wait, on any of these: a failed check; a number that
  contradicts the plan or `docs/REFERENCE_VALUES.md`; a missing input; a file with mixed
  line endings; a change this prompt does not ask for; any researcher message other than
  a question.
- It never covers a commit or any other git command that changes history, a change to
  model code, a notebook, `validation_framework.py` or an `*_EXPORT_*.py` file, or a
  model run.

## Current state the tasks depend on

- **Stage 1 is done.** HEAD is `0624802` on `resstock2025-dual-fuel-codebase-update`
  (boiler fix `5a91996`, its records `0624802`). The working tree holds one uncommitted
  line of the researcher's, `GRID_IMPACT_ANALYSIS = False` in `constants.py`: leave it,
  and do not report it as a stray change. Tests: 417 passed, 1 skipped.
- **The new data.** Final runs `2026-10-09_00-46` (ResStock 2022.1.1, MP3 and MP4:
  221,205 rdu) and `2026-10-09_00-38` (ResStock 2025.1, MP5: 161,983 rdu). Their Tepper
  files (main, detailed and county; national and Allegheny) are in
  `cmu_tare_model/output_results/tepper_export/`. A main file holds 167 columns (MP3,
  MP4) or 182 (MP5), a detailed copy 317 or 332, and `bldg_id` is one more on disk. That
  folder holds the files of 25 runs: select every file by its stamp, never by "latest".
- **The old data: the files as sent to Chris on 19 Aug 2026** (run `2026-08-19_20-56`).
  They are in `~/tare_port/s9_inputs/`, copied there on 9 Oct 2026 from the researcher's
  two folders of what went to Chris. They are NOT in `tepper_export/`: section 5 of the
  plan says they are, and is wrong on this point. Change none of them.
  - `tepper_household_mp{3,4}_National_2026-08-19_20-56.csv`: 331,531 rows, every
    occupied single-family rdu, and 155 columns on disk (`bldg_id` and 154 others).
    `weight` is written 242.13. `include_heating` is True on 260,211 rows and
    `include_cooling` on 250,576; there is no `include_sample`. The result columns are
    blank on the 71,320 rows that did not pass the heating check. Adopters of the shipped
    case (`heatingLCC_coolingLCC_unsub`): 70,715 (MP3) and 47,094 (MP4), which is 27.1760%
    and 18.0984% of the 260,211 rdu with a flag (21.3298% and 14.2050% of all rows).
  - `tepper_household_mp{3,4}_Allegheny_2026-08-19_20-56.csv`: 1,356 rows, the county's
    rows with `include_heating` True, same 155 columns. The national file has 1,610 rows
    for the county (`county == 'G4200030'`), so the two scopes were not cut the same way.
    Adopters: 256 (18.88%) and 125 (9.22%). 210 rows have a blank cooling replacement
    cost (the plan counts 209 with no AC).
  - `tepper_county_mp{3,4}_National_2026-08-19_20-56.csv`: 3,098 counties and the same 11
    columns as today. **Its two headline columns were measured over every rdu, not over
    homes with a result.** `home_count` is every rdu in the county times 242.13
    (80,273,601 homes in all), and `adoption_rate_pct` is adopters divided by every rdu in
    the county, a blank flag counted as a non-adopter (true in all 3,098 counties). Today
    both are measured over the study sample. An old and a new county rate are therefore
    not the same measure: say so wherever they are shown side by side, and give the old
    rate over rdu with a flag as well. 8 counties have a blank
    `operating_cost_pct_change`. No Allegheny county file was sent.
  - `README_sent_2026-08-19.docx` (`README.docx` in the folder sent), one sentence: "Found
    a bug with the capital cost estimation. This changed the cost estimates but did not
    substantially change the results."
  - Two source CSVs (see "Source data").
  - Checked 9 Oct 2026: all 154 columns of each household file are in the 19 Aug run's
    saved files with the same values (largest relative difference 3.2e-16), and each
    Allegheny file equals the national file's rows for the same homes.
  - The 19 Aug run's own files are on disk too, for anything the files as sent do not
    hold (the other eight NPV cases, rebates under 2024 guidance). They are under
    `output_results/` with the same stamp:
    `retrofit_mp{3,4}_results/summary_mp{3,4}_fixed_base/mp{3,4}_results_National_2026-08-19_20-56.csv`
    (195 columns, 331,531 rows) and the fuel-cost files under
    `supplemental_data_fuelCosts/`.
  - Every old column is in the new MP3 file. The new file adds 13: `include_sample` and
    the 12 base-year fan, pump and backup columns (168 on disk against 155). The
    workbook's `Data Dictionary` sheet has a row for every old column: its 98 rows give
    154 names for one package (`{mp}`, and `{year}` for 2025 to 2039), and one of them,
    `Longitude / Latitude`, stands for the columns `Longitude` and `Latitude`.
- **Chris's workbook:** `~/tare_port/s9_inputs/TARE_Export_Tepper_Cashflow_Model_JMJ_DRAFT.xlsx`,
  in place on 9 Oct 2026 (564,522 bytes). `README`: 35 rows in one column, 22 with text.
  `Data Dictionary`: 99 rows by five columns, `column_name`, `group`, `description`,
  `units`, `source`, one row per column or column pattern. The other three sheets,
  `Fuel Price Data` (209 by 9), `Projection Factors` (41 by 29) and `TARE MP3 Data`
  (1,357 by 155), are empty placeholders: formatted cells with no values, which the
  README tells Chris to fill from the CSVs. Finding no data in them is expected, not a
  missing input. Read the file with `openpyxl`; never save it. If the file is not there,
  that is a missing input: stop.
- **Source data.** `tepper_export/source_data/` holds one set of three CSVs,
  byte-identical to the model's input files of the same names under
  `cmu_tare_model/data/` (git-ignored). The two that went to Chris on 19 Aug,
  `eia_fuel_price_data_2025_usd2025.csv` and `aeo2026_fuel_price_factors_2025_2050.csv`,
  are in `s9_inputs/` and are byte-identical to today's copies (checked 9 Oct 2026): the
  fuel prices and their factors have not changed. They have 209 and 41 lines, the shapes
  of the workbook's two placeholder sheets. The third,
  `aeo2026_degree_day_factors_2025_2050.csv` (21 lines), was not in the folder of what
  was sent and has no sheet in the workbook.
- **What Stage 1 changed in the new files.** The credit for an old natural gas, propane
  or fuel oil boiler is priced on a REMDB boiler row, not as a gas furnace. Only homes
  with a fossil boiler differ from the files of 7 Oct (20,973 rdu in 2022.1.1, 2,528 in
  2025.1), in the heating credit, the net capital cost, the NPV and the adopter flag;
  their mean credit went from about $3,470 to $7,640. Electric furnaces and electric
  boilers are still priced as baseboard (CLAUDE.md Limitation 16). Section 6 of the plan
  has the row, and group 12 of `docs/tare_tepper_exports_data_dictionary.md` the pricing
  table.
- **Two releases, two kinds of comparison.** Old MP3 and new MP3 are the same ResStock
  2022.1.1 homes: match them by `bldg_id`. New MP5 is ResStock 2025.1, a different set of
  homes with the same `bldg_id` values: compare it with MP3 by group only (all homes,
  baseline fuel, system type), never row by row. One rdu is 242.131013 homes in 2022.1.1
  and 253.90367 in 2025.1; label every count rdu or homes.
- **Populations differ.** The old national files hold every occupied single-family rdu
  (331,531), 260,211 of them with a result; the old Allegheny files hold the 1,356 with a
  result; the new files hold the study sample only (221,205 nationally, 1,146 in
  Allegheny). Show the old data three ways where it matters: as shipped, limited to rdu
  with a result, and limited to rdu with a central or room AC of their own. Say which is
  which in every table, and never divide old adopters by all 331,531 rows without
  saying so.
- **Three things that look like differences and are not** (the same in runs made before
  any change): the county Tepper file is sorted by adoption rate, and counties with
  equal rates change places between runs, so match it by `county`; `occupancy` reads as
  text in the national MP5 file (it holds `10+`) and as numbers in the Allegheny one; a
  number in a Tepper file can differ from the same number in the results file in its
  last digits (relative difference up to 7.5e-13), so compare with a tolerance of 1e-11.
- **Helpers that depend on the release.** `cmu_tare_model` reads the release once, at
  import, and the default is 2022.1.1. A script that needs a 2025.1 name from a model
  helper sets `TARE_RESSTOCK_RELEASE=2025.1` for that one process, never globally. The
  files' own headers are the authority for which columns exist.
- **Stage 1 scripts to reuse**, in `~/tare_port`: `s8_task9_tepper_checks.py` (the checks
  of sections 13 and 14.4 of the data dictionary, 25 per package; all passed on both
  final runs), `s8_task8_reference_rows.py` (adoption, NPV and by-fuel figures of a run)
  and `compare_runs.py`. Copy and adapt them under an `s9_` name; do not edit the `s8_`
  files.
- Scratch files go in `~/tare_port` with the prefix `s9_`; outputs go in
  `~/tare_port/s9_outputs/`. At the start `~/tare_port` holds this prompt
  (`s9_stage2_kickoff_prompt.md`) and `s9_inputs/` with ten files: the workbook, the
  README sent on 19 Aug, six Tepper files as sent (two national household, two Allegheny
  household, two national county) and two source CSVs as sent. Change none of them. A
  national household file as sent is about 400 MB: read only the columns a step needs.
  `openpyxl` 3.1.5 is installed; install nothing.

## Tasks (in order)

**Task 1 -- Audit (no edits).** Confirm the branch, HEAD and the one uncommitted line.
Locate every input named above; a missing workbook is a stop. For each file give rows,
columns and rdu. Give the header differences between the old MP3 file and the new MP3 and
MP5 files: added, removed, renamed. Read the workbook's sheets and report their exact
layout (rows, columns, header row, text styling). Report.

**Task 2 -- Comparison.** In an `s9_` script, for national and for Allegheny, in the
shape Q11 decides: old MP3, then new MP3, then new MP5.

1. No-rebate adoption of the shipped case, all homes and by baseline heating fuel, with
   rdu, homes and the share of the sample in each fuel.
2. The parts of the adoption decision, mean and median per home, all homes and by
   baseline fuel: credit for the old heating system; credit for the old AC; heat pump
   installed cost; backup furnace installed cost (MP5 only); net capital cost; discounted
   heating and cooling savings; NPV.
3. Old MP3 to new MP3, matched by `bldg_id`: rdu that start and stop adopting, by fuel,
   and the part that moved most for them. Tie each move to a row of the plan's section 6.
4. The national county files, old against new, matched by `county`: the number of
   counties (3,098 old, 3,079 new for MP3 and MP4), and `home_count` and
   `adoption_rate_pct` with the old rate given both as shipped (over every rdu) and over
   rdu with a flag. If no row of the plan's section 6 covers the change of what the
   county rate is measured over, say so.

Save each table as CSV and one summary as `s9_outputs/s9_comparison_summary.md`. Report
the headline tables in the chat.

**Task 3 -- Validation of the new files.** Run the Stage 1 checks again under an `s9_`
name on every new file: row and column counts, the three sample flags, blanks, each
Allegheny file against its national file, per-fuel columns adding up to their totals, the
net capital cost and NPV identities, and the adopter flag. All must pass. Compare the
three `source_data/` CSVs with the model's input files and with the two CSVs as sent in
`s9_inputs/` (hash); list the degree-day factors file as one Chris did not receive, and
say which of the three has a tab in the workbook. Save
`s9_outputs/s9_validation_report.md`. A failed check is a stop, not a fix.

**Task 4 -- README and Data Dictionary sheets.** Write
`s9_outputs/TARE_Export_README_and_Dictionary.xlsx` with two sheets, `README` and
`Data Dictionary`, laid out exactly as the workbook's (Task 1).

- *Data Dictionary:* one row for every column of the new main files (and the detailed
  copy's per-fuel pattern), same five columns and group labels as before, new groups
  where needed. Mark a column that exists for MP5 only, or for MP3 and MP4 only, in its
  description. Units and sources as in `docs/tare_tepper_exports_data_dictionary.md`.
- *README:* keep the old sheet's sections and update every statement that no longer
  holds: the weight per release; MP3 and MP4 share homes but MP5 does not; the tabs Chris
  needs for MP3, MP4 and MP5; the practice exercise (an MP5 home has two heating fuels
  after the retrofit, so price each fuel, using the detailed copy; the net capital cost
  has a furnace term). Add a section, "What changed since the 19 August 2026 data", one
  short entry per row of the plan's section 6, each with what it moved, using Task 2's
  numbers. Say how the heating credit is priced, and that a home with a boiler has a
  credit about twice a furnace's. Say that the household files now hold only homes with
  a result (the old national file held every occupied single-family home, with blanks),
  and that the county file's `home_count` and `adoption_rate_pct` are now measured over
  those homes. Plain language; the reader works in Excel and has not seen the code.

Verify: every column of every new main file has exactly one dictionary row (patterns
counted), and no dictionary row names a column that does not exist. Report where the
file is; the researcher reviews it at the end.

**Task 5 -- Facts for the email, and the records.** Write `s9_outputs/s9_email_facts.md`:
the files Chris gets (names, rows, run stamps); the five to eight changes that matter
most to him, each with one number from Task 2; what to replace in his workbook; and the
caveats he must not miss (blank-as-zero no longer applies because the files hold the
sample only; a county adoption rate in the new files is not comparable with one in the
old files, which was measured over every home; never match MP5 rows to MP3 rows; do not
mix files made before and after 9 Oct 2026). At most one page. Then, in the repo:

- add a Stage 2 entry to section 7 of the plan, and correct section 5's statement about
  where the 19 Aug Tepper files are;
- replace `cmu_tare_model/docs/DF_STAGE2_SESSION_PROMPT.md` with this prompt
  (`~/tare_port/s9_stage2_kickoff_prompt.md`), keeping the repo file's line endings: the
  repo copy is the version of 8 Oct 2026.

This is the commit point. Stop with the commands and the commit message.

## Start

Do Tasks 1 to 5 in order, under auto-approve. Stop at the commit point, or earlier at any
stop that "Auto-approve" lists.

## Deferred (do NOT start now)

- Grid impact for ResStock 2025.1, and a guard on its cells (plan Q9, D-10).
- The EUSS-to-ResStock rename (plan Q7, Q8), as one change.
- The 17 tests that assume 2022.1.1 when the default release reads 2025.1 (plan D-11).
- Pricing of electric furnaces and electric boilers, after asking NLR's REMDB researchers
  (CLAUDE.md Limitation 16).
- A review of DOE Program Notice 26-3.
