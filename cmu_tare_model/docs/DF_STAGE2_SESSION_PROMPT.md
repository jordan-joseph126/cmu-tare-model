# TARE model -- Stage 2: data validation and the data share with Chris -- session prompt

CLAUDE.md is the source of truth for conventions, file rules, reference values, coding
standards and every non-negotiable; it is auto-loaded, so this prompt adds only
session-specific state and the task list. Audit before editing, one diff per stop gate,
and never stage or commit, per CLAUDE.md.

## Context

**Priority: show Chris, with numbers he can check, what changed between the files he has
and the files he will get, and give him a README and a Data Dictionary for the new ones.**

Read `cmu_tare_model/docs/DF_COST_FIXES_AND_DATA_SHARE_PLAN.md` before Task 1: section 5
is this stage's plan, section 6 the list of changes since 19 Aug 2026, section 3 the
decisions (Q10 names the packages Chris receives, Q11 the shape of the comparison), and
section 7.1 the runs. Its decisions and values win over anything below that differs.

This session analyses and writes; it changes no model code and moves no value. Keep the
`s9_` scripts short and import the model's own helpers (column-name builders, the Tepper
column list, the sample flag) rather than re-deriving them. Every script follows CLAUDE.md
and the researcher's saved preferences (memory). Out of
scope: any fix to the model (a problem found is reported, not fixed), and the email
itself, which is drafted elsewhere from Task 5's file.

## Current state the tasks depend on

- **Stage 1 is done** only if section 7.1 of the plan lists its final national runs for
  both releases. If it does not, report that and stop.
- **The old data:** run `2026-08-19_20-56` (ResStock 2022.1.1, MP3 and MP4). The Tepper
  household and county files Chris received, national and Allegheny, are in
  `cmu_tare_model/output_results/tepper_export/`, named with the stamp
  `2026-08-19_20-56`; take the `source_data/` CSVs of the same export from there too.
  The new files sit in the same folder under their own run stamps, so select every file
  by stamp, never by "latest". The old household files hold 155 columns and every rdu that passed the
  heating check: 1,356 rdu in Allegheny, 209 of them with no AC and a blank cooling
  replacement cost. They have no `include_sample` column. Nationally that run's no-rebate
  adoption (`heatingLCC_coolingLCC_unsub`) was 27.1760% (MP3) and 18.0984% (MP4) of
  260,211 rdu.
- **Chris's workbook:** `~/tare_port/s9_inputs/TARE_Export_Tepper_Cashflow_Model_JMJ_DRAFT.xlsx`.
  Its `README` sheet is one column of text; its `Data Dictionary` sheet has five columns,
  `column_name`, `group`, `description`, `units`, `source`, one row per column or
  column pattern, with `mp{mp}` and `{year}` placeholders. Read both sheets; do not
  change this file.
- **The new data:** Stage 1's final runs and their Tepper files (main and detailed;
  national and Allegheny) for every package Q10 names. MP5 files have 182 columns (332 in
  the detailed copy); MP3 and MP4 files 167 (317).
- **Two releases, two kinds of comparison.** Old MP3 and new MP3 are the same ResStock
  2022.1.1 homes: match them by `bldg_id`. New MP5 is ResStock 2025.1, a different set of
  homes with the same `bldg_id` values: compare it with MP3 by group only (all homes,
  baseline fuel, system type), never row by row. One rdu is 242.131013 homes in 2022.1.1
  and 253.90367 in 2025.1; label every count rdu or homes.
- **Populations differ.** The old files include homes without an AC; the new files hold
  the study sample only. Show the old files both as shipped and limited to rdu with a
  central or room AC, and say which is which in every table.
- Scratch files go in `~/tare_port` with the prefix `s9_`. Outputs go in
  `~/tare_port/s9_outputs/`.

## Tasks (in order; stop at each gate)

**Task 1 -- Audit (no edits).** Confirm Stage 1 is done (section 7.1 of the plan). Locate
every old and new file named above, and the workbook. For each file give rows, columns
and rdu. Give the header differences between the old MP3 file and the new MP3 and MP5
files: added, removed, renamed. Read the workbook's two sheets and report their exact
layout (rows, columns, header row, text styling). Report and stop.

**Task 2 -- Comparison.** In an `s9_` script, for national and for Allegheny, in the
shape Q11 decides (recommended: old MP3, then new MP3, then new MP5):

1. No-rebate adoption of the shipped case, all homes and by baseline heating fuel, with
   rdu, homes and the share of the sample in each fuel.
2. The parts of the adoption decision, mean and median per home, all homes and by
   baseline fuel: credit for the old heating system; credit for the old AC; heat pump
   installed cost; backup furnace installed cost (MP5 only); net capital cost; discounted
   heating and cooling savings; NPV.
3. Old MP3 to new MP3, matched by `bldg_id`: rdu that start and stop adopting, by fuel,
   and the part that moved most for them. Tie each move to a row of the plan's section 6.

Save each table as CSV and one summary as `s9_outputs/s9_comparison_summary.md`. Report
the headline tables in the chat and stop.

**Task 3 -- Validation of the new files.** Run the checks of sections 13 and 14.4 of
`docs/tare_tepper_exports_data_dictionary.md` on every new file: row and column counts,
the three sample flags, blanks, each Allegheny file against its national file, per-fuel
columns adding up to their totals, the net capital cost and NPV identities, and the
adopter flag. Compare the new `source_data/` CSVs with the copies sent on 19 Aug
(hash; if different, list what changed; a file Chris did not receive is listed as new). Save `s9_outputs/s9_validation_report.md`.
Report and stop. A failed check is a stop, not a fix.

**Task 4 -- README and Data Dictionary sheets.** Write
`s9_outputs/TARE_Export_README_and_Dictionary.xlsx` with two sheets, `README` and
`Data Dictionary`, laid out exactly as the workbook's (Task 1).

- *Data Dictionary:* one row for every column of the new main files (and the detailed
  copy's per-fuel pattern), same five columns and group labels as before, new groups
  where needed. Mark a column that exists for MP5 only, or for MP3 and MP4 only, in its
  description. Units and sources as in `docs/tare_tepper_exports_data_dictionary.md`.
- *README:* keep the old sheet's sections and update every statement that no longer
  holds: the weight per release; MP3 and MP4 share homes but MP5 does not; the tabs Chris
  needs for the packages of Q10; the practice exercise (an MP5 home has two heating fuels
  after the retrofit, so price each fuel, using the detailed copy; the net capital cost
  has a furnace term). Add a section, "What changed since the 19 August 2026 data", one
  short entry per row of the plan's section 6, each with what it moved, using Task 2's
  numbers. Plain language; the reader works in Excel and has not seen the code.

Verify: every column of every new main file has exactly one dictionary row (patterns
counted), and no dictionary row names a column that does not exist. Stop for the
researcher's review.

**Task 5 -- Facts for the email.** Write `s9_outputs/s9_email_facts.md`: the files Chris
gets (names, rows, run timestamps); the five to eight changes that matter most to him,
each with one number from Task 2; what to replace in his workbook; and the caveats he
must not miss (blank-as-zero no longer applies because the files hold the sample only;
never match MP5 rows to MP3 rows). At most one page. Then a gated diff adding a Stage 2
entry to section 7 of the plan. Stop.

## Start

Do Task 1 only. Report the audit and stop at the first gate.
