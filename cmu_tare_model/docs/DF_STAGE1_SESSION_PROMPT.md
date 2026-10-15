# TARE model -- Stage 1: cleanup and cost fixes -- session prompt

CLAUDE.md is the source of truth for conventions, file rules, reference values, coding
standards and every non-negotiable; it is auto-loaded, so this prompt adds only
session-specific state and the task list. Audit before editing, one diff per stop gate,
and never stage or commit, per CLAUDE.md.

## Context

**Priority: price every old heating system with the closest REMDB row, starting with
boilers, and leave the model ready to make the files for Chris.**

Read, before Task 1:

- `cmu_tare_model/docs/DF_COST_FIXES_AND_DATA_SHARE_PLAN.md`: this stage's plan. Section 1
  holds the results of record; section 3 the decisions, one per question (Q1 to Q11);
  section 4 the order of Stage 1. Its decisions and values win over anything below that
  differs.
- Part 2 of `cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md`, and section 3.1 and
  Session 7 of `cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md`. The item
  numbers below (1, 9, 10, 13 to 18) are Session 7's.

**Nearly all of the architecture is already in place.** Both releases run end to end,
and every REMDB row this stage needs is already in the cost table. This session changes a
few assumptions, not the design: extend the existing replacement row mapping and
efficiency floors, add one small guard, move one default. Expect short diffs. Do not add
modules, a new pricing path, or refactors of code a task does not need; if a task seems to
need more than a small edit to existing code, stop and explain why before writing it.
Every line follows CLAUDE.md and the researcher's saved preferences (memory).

In scope: Session 7 items 1, 9, 10, 13 to 18, and the records they need. Out of scope:
Session 7 items 2 to 8, 11 and 12 (grid impact for 2025.1 beyond the guard, and the
EUSS-to-ResStock rename), DOE Program Notice 26-3, and any rebate rule.

## Current state the tasks depend on

- **Branch** `resstock2025-dual-fuel-codebase-update`. The working tree must be clean at
  the start; if it is not, report what is there and stop.
- **Tests:** the command of decision (T); 399 passed, 1 skipped at the start.
- **Default release is 2022.1.1** (`constants.py`) and stays so until Task 7.
- **Reference runs:** `2026-10-07_19-22` (2022.1.1, MP3 and MP4: 221,205 rdu) and
  `2026-10-07_19-14` (2025.1, MP5: 161,983 rdu). Every national comparison in Tasks 3 to 6
  starts from them. Pennsylvania reference runs: `2026-10-08_01-51` (2022.1.1) and
  `2026-10-08_01-49` (2025.1).
- **Where the costs are priced:** `utils/remdb_v4_installed_cost_utils.py`. The credit for
  an old heating system uses `furnaces_gas_furnace` for every fossil system and
  `electric_baseboard_default` for every electric one, at the old system's own size
  (`base_size_heating_system_primary_k_btu_h`) and efficiency. Efficiency floors are in
  `EFFICIENCY_FLOORS_PM2` (`constants.py`). The rows `boiler_gas_non_condensing`,
  `boiler_gas_condensing` and `boiler_oil` exist in
  `data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv` and are never selected. That
  file is an exact cut of NREL's REMDB 2024 table (checked 8 Oct 2026, plan Q12), which
  has no electric furnace or electric boiler row: do not look for one.
- **Credits do not depend on the package.** MP3 and MP4 must keep identical heating and
  cooling replacement-cost columns on every rdu after every change. A mismatch is a stop.
- **A value-moving change moves only what it names.** For the boiler fix: the heating
  replacement credit of boiler rdu, and the columns computed from it (net capital cost,
  rebates where they depend on it, NPV and adopter flags of the `heatingLCC_*` scopes).
  The `heatingSavings_coolingLCC` scope, every bill and energy column, and every
  non-boiler rdu must not move. Do not compare MP3 and MP5 home by home: the releases
  hold different homes.
- **Never run a 2025.1 run without `--skip-grid-impact`,** except the one guarded check in
  Task 2. On 8 Oct 2026 an unguarded run sent 2025.1 building ids to the 2022.1.1 AWS
  table and held about 76 GB of memory.
- Scratch files go in `~/tare_port` with the prefix `s8_`.

## Tasks (in order; stop at each gate)

**Task 1 -- Audit (no edits).** Confirm the branch, HEAD, a clean tree and the test count.
Read sections 3 and 4 of the plan and list each question with its decision, marking every
one still OPEN. Locate, by file and line: the replacement row-id mapping; the pm2 floor
step and `EFFICIENCY_FLOORS_PM2`; the grid impact cells of `tare_model_main_v3_0.ipynb`
(cells 32 and 33) and the first one that queries AWS; the tests that take 2022.1.1 for
granted when `TARE_RESSTOCK_RELEASE` is unset (item 10). Report and stop.

**Task 2 -- Grid impact guard (item 1).** Make the grid impact cells stop with a clear
message on a release the analysis does not support, before any query, through a small
tested function in a module that the cell calls (cell change through the
notebook-cell-edit skill). Verify: the new tests pass; Pennsylvania runs on both releases
are identical to the Pennsylvania reference runs; then one 2025.1
Pennsylvania run without `--skip-grid-impact` stops at the guard with its message and no
AWS query in the log. Moves no value. Own commit. Stop.

**Task 3 -- Measure the decided cost changes (no edits).** In an `s8_` script outside the
repo, on the reference runs' saved national results, compute for each choice decided in
Q2, Q3 and Q3b of the plan: the rdu it touches by release, package, fuel and system type; the mean and
median credit before and after; and the no-rebate adoption of each scope before and
after, from the NPV identity (new NPV = old NPV + change in credit). Rebated cases are
approximate where the rebate depends on cost; say which. Report one table per question
and stop. The researcher confirms before Task 4.

The rule for Tasks 4 and 5 is section 2.1 of the plan: match the old system's equipment
type first, then its fuel; one row per old system as its table gives.

**Task 4 -- Boiler fix (item 13), as Q2 decides.** In the existing replacement row
mapping, select the boiler rows for old boilers, and set their floors where the existing
floors are set; add tests for the row choice and the floors. Reuse the existing pricing
function unchanged. Verify: tests pass; the
MP3-MP4 credit identity holds; Pennsylvania runs differ from the reference only on boiler
rdu and only in the columns named above; then national runs on both releases, with the
change measured against Task 3's figures. Own commit, `VALUE-MOVING` on both releases.
Stop.

**Task 5 -- Electric furnaces and electric boilers (item 14), as Q3 and Q3b decide.**
An old electric furnace is priced on `furnaces_gas_furnace`, and an old electric boiler on
`boiler_gas_non_condensing`, each at AFUE 0.80 whatever its published efficiency;
electric baseboard stays on `electric_baseboard_default`. No price adjustment. Add tests
for the row choice and the fixed AFUE. Verify as in Task 4, on electric furnace and
electric boiler rdu. Own commit, `VALUE-MOVING` on both releases. Stop.

**Task 6 -- Nothing to change.** Q1 keeps the study sample as it is (item 9); Q4, Q5 and
Q6 change no code (items 16, 17, 18). Do not touch the sample funnel, the range handling,
the efficiency floors other than the boiler and electric choices above, or the heat pump
rows. Their records are in Task 8.

**Task 7 -- Default release to 2025.1 (item 10).** Change the line and its comment in
`constants.py`; make each test that assumed 2022.1.1 name the release it needs; update
what says "unset means 2022.1.1" (CLAUDE.md, the runner's notes). Verify: tests pass with
the variable unset and with it set to each release; Pennsylvania runs on both releases
(2022.1.1 now through `--release 2022.1.1`) identical to the runs of Tasks 4 and 5. Moves
no value. Own commit. Stop.

**Task 8 -- Records.** One gated diff each: new REFERENCE_VALUES rows from the national
runs of Tasks 4 and 5, marked as superseding, old rows kept; CLAUDE.md: Limitation 9
gains the existing out-of-range rule and the counts of item 16 (Q4); new limitations for
item 15 (propane and fuel oil furnaces on the gas furnace row), for Q3 and Q3b (electric
furnaces and boilers on gas rows, which price venting and a gas line they do not have, so
their credits are likely somewhat high), for Q5 (no room AC floor; CEER against EER) and
for Q6 (no panel upgrade assumed; two heat pump rows unused); and the "Last updated"
block; a `docs/SESSION_LOG.md` entry; the Tepper data dictionary
(`docs/tare_tepper_exports_data_dictionary.md`), where a section names the old pricing or
quotes a value that moved; and sections 6 and 7 of the plan (what changed, the runs,
the results). Stop.

**Task 9 -- Final national runs.** One national run per release on the committed code,
with the Tepper exports, national and Allegheny, for every package Q10 names. Verify:
each run passes the runner's checks; the Tepper checks of sections 13 and 14.4 of the
data dictionary hold. Record the timestamps in section 7.1 of the plan (gated diff).
Report and stop.

## Start

Do Task 1 only. Report the audit and stop at the first gate.

## Deferred (do NOT start now)

- Stage 2, the data share with Chris: `DF_STAGE2_SESSION_PROMPT.md`, after Task 9.
- Grid impact for 2025.1 beyond the guard (items 2 to 5), after Q9.
- The EUSS-to-ResStock rename (items 6 to 8), after Q7 and Q8, as one change.
- Items 11 and 12, only if the researcher asks.
