# Local port session plan -- cloud ResStock 2025.1 dual fuel (package 5) work

Written by Session 0 on 6 Oct 2026, from that day's audit. Updated the same day:
decisions (i) and (k) taken, the other pending decisions are asked at the start of
their session, and each start line names the test command. Updated again on 6 Oct
2026, during Session 1: commit messages use the researcher's bracketed-bullet format
(decision (C)), and the remaining patches are ported as written and followed by one
cleanup commit (decision (R), Session 5b). Updated on 7 Oct 2026, during Session 2:
the comment on the two rating factors is kept as written and reworded in Session
5b (decision (i), item 4 of Session 5b). Updated again on 7 Oct 2026, after
Session 2: decision (j) is taken, and Session 3 runs unattended (decision (U)).
Updated on 7 Oct 2026, after Session 3: the second part of decision (k) is taken,
and Session 4 runs unattended (decision (U4)). Updated again on 7 Oct 2026, after
Session 4: decision (l) is taken, and Session 5 runs unattended with its four
commits made at the end (decision (U5)). Updated on 7 Oct 2026, during and after
Session 5b: that session gains items 5 to 9 (decision (5b)), and a second cleanup
session, 5c, follows it (decision (5c)). Updated on 7 Oct 2026, at the start of
Session 5c: its opening answers are recorded (decision (5c)). Updated on 7 Oct
2026, during Session 6: decision (m) is taken, and Session 6 gains two CLAUDE.md
units, 5a and 5b (decision (m)). Updated on 8 Oct 2026, after Session 6's records
were committed: decision (b) is taken, and this file is tracked from commit
`5681706` on (decision (b)). Updated again on 8 Oct 2026, after the port: Session
7 is added for the 2025.1 grid impact work, the ResStock naming and the cleanup
found after the port (decision (S7)).

- `LOCAL_KICKOFF_PROMPT.md` and CLAUDE.md win wherever this plan differs from them.
  This plan relaxes neither.
- Any edit to this file is a gated diff. The file was untracked during the port and
  is tracked from commit `5681706` on (decision (b)); the researcher makes every
  commit of it.
- Cloud numbers quoted here are expected values for checking, never reference values.
- "C1" to "C7" and "2.1" to "2.9" are the kickoff prompt's commands and protocol steps.
  "Section 11.x" is in `LOCAL_PORT_INSTRUCTIONS.md`.

## 1. Audit findings

### 1.1 Repo

| Item | Finding |
|---|---|
| Branch | `resstock2025-dual-fuel-codebase-update` |
| HEAD | `b17d7af`, the commit the patches were made against |
| Commits since `b17d7af` | None |
| `git status --short` | Only `?? cmu_tare_model/docs/cloud_run/` |
| `git diff --stat` | Empty |
| `core.autocrlf` | `true`, from `C:/Program Files/Git/etc/gitconfig` |
| `.gitattributes` | None |
| Bundle | 15 patches, 4 changes files, 7 `.md` files; all 15 SHA-256 values match |
| Safety copy (made by the researcher, 6 Oct) | Branch `backup/pre-cloud-port-2026-10-06` and annotated tag `pre-cloud-port-2026-10-06`, both at `b17d7af` |

### 1.2 Line endings

| Files | Working tree | Committed copy | `core.autocrlf` for the apply |
|---|---|---|---|
| 19 existing `.py` files, `cmu_tare_model/constants.py` among them | CRLF | LF | `true` |
| `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | LF | LF | `false` |
| The 3 notebooks | LF | LF | None: notebook-cell-edit skill only |
| The 5 new files | Follow `constants.py`: CRLF | n/a | `true` |

- No file has mixed endings. No committed copy is CRLF or mixed. So no unit is a hand
  port.
- `cmu_tare_model/adoption_kpis/data_loading.py` has no final newline, as patch 05
  expects.

### 1.3 Environment and data

| Item | Here | Cloud |
|---|---|---|
| Python | 3.11.13, `...\anaconda3\envs\cmu-tare-model\python.exe` | 3.11.17 |
| pandas / numpy / pyarrow | 2.1.4 / 1.26.4 / 23.0.1 | Same |
| IPython | 9.1.0 | 9.17.1 |
| nbformat / psutil / pytest | 5.10.4 / 5.9.0 / 9.0.2 | 5.11.1 / 7.2.2 / 9.1.1 |
| `TARE_RESSTOCK_RELEASE` | Unset | |
| 2025.1 data | `data/resstock_2025_1/upgrade0.parquet` and `upgrade5.parquet` present | |
| 2022.1.1 data | Baseline, `upgrade03` and `upgrade04` CSVs present | |
| Run `2026-10-05_21-33` | Its 10 output files are on disk | |
| Tepper export CSVs | None on disk | |
| Machine | 31.1 GB RAM, 14 logical CPUs, 161 GB free disk | 16 GB, 4 CPUs |

`~/tare_port/env.sh` holds two lines more than the kickoff prompt lists. VS Code's
environment puts base Anaconda ahead of the env on `PATH`, so `conda activate` alone
left `python` as base Anaconda. The researcher approved the two lines on 6 Oct:

```bash
source /c/Users/jorda/AppData/Local/anaconda3/etc/profile.d/conda.sh
conda activate cmu-tare-model
unset TARE_RESSTOCK_RELEASE
REPO="/c/Users/jorda/Desktop/Projects/cmu-tare-model"
ENV_ROOT=/c/Users/jorda/AppData/Local/anaconda3/envs/cmu-tare-model
export PATH="$ENV_ROOT:$ENV_ROOT/Library/mingw-w64/bin:$ENV_ROOT/Library/usr/bin:$ENV_ROOT/Library/bin:$ENV_ROOT/Scripts:$PATH"
```

### 1.4 Dry run

- **Throwaway copy** (`~/tare_port/audit_copy`, outside any repo): all 41 `.py`
  sections checked and applied in patch order with no offset and no warning. Each kept
  its endings and changed exactly the patch's line counts. Changes files 01 to 04
  checked and applied, each "writes back byte for byte: yes"; the changed notebook lines
  equal patches 04, 09 plus 11, and 14, line for line. Every patched `.py` file compiles.
- **Real repo** (read-only): the 20 existing `.py` paths pass `--check` with the first
  patch that touches them; the 5 new paths do not exist yet; the skill's `check` passes
  for changes files 01, 02 and 04. The other 17 sections can only be checked once an
  earlier patch is in.
- **Totals:** 45 sections, 114 hunks (103 in `.py` sections, 11 in notebooks), as the
  kickoff prompt says.
- **Added lines:** none is non-ASCII; none in a `.py` file is over 88 characters. Sixteen
  cite a decision ID in a comment (patches 05: 1, 06: 2, 07: 2, 08: 4, 10: 5, 11: 1,
  12: 1). Step 2.2.3 notes them at each unit; they change only if the researcher asks.

### 1.5 Test baseline

| Command | Result at `b17d7af` |
|---|---|
| C6 as written | 9 failed, 358 passed, 1 skipped (exit 1) |
| C6 with `--ignore=archived_files` | 352 passed, 1 skipped (exit 0): the cloud's baseline |

The 9 failures and 6 of the passes are in `archived_files/`, a git-ignored folder at the
repo root that the cloud copy did not have. All 9 are in
`archived_files/test_hdd_consumption_utils.py`.

### 1.6 What the audit could not close

- **The runner on Windows.** Never run here; IPython and psutil are older than the
  cloud's; it passes a Windows path to `%run`. Session 1's "before" run is the first
  test. If it fails, the cause is the runner or Windows, not a patch (kickoff 2.6).
- **No Tepper CSV on disk.** So only a fresh pre-port run can give a national Tepper
  file to compare; see decision (g).
- **Commits after run `2026-10-05_21-33`.** Five (`2fabbce`, `ed56dc3`, `cd5ca61`,
  `4a8fac3`, `b17d7af`); each message says no modeled value moves. Not re-checked by the
  audit. Session 1's national "before" run checks it (Session 1, step B).
- **Grid impact.** Off in every run of this port, so the port never exercises the
  grid-impact cells of the main notebook.
- **Memory.** A 2022.1.1 national run has never been timed or measured through the
  runner. Close other programs before a national run.
- **Decision IDs.** The cloud session prompt that defines R1 to R9 and G1 to G10 is not
  in the repo. `NEXT_STEPS_2026-09-19_DualFuel_ResStock2025_1.md` (D8) is.

## 2. Decisions

### 2.1 Taken (researcher, 6 Oct 2026)

- (a) Port directly onto `resstock2025-dual-fuel-codebase-update`. The safety copy is
  the branch and tag in 1.1.
- (b) `cmu_tare_model/docs/cloud_run/` stays untracked until Session 6. Second part
  (researcher, 8 Oct 2026, after Session 6's records: "Yes" to the option
  recommended on 7 Oct 2026): the eight documents are committed, in a commit of
  their own (`5681706`), and `patches/` (15 files) and `notebook_changes/` (4
  changes files) are moved out of the repo, to `~/tare_port/cloud_run_patches/`.
  Commits `6e5c549` to `c53a9bc` hold the same changes as those files. The
  researcher made the move and the commit on 8 Oct 2026, before the session that
  records the decision here, and confirmed the choice in that session's chat.
- (c) Yes: an approved file is applied with C3's `git apply --include=<file>`.
- (d) One approval per file: 45 in all. This loosens CLAUDE.md's "across functions"
  rule for this port only; one approval never covers two files. Everything is printed
  in the chat, long changes included. `hunk by hunk` and `file by file` switch as the
  kickoff prompt says. Auto-approve is off unless the researcher turns it on in the
  chat with `auto-approve <scope>`.
- (e) One commit per patch, except 08 with 09 in one commit and 10 with 11 in one
  commit, so that every commit runs. 13 commits in all.
- (f) A Pennsylvania run each session. No MN or FL runs.
- (g) A fresh national 2022.1.1 run in Session 1, before any model patch, with grid
  impact off. Session 6's national run is compared with it.
- (h) Checked, nothing to decide: both 2025.1 files are present and pyarrow 23.0.1 is
  installed.
- (T) Test command: C6 with --ignore=archived_files added (researcher's choice, 6 Oct 2026). Session 0 count: 352 passed, 1 skipped, exit 0.
- (C) Commit messages: the researcher's bracketed-bullet format, for every commit of
  the port from patch 05 on (researcher's choice, 6 Oct 2026, during Session 1). The
  format is set out in 3.1.
- (R) Review of the remaining patches (researcher's choice, 6 Oct 2026, during
  Session 1): patches 06 to 15 are ported as written, all of them, so that every
  file stays checkable against its patch. One cleanup commit that changes no
  behavior follows Session 5 (Session 5b). It makes three things smaller: the two
  small new helper files of patches 06 and 07, and the new test file of patch 08.
- (i) P1: keep SEER1 = SEER2 / 0.95 and HSPF1 = HSPF2 / 0.85. The researcher's reason:
  a reasonable and standard assumption for the vast majority of residential
  installations, ducted split systems above all, which are what dual fuel is about.
  So Session 2's diffs and expected numbers stand as the cloud measured them. Patch
  07's comment in `constants.py` still calls the two factors "PROVISIONAL" and not
  checked against DOE Appendix M1; it goes in as written unless the researcher asks
  for a change when that unit is shown (Session 2, unit 5). Shown in Session 2 (7 Oct
  2026): the researcher kept it as written, so the file stays checkable against the
  patch. The comment is reworded in the cleanup commit (Session 5b, item 4).
- (k) O-1: yes, June 2026 equal to 2024 for the dual-fuel package is intended. The
  researcher's reason: the heat pump qualifies for the rebate under either guidance,
  and the furnace gets no rebate of its own. In the 2022.1.1 results adoption falls
  from 2024 to June 2026 only because fuel switching is not allowed, while electric
  homes are the same in both; once a dual-fuel retrofit lets fossil-baseline homes be
  funded, they are the same in both too. So patch 10 and changes file 03 go in as
  written. Second part (researcher, 7 Oct 2026, after Session 3): the Program
  Notice 26-3 comment that patch 10 adds to `constants.py` is kept as written. It
  says only that DOE has released Program Notice 26-3, that the researcher has not
  yet reviewed how it differs from 26-1 and 26-2, and that no rule was changed
  because of it. Kept as written, `constants.py` stays checkable against the
  patch. If the researcher wants it reworded later, that is an item for the
  cleanup commit (Session 5b).
- (j) The furnace decisions: all four accepted as patch 08 writes them (researcher,
  7 Oct 2026, after Session 2). P5: the furnace cost is not in the rebate base.
  R1: the furnace is priced at its own backup size. D-S8: only a gas backup is
  priced, and a propane or oil backup stops the run. P8 / D-S7: the column is
  `mp5_heating_backupFurnace_installed_cost_v4MID`. On P5 the researcher was told
  first that the rebate cap does not always bind: in Session 2's Pennsylvania run
  the 2024 rebate moved with the cost for 1,716 of 6,469 sample rdu, so leaving
  the furnace out of the base makes those homes' rebates smaller than with it in.
- (U) Session 3 runs unattended (researcher's choice, 7 Oct 2026). Every diff is
  pre-approved by `auto-approve this session` in the start message. The session
  does not stop at the commit point, makes its Pennsylvania runs on the
  uncommitted working tree, and ends with one stop that gives the `git add`
  commands and the commit message. It stops earlier only when something is
  unexpected and wrong. A usage-limit pause, a compacted conversation, and the
  blank-line count case of Session 1 are not stops in this session; the start
  message says how each is handled. The kickoff prompt wins over this plan, so
  the start message says all of this in the researcher's own words. Session 3
  only: a later session is unattended only if its own start message says so.
- (U4) Session 4 runs unattended too, on the terms of (U) (researcher's choice,
  7 Oct 2026, after Session 3, which ran that way with no early stop). One more
  thing is not a stop in Session 4: in the 2022.1.1 log the MP3 and MP4
  `[PASS] ... June 2026 fuel gate holds` lines gain ", non-participating states
  $0.", as the session's checks expect. Those two lines must be the only marked
  lines that differ, and the 22 output files must still be the same bytes as
  "before". At its one stop the session also saves the commit message as plain
  text in `~/tare_port` and gives the `git commit -F` command for it. Session 4
  only: a later session is unattended only if its own start message says so.
- (l) The names, columns and title of Session 5: all three accepted as written
  (researcher, 7 Oct 2026, after Session 4: "Accept all three as written."). P7:
  the 2025.1 peak electric demand columns are named
  `..._peak_electricity_winter_kw` and `..._summer_kw` (patch 12); the 2022.1.1
  names stay. D-S11: the Tepper household files for package 5 gain 13 columns
  (patch 13): seven dual-fuel ratings, the backup furnace's size and its cost,
  and four electric panel columns. D-S12: the package 5 title is "Dual-fuel heat
  pump with gas backup furnace" (changes file 04). The researcher was told two
  things first. Six of the seven rating columns hold one value on every
  Pennsylvania sample home (not checked nationally). And
  `cmu_tare_model/docs/tare_tepper_exports_data_dictionary.md` has no row for
  package 5, the 13 columns or the winter and summer names; no patch and no
  planned Session 6 unit adds them.
- (U5) Session 5 runs unattended too, on the terms of (U) (researcher's choice,
  7 Oct 2026, after Session 4: "Session 5 should also be auto-approved and setup
  like session 3/4"). One thing is different. Session 5 holds four patches and
  four commits, and three files are changed by two of its patches:
  `export_tepper_csv.py` and `test_column_names.py` by 12 and 13, and
  `process_euss_data.py` by 12 and 15. No commit is made between patches. So for
  units 5, 6 and 8 the session saves a copy of the file in `~/tare_port` just
  before the unit, checks the counts against that copy and not against the last
  commit, and confirms that the file equals the committed text with both patches
  applied in the scratch folder. Tried in the scratch folder on 7 Oct 2026
  (`s5_rehearsal.sh`): each second patch applies on top of the first with its
  own counts (39 / 2, 20 / 0, 37 / 7), and every file Session 5 changes ends
  equal to the audit copy's. Decision (e) stands, one commit for each patch: at
  its one stop the session gives the commands for four commits, in order. For
  12 and 13 that is `git apply --cached` with the patch's file, because patch 13
  removes a line that patch 12 adds in `export_tepper_csv.py` and whole files
  cannot tell the two apart; for 14 and 15 it is `git add` with whole files. The
  researcher runs every one of them. The session also saves four commit
  messages in `~/tare_port`, each with its `git commit -F` command. The two
  2022.1.1 log lines that changed in Session 4 are not a stop. Session 5 only: a
  later session is unattended only if its own start message says so.
- (5b) Session 5b's choices and added items (researcher, 7 Oct 2026, from the
  session's inventory). Item 2: `is_dual_fuel_package` moves to
  `utils/calculation_utils.py`, which reads no data file when imported and which
  four of the seven modules that import the function already import from.
  `utils/resstock_schema.py`, the other candidate, reads the column map CSV when
  imported. Item 4: yes, the SEER2 / HSPF2 block also moves below
  `EFFICIENCY_FLOORS_PM2`. Item 3: the furnace cost row and the two fixtures that
  both destination test files need go in `cmu_tare_model/tests/conftest.py`; the
  furnace table fixture is renamed `backup_furnace_remdb_costs`, because
  `test_remdb_v4_installed_cost_utils.py` already has a fixture named
  `remdb_v4_costs`. Added by the researcher in the same session ("Also please
  incorporate the cleanup items mentioned in the Session 5 notes"): items 5 to 8
  of Session 5b, and the Tepper data dictionary rows, which become unit 7 of
  Session 6's documentation edits because they need the local national run.
- (5c) A second cleanup session before Session 6 (researcher, 7 Oct 2026, after
  Session 5b closed: "Please update the plan and kickoff prompt to also address
  these cleanup tasks", with the five things Session 5b's report listed as left
  as they are). Four of them make up Session 5c: decision IDs and
  planning-document names in comments, docstrings for the moved furnace tests, a
  test for the four Tepper panel columns, and the stale pointers in
  `grid_impact/build_parcel_frame.py`. The fifth, the Tepper data dictionary, is
  already unit 7 of Session 6's documentation edits. Session 5b's report named
  three test files that hold a decision ID; a search of the code on 7 Oct 2026
  finds twelve lines in eleven files, and Session 5c's table lists them all.
  Answers at the start of Session 5c (researcher, 7 Oct 2026). "Decide first":
  `build_parcel_frame` is kept ("Keep it"), so item 4 rewords its stale
  pointers and removes nothing. Item 1, from the session's inventory: the seven
  pointers to `docs/SESSION_CHANGELOG_2026-08-20.md` are left as they are,
  because they point to a record that exists. The session's search also found
  five decision-ID lines that item 1's list did not hold, all older than the
  port ("D8" once, "D3" four times). Item 1 as written covers them, so they are
  taken ("Yes for A"), and item 1's row now names them. Labels with no decision
  ID ("Phase 3", "Session 3", "P0.2": nine lines) are taken too, added by the
  researcher in the chat ("I think these can be removed as well"). Item 1's row
  names them, and each is reviewed in its file's diff. One of the nine is
  printed: "(Phase 3)" in the opening banner of the baseline notebook, so one
  line of every run's log changes. Added at the commit point (researcher,
  7 Oct 2026: "Please fix the stale notebook / step references"): item 5.
- (m) Session 6's records (researcher, 7 Oct 2026, after the two national runs
  and the final test run: "Yes to all 7"). Yes to each of the five parts the
  plan held: new REFERENCE_VALUES rows for 2025.1 package 5, from the local
  national run `2026-10-07_19-14`; and the four CLAUDE.md blocks (Program
  Notice 26-3 and dual fuel; `TARE_RESSTOCK_RELEASE` and the runner;
  Limitation 14; the winter and summer peak names). The blocks are adjusted to
  the decisions taken: no decision IDs, `is_dual_fuel_package` named in
  `utils/calculation_utils.py`, and in Limitation 14 the researcher's reason
  from decision (i) in place of "PROVISIONAL". Yes also to two parts the plan
  did not hold, which become units 5a and 5b of Session 6: CLAUDE.md's
  Limitation 2 says the dual-fuel package's June 2026 HEEHR treatment "is not
  yet set (Phase 7)", and is corrected to what decision (k) set; and
  CLAUDE.md's dated "Last updated" block gets an entry for the port. Each
  edit is still its own gated diff.
- (S7) A session after the port (researcher, 8 Oct 2026: "Add another session to
  the dual fuel implementation plan to include the dual fuel fix and other
  cleanup items"). "The dual fuel fix" is read as the grid impact analysis for
  the ResStock 2025.1 (dual fuel) run; the researcher asked for that TODO in the
  same message. On names (same message): "EUSS is the nickname for the 2022
  release only and for consistency it makes more sense to refer to them as
  ResStock with the release year/number", with every dataframe and variable
  holding `euss` renamed to `resstock`, as "a future commit and not to be messed
  with now". Session 7 holds both, and the smaller things found after the port.
  On costs (researcher, 8 Oct 2026, later): "The REMDB cost estimation should be
  using the closest fit available. In other words, if there are boiler data
  available it should be using the boiler data NOT furnace data"; Part 4 holds
  that fix and the other cost questions found the same day. On the default
  release (same message): it stays 2022.1.1, which the paper and a colleague's
  work use, and changes to 2025.1 only after the boiler fix (item 10).
  Nothing of Session 7 is started.

### 2.2 Pending

| Decision | Must be made before |
|---|---|
| (n) By hand: move the package 5 block into its own cells, update the notebook text the cloud left alone, regenerate the `*_EXPORT_*.py` snapshots | After the port |

The researcher asked on 6 Oct 2026 to be asked each of these at the start of its
session. The session asks before presenting unit 1 and waits. It then logs the answer
in `PORT_PROGRESS.md` and shows the one-line update to 2.1 as a gated diff (step
2.1.3).

If (i) or (k) is changed later, the affected diffs and expected numbers change: stop
and re-plan that session as a gated edit to this file.

## 3. Sessions

### 3.1 What holds in every session

- **Start.** One message: the session's start line, given in each session below.
  `/add-dir` is not available in the VS Code extension, and is not needed:
  `C:\Users\jorda\tare_port` is a working directory in every conversation of this
  project, through `permissions.additionalDirectories` in
  `.claude/settings.local.json`. Step 2.1.2 still confirms it. No answer needs adding
  to the start line: a session whose decision is still pending in 2.2 asks it first
  and waits.
- **Test command.** C6 with `--ignore=archived_files` added (decision (T)). The kickoff
  prompt's C6 has no such flag, and the kickoff prompt wins over this plan, so each
  start line says it in the researcher's own message.
- **Commit messages.** Decision (C). Each suggested message has:
  - a subject of one to three sentences: `Port cloud patch NN in <module or area>
    (<what the change does>).`, then a short sentence for any second change, then
    whether a modeled value moves: `No modeled value moves.`, or `VALUE-MOVING:` with
    what moves and for which release. The subject is not held to 72 characters;
  - a body of bullets, one for each piece of the change. Each bullet opens with a
    short tag in capitals and square brackets that names that piece (for example
    `[COLUMN NAMES FROM THE COLUMN MAP]`), and keeps the file and function names,
    counts and measured numbers;
  - a `[TESTS]` bullet with the new tests and the count from the test command above;
  - nothing after the body: no Co-Authored-By line, no session link, no other
    trailer.

  The "Suggested subject line" column in each session's table gives the topic only;
  the subject is written in the form above. The kickoff prompt's step 2.4.4 asks for a
  short subject and a short body, and the kickoff prompt wins over this plan, so each
  start line says it in the researcher's own message.
- **Units.** Each row of a unit table is one approval. Endings and `core.autocrlf` come
  from the audit; step 2.2.1 re-checks them every time. No unit is a hand port.
- **Runs.** Every run is C7 with `--skip-grid-impact`, in the background, one at a
  time, with no repo file changing while it runs. Logs go to `~/tare_port/logs/`, named
  `s<N>_<PA|national>_<2025|2022>.log`. Check the `[runner] release ...` line of every
  log.
- **2022.1.1 Pennsylvania.** At the end of every session it must be identical to
  Session 1's "before" run, compared as step 2.5.2 says. Any difference is a stop.
- **Test counts.** Each check is the Session 0 count plus the change below (section
  11.1). Skipped is always 1.

| After | Change | Passed, with `--ignore=archived_files` | Passed, C6 as written (plus 9 failed) |
|---|---:|---:|---:|
| Session 0 | +0 | 352 | 358 |
| 03 | +0 | 352 | 358 |
| 01 | +3 | 355 | 361 |
| 02 | +7 | 359 | 365 |
| 04 (changes 01) | +7 | 359 | 365 |
| 05 | +15 | 367 | 373 |
| 06 | +18 | 370 | 376 |
| 07 | +23 | 375 | 381 |
| 08, and 09 (changes 02) | +33 | 385 | 391 |
| 10, and 11 (changes 03) | +37 | 389 | 395 |
| 12 | +40 | 392 | 398 |
| 13, and 14 (changes 04) | +41 | 393 | 399 |
| 15 | +47 | 399 | 405 |
| Session 5b cleanup | +46 | 398 | 404 |
| Session 5c cleanup | +47 | 399 | 405 |

### Session 1 -- 2025.1 runs end to end, no value moves (03, 01, 02, changes 01, 05)

- **Decide first:** nothing. (T) is taken (2.1).
- **Modeled values:** none move, on either release.
- **Start line:**
  `Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 1 of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them. Stop at every gate. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records.`

| Unit | Patch | File | Hunks | + / - | Endings | `core.autocrlf` |
|---:|---|---|---:|---|---|---|
| 1 | 03 | `scripts/run_tare_notebooks.py` (new; printed whole) | 1 | 677 / 0 | New: CRLF | `true` |
| 2 | 01 | `cmu_tare_model/constants.py` | 3 | 17 / 5 | CRLF | `true` |
| 3 | 01 | `cmu_tare_model/tests/test_constants.py` | 1 | 48 / 0 | CRLF | `true` |
| 4 | 02 | `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | 4 | 78 / 2 | LF | `false` |
| 5 | 02 | `cmu_tare_model/tests/energy_consumption_and_metadata/test_process_euss_data.py` | 1 | 83 / 0 | CRLF | `true` |
| 6 | 04 | Changes file 01: `cmu_tare_model/model_scenarios/tare_run_simulation_v3_0.ipynb`, cell 15 | 1 cell | 113 / 0 | LF | Skill |
| 7 | 05 | `cmu_tare_model/adoption_kpis/data_loading.py` (the "\ No newline at end of file" marker is expected) | 7 | 159 / 99 | CRLF | `true` |
| 8 | 05 | `cmu_tare_model/adoption_kpis/demand.py` | 3 | 12 / 5 | CRLF | `true` |
| 9 | 05 | `cmu_tare_model/adoption_kpis/thermal_cop.py` | 2 | 4 / 2 | CRLF | `true` |
| 10 | 05 | `cmu_tare_model/tests/adoption_kpis/test_data_loading.py` (new; printed whole) | 1 | 181 / 0 | New: CRLF | `true` |

| Commit point | After unit | Check | Suggested subject line |
|---:|---:|---|---|
| 1 | 1 | Tests +0 | `Port cloud patch 03: add the notebook runner script` |
| 2 | 3 | Tests +3; with the variable unset, `python -c "from cmu_tare_model.constants import VALID_MENU_MPS; print(VALID_MENU_MPS)"` prints `[0, 3, 4]` | `Port cloud patch 01: read the ResStock release from TARE_RESSTOCK_RELEASE` |
| 3 | 5 | Tests +7 | `Port cloud patch 02: read only the needed columns of the 2025.1 files` |
| 4 | 6 | Tests +7 | `Port cloud patch 04: add the package 5 block to the simulation notebook` |
| 5 | 10 | Tests +15 | `Port cloud patch 05: KPI loaders and column names follow the release` |

**After commit point 1 and before unit 2, the two "before" runs.** Only the runner is
in, so both describe the model as it stood at `b17d7af`.

- **Step A -- Pennsylvania.** `--release 2022.1.1 --state PA --skip-grid-impact`, log
  `s1_PA_2022_before.log`. It is the first 2022.1.1 run of the runner anywhere. Copy its
  stamp, check count, and `[OK]` and `[PASS]` lines into `PORT_PROGRESS.md`. If it
  fails, report it and stop before unit 2.
- **Step B -- national, decision (g).** `--release 2022.1.1 --skip-grid-impact`, log
  `s1_national_2022_before.log`. Wait for `[runner] exit code N` before unit 2. Record
  its stamp, wall time and peak memory. Then compare its model output files with run
  `2026-10-05_21-33`'s 10 files, using the step 2.5.2 script. They should be identical,
  because no commit since that run is said to move a value (1.6). Report the result; a
  difference is a stop.
- Until patch 01 is in, `constants.py` does not read `TARE_RESSTOCK_RELEASE`, so do not
  try `--release 2025.1` before commit point 2: it would run 2022.1.1.

**End-of-session runs and checks**

- 2025.1 Pennsylvania, log `s1_PA_2025.log`: section 11.2, Session 1 column.
- 2022.1.1 Pennsylvania, log `s1_PA_2022.log`: identical to Step A, and main notebook
  cell 23 prints `home_count agrees in all N counties`. This is the first 2022.1.1 test
  of patch 05; a difference may be a real finding.
- Optional, only if the researcher asks: 2025.1 national, section 11.4, "After Session
  1".

### Session 2 -- dual-fuel ratings and SEER1 pricing (06, 07)

- **Decide first:** nothing left. (i) is taken (2.1): the factors are kept.
- **Modeled values:** 06 moves none. 07 moves 2025.1 only: heat-pump upgrade cost up
  about $754 per home, and with it net capital, NPV, adoption and cost-share rebates.
  No 2022.1.1 value moves.
- **Start line:**
  `Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 2 of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them. Stop at every gate. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records. Suggest each commit message in my bracketed-bullet format, as decision (C) in the plan records.`

| Unit | Patch | File | Hunks | + / - | Endings | `core.autocrlf` |
|---:|---|---|---:|---|---|---|
| 1 | 06 | `cmu_tare_model/constants.py` | 1 | 11 / 0 | CRLF | `true` |
| 2 | 06 | `cmu_tare_model/utils/measure_packages.py` (new) | 1 | 42 / 0 | New: CRLF | `true` |
| 3 | 06 | `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | 3 | 61 / 1 | LF | `false` |
| 4 | 06 | `cmu_tare_model/tests/energy_consumption_and_metadata/test_process_euss_data.py` | 1 | 54 / 0 | CRLF | `true` |
| 5 | 07 | `cmu_tare_model/constants.py` | 1 | 17 / 0 | CRLF | `true` |
| 6 | 07 | `cmu_tare_model/utils/efficiency_ratings.py` (new) | 1 | 42 / 0 | New: CRLF | `true` |
| 7 | 07 | `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | 4 | 10 / 1 | LF | `false` |
| 8 | 07 | `cmu_tare_model/utils/remdb_v4_installed_cost_utils.py` | 2 | 27 / 0 | CRLF | `true` |
| 9 | 07 | `cmu_tare_model/tests/utils/test_remdb_v4_installed_cost_utils.py` | 1 | 53 / 0 | CRLF | `true` |
| 10 | 07 | `cmu_tare_model/tests/energy_consumption_and_metadata/test_process_euss_data.py` | 1 | 10 / 0 | CRLF | `true` |

| Commit point | After unit | Check | Suggested subject line |
|---:|---:|---|---|
| 1 | 4 | Tests +18 | `Port cloud patch 06: read the dual-fuel ratings from the option string` |
| 2 | 10 | Tests +23 | `Port cloud patch 07: price the dual-fuel heat pump at SEER1` |

- Patch 06 must be committed before unit 5. Units 5, 7 and 10 edit files that 06 also
  edits, so at session start they are checked when reached, not before (step 2.1.5).

**End-of-session runs and checks**

- 2025.1 Pennsylvania, log `s2_PA_2025.log`: section 11.2, Session 2 column. On every
  sample rdu, `heating_upgrade_pm2_euss` is 16.0 (it was 15.2), `upgrade_hp_seer2` is
  15.2, `upgrade_hp_hspf2` is 7.8, and `upgrade_backup_afue` is 0.925 or 0.95.
- 2022.1.1 Pennsylvania, log `s2_PA_2022.log`: identical to "before".

### Session 3 -- backup furnace cost (08 with changes 02)

- **Decide first:** nothing left. (j) is taken (2.1): all four as written.
- **Modeled values:** 2025.1 only: total and net capital up by the furnace cost (cloud
  mean about $4,112 nationally), NPV down by the same, adoption down. No 2022.1.1 value
  moves.
- **No 2025.1 run between unit 1 and unit 7** (section 11.3).
- **Unattended:** decision (U). The session does not wait at commit point 1. After
  unit 7 it runs the tests, then the end-of-session runs below on the uncommitted
  working tree, and stops once at the end with the results, the `git add` commands
  for the commit's paths, and the commit message with the measured Pennsylvania
  numbers. The researcher stages and commits; the session never does.
- **Start message:** the whole block below, pasted as one message. It replaces the
  one-line start line for this session.

```text
Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 3 of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them with the changes below. The changes are mine, hold for Session 3 only, and are recorded in the plan as decision (U).

auto-approve this session

This session runs unattended. I pre-approve all seven units: the six .py files of patch 08 and notebook changes file 02. Still show each unit's whole diff and its dry run in the chat and log it in PORT_PROGRESS.md, then apply it without waiting for me.

Do not stop at the commit point (this replaces step 2.4.5 of the kickoff prompt for this session). After unit 7 run the tests, then go straight on to the end-of-session runs, 2025.1 Pennsylvania and then 2022.1.1 Pennsylvania, and their comparisons, all on the uncommitted working tree. Make no 2025.1 run before unit 7 is in.

Stop once, at the end, and give me three things. (1) The results, with the expected-against-actual table for 2025.1 and the comparison of 2022.1.1 with the "before" run. (2) The exact git add commands for the paths that belong in the commit, for me to run myself: whole files, or named hunks only if a file holds a change that does not belong in the commit. (3) One commit message for patches 08 and 09 together, in my bracketed-bullet format (decision (C)), with the Pennsylvania numbers this session measured. Never run git add or git commit yourself.

Stop earlier only if something is unexpected and wrong: anything on the kickoff prompt's list of what ends auto-approve, a 2022.1.1 difference from the "before" run, or a 2025.1 number that does not match section 11.2. Then report it and leave the files as they are. Two things on that list are not stops in this session. First, a usage-limit pause or a compacted conversation: after one, read PORT_PROGRESS.md and Session 3 of the plan again, check the working tree against the units logged as applied (step 2.1.4), and go on. Second, the known line-count case from Session 1: if git's default count differs from the patch's only because a blank line lines up differently, confirm the file is exactly the patch's result (git diff --numstat --minimal gives the patch's counts, and the file equals the audit copy's file or the committed text patched in the scratch folder), log it and go on.

Decision (j) is taken: I accept all four furnace choices as patch 08 writes them (P5, R1, D-S8, P8 / D-S7), as the plan records. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records.
```

| Unit | Patch | File | Hunks | + / - | Endings | `core.autocrlf` |
|---:|---|---|---:|---|---|---|
| 1 | 08 | `cmu_tare_model/utils/column_names.py` | 2 | 7 / 1 | CRLF | `true` |
| 2 | 08 | `cmu_tare_model/utils/remdb_v4_installed_cost_utils.py` (printed whole) | 1 | 158 / 0 | CRLF | `true` |
| 3 | 08 | `cmu_tare_model/private_impact/calculations/calculate_equipment_installation_costs.py` | 5 | 87 / 3 | CRLF | `true` |
| 4 | 08 | `cmu_tare_model/private_impact/calculate_lifetime_private_impact.py` | 4 | 28 / 0 | CRLF | `true` |
| 5 | 08 | `cmu_tare_model/tests/private_impact/calculations/test_backup_furnace_cost.py` (new; printed whole) | 1 | 129 / 0 | New: CRLF | `true` |
| 6 | 08 | `cmu_tare_model/tests/private_impact/test_calculate_lifetime_private_impact.py` | 1 | 55 / 0 | CRLF | `true` |
| 7 | 09 | Changes file 02: `cmu_tare_model/model_scenarios/tare_scenarios_v3_0.ipynb`, cells 13, 16, 17 | 3 cells | 35 / 4 | LF | Skill |

| Point | After unit | Check | Suggested subject line |
|---|---:|---|---|
| Test check, no commit | 6 | Tests +33 | -- |
| Commit point 1 (08 and 09 together) | 7 | Tests +33 | `Port cloud patches 08 and 09: price the backup gas furnace` |

**End-of-session runs and checks**

- 2025.1 Pennsylvania, log `s3_PA_2025.log`: section 11.2, Session 3 column, with the
  `Backup furnace:` line and no blank furnace cost.
- 2022.1.1 Pennsylvania, log `s3_PA_2022.log`: identical to "before".
- Optional, only if the researcher asks: 2025.1 national, section 11.4, "After Session
  3".

### Session 4 -- June 2026 rebates for dual fuel (10 with changes 03)

- **Decide first:** nothing left. (k) is taken (2.1), both parts: O-1 yes, and the
  Program Notice 26-3 comment goes in as written (unit 1).
- **Modeled values:** 2025.1 only: the `_sub_june2026` rebate, NPV and adoption. No
  2022.1.1 value moves.
- **No 2025.1 run between unit 1 and unit 7** (section 11.3).
- **Unattended:** decision (U4). The session does not wait at commit point 1. After
  unit 7 it runs the tests, then the end-of-session runs below on the uncommitted
  working tree, and stops once at the end with the results, the `git add` commands
  for the commit's paths, the commit message with the measured Pennsylvania
  numbers, and the `git commit -F` command for it. The researcher stages and
  commits; the session never does.
- **Start message:** the whole block below, pasted as one message. It replaces the
  one-line start line for this session.

```text
Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 4 of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them with the changes below. The changes are mine, hold for Session 4 only, and are recorded in the plan as decision (U4).

auto-approve this session

This session runs unattended. I pre-approve all seven units: the six .py files of patch 10 and notebook changes file 03. Still show each unit's whole diff and its dry run in the chat and log it in PORT_PROGRESS.md, then apply it without waiting for me.

Do not stop at the commit point (this replaces step 2.4.5 of the kickoff prompt for this session). After unit 7 run the tests, then go straight on to the end-of-session runs, 2025.1 Pennsylvania and then 2022.1.1 Pennsylvania, and their comparisons, all on the uncommitted working tree. Make no 2025.1 run before unit 7 is in.

Stop once, at the end, and give me three things. (1) The results, with the expected-against-actual table for 2025.1 and the comparison of 2022.1.1 with the "before" run. (2) The exact git add commands for the paths that belong in the commit, for me to run myself: whole files, or named hunks only if a file holds a change that does not belong in the commit. (3) One commit message for patches 10 and 11 together, in my bracketed-bullet format (decision (C)), with the Pennsylvania numbers this session measured. Save the same message as plain text in ~/tare_port and give me the git commit -F command that uses it. Never run git add or git commit yourself.

Stop earlier only if something is unexpected and wrong: anything on the kickoff prompt's list of what ends auto-approve, a 2022.1.1 difference from the "before" run, or a 2025.1 number that does not match section 11.2 (where section 11.2 gives a rounded figure, compare to the digits it gives). Then report it and leave the files as they are. Three things are not stops in this session. First, a usage-limit pause or a compacted conversation: after one, read PORT_PROGRESS.md and Session 4 of the plan again, check the working tree against the units logged as applied (step 2.1.4), and go on. Second, the known line-count case from Session 1: if git's default count differs from the patch's only because a blank line lines up differently, confirm the file is exactly the patch's result (git diff --numstat --minimal gives the patch's counts, and the file equals the audit copy's file or the committed text patched in the scratch folder), log it and go on. Third, the one expected change in the 2022.1.1 log: the MP3 and MP4 "[PASS] ... June 2026 fuel gate holds" lines now end with ", non-participating states $0.". Confirm that those two lines are the only marked lines that differ from the "before" log and that all 22 output files are the same bytes as the "before" run's, log it and go on; any other difference is a stop.

Decision (k) is taken, both parts, as the plan records: June 2026 equal to 2024 for the dual-fuel package is intended, and the Program Notice 26-3 comment that patch 10 adds to constants.py goes in as written. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records.
```

| Unit | Patch | File | Hunks | + / - | Endings | `core.autocrlf` |
|---:|---|---|---:|---|---|---|
| 1 | 10 | `cmu_tare_model/constants.py` | 3 | 20 / 0 | CRLF | `true` |
| 2 | 10 | `cmu_tare_model/private_impact/data_processing/determine_rebate_eligibility_and_amount.py` | 6 | 18 / 6 | CRLF | `true` |
| 3 | 10 | `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py` | 7 | 14 / 8 | CRLF | `true` |
| 4 | 10 | `scripts/verify_june2026_rebate_fossil_gate.py` | 3 | 86 / 54 | CRLF | `true` |
| 5 | 10 | `cmu_tare_model/tests/private_impact/test_rebate_june2026.py` | 1 | 84 / 0 | CRLF | `true` |
| 6 | 10 | `cmu_tare_model/tests/adoption_potential/test_adoption_reconciliation.py` | 1 | 15 / 0 | CRLF | `true` |
| 7 | 11 | Changes file 03: `cmu_tare_model/model_scenarios/tare_scenarios_v3_0.ipynb`, cell 30 | 1 cell | 38 / 12 | LF | Skill |

| Point | After unit | Check | Suggested subject line |
|---|---:|---|---|
| Test check, no commit | 6 | Tests +37 | -- |
| Commit point 1 (10 and 11 together) | 7 | Tests +37 | `Port cloud patches 10 and 11: June 2026 rebates for dual fuel` |

**End-of-session runs and checks**

- 2025.1 Pennsylvania, log `s4_PA_2025.log`: section 11.2, Session 4 column: the new
  June line, and every `_sub_june2026` rate equal to its `_sub` rate.
- 2022.1.1 Pennsylvania, log `s4_PA_2022.log`: identical to "before". Only the MP3 and
  MP4 `[PASS] ... June 2026 fuel gate holds` line changes: it now ends with
  ", non-participating states $0.". The values behind it are unchanged.

### Session 5 -- names, export columns, title, input checks (12, 13, changes 04, 15)

- **Decide first:** nothing left. (l) is taken (2.1): all three as written.
- **Modeled values:** none move, on either release.
- **Unattended:** decision (U5). The session does not wait at the four commit points
  and makes no commit between patches; at each point it runs the tests and goes on.
  After unit 11 it runs the tests, then the end-of-session runs below on the
  uncommitted working tree, and stops once at the end with the results, the
  commands for four commits (one for each patch), the four commit messages with
  the measured Pennsylvania numbers, and the `git commit -F` command for each. The
  researcher stages and commits; the session never does.
- **Start message:** the whole block below, pasted as one message. It replaces the
  one-line start line for this session.

```text
Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 5 of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them with the changes below. The changes are mine, hold for Session 5 only, and are recorded in the plan as decision (U5).

auto-approve this session

This session runs unattended. I pre-approve all eleven units: the ten .py units of patches 12, 13 and 15, and notebook changes file 04. Still show each unit's whole diff and its dry run in the chat and log it in PORT_PROGRESS.md, then apply it without waiting for me.

Do not stop at the four commit points, and do not wait for a commit between patches (for this session this replaces step 2.4.5 of the kickoff prompt and its rule to commit each patch before the next one that touches the same file). At each commit point run the tests, log the count and go on. Units 5, 6 and 8 change a file that patch 12 has already changed and that is not committed yet. For those three units, save a copy of the file in ~/tare_port just before the unit, and check the counts against that copy, not against the last commit: the lines added and removed from the copy to the file must equal the patch's section, in order, and the file must equal the committed text with both patches applied in the scratch folder. Any difference is a stop.

After unit 11 run the tests, then go straight on to the end-of-session runs, 2025.1 Pennsylvania and then 2022.1.1 Pennsylvania, and their comparisons, all on the uncommitted working tree.

Stop once, at the end, and give me three things. (1) The results, with the expected-against-actual table for 2025.1 and the comparison of 2022.1.1 with the "before" run. (2) The exact commands to make four commits, one for each patch (12, 13, 14, 15), in that order, for me to run myself. Patches 12 and 13 change two of the same files, so whole files cannot tell them apart: for 12 and for 13 give me git apply --cached with the patch's file in ~/tare_port/patches_lf, and for 14 and 15 give me git add with whole files. After each one, say what git diff --cached --stat should show. (3) Four commit messages, one for each patch, in my bracketed-bullet format (decision (C)), with the Pennsylvania numbers this session measured where they apply. Save each as plain text in ~/tare_port and give me the git commit -F command that uses it. Never run git add, git apply --cached or git commit yourself.

Stop earlier only if something is unexpected and wrong: anything on the kickoff prompt's list of what ends auto-approve, a 2022.1.1 difference from the "before" run, or a 2025.1 number that does not match section 11.2 (where section 11.2 gives a rounded figure, compare to the digits it gives). Then report it and leave the files as they are. Three things are not stops in this session. First, a usage-limit pause or a compacted conversation: after one, read PORT_PROGRESS.md and Session 5 of the plan again, check the working tree against the units logged as applied (step 2.1.4; for a file that two patches have changed, against the committed text with both patches applied in the scratch folder), and go on. Second, the known line-count case from Session 1: if git's default count differs from the patch's only because a blank line lines up differently, confirm the file is exactly the patch's result (git diff --numstat --minimal gives the patch's counts, and the file equals the audit copy's file or the committed text patched in the scratch folder), log it and go on. Third, the two 2022.1.1 log lines that changed in Session 4: the MP3 and MP4 "[PASS] ... June 2026 fuel gate holds" lines end with ", non-participating states $0.". Confirm that those two lines are the only marked lines that differ from the "before" log and that all 22 output files are the same bytes as the "before" run's, log it and go on; any other difference is a stop.

Decision (l) is taken, as the plan records: the 2025.1 peak names, the 13 extra Tepper columns and the package 5 title go in as written. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records.
```

| Unit | Patch | File | Hunks | + / - | Endings | `core.autocrlf` |
|---:|---|---|---:|---|---|---|
| 1 | 12 | `cmu_tare_model/utils/column_names.py` | 1 | 44 / 0 | CRLF | `true` |
| 2 | 12 | `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | 3 | 19 / 16 | LF | `false` |
| 3 | 12 | `cmu_tare_model/utils/export_tepper_csv.py` | 3 | 13 / 6 | CRLF | `true` |
| 4 | 12 | `cmu_tare_model/tests/utils/test_column_names.py` | 1 | 31 / 0 | CRLF | `true` |
| 5 | 13 | `cmu_tare_model/utils/export_tepper_csv.py` | 8 | 39 / 2 | CRLF | `true` |
| 6 | 13 | `cmu_tare_model/tests/utils/test_column_names.py` | 1 | 20 / 0 | CRLF | `true` |
| 7 | 14 | Changes file 04: `cmu_tare_model/tare_model_main_v3_0.ipynb`, cell 9 | 1 cell | 3 / 0 | LF | Skill |
| 8 | 15 | `cmu_tare_model/energy_consumption_and_metadata/process_euss_data.py` | 3 | 37 / 7 | LF | `false` |
| 9 | 15 | `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py` | 6 | 21 / 4 | CRLF | `true` |
| 10 | 15 | `cmu_tare_model/tests/energy_consumption_and_metadata/test_process_euss_data.py` | 1 | 22 / 0 | CRLF | `true` |
| 11 | 15 | `cmu_tare_model/tests/adoption_potential/test_adoption_reconciliation.py` | 2 | 25 / 0 | CRLF | `true` |

| Commit point | After unit | Check | Suggested subject line |
|---:|---:|---|---|
| 1 | 4 | Tests +40 | `Port cloud patch 12: winter and summer names for 2025.1 peak columns` |
| 2 | 6 | Tests +41 | `Port cloud patch 13: dual-fuel and panel columns in the Tepper export` |
| 3 | 7 | Tests +41 | `Port cloud patch 14: add the package 5 title to the main notebook` |
| 4 | 11 | Tests +47 | `Port cloud patch 15: require true/false applicability; no default weight` |

- Under decision (U5) patch 12 is not committed before unit 5. Units 5 and 6 edit
  files that 12 also edits, so at session start they are checked when reached (step
  2.1.5), and their counts are checked against a copy of the file saved just before
  the unit. The same holds for unit 8, after unit 2.

**End-of-session runs and checks**

- 2025.1 Pennsylvania, log `s5_PA_2025.log`: section 11.2, Session 5 column: Tepper
  main 182 and detailed 332 columns; peak columns named
  `..._peak_electricity_winter_kw` and `..._summer_kw`; titles read "Dual-fuel heat pump
  with gas backup furnace".
- 2022.1.1 Pennsylvania, log `s5_PA_2022.log`: identical to "before", including both
  Tepper column lists and the peak names. As from Session 4, the MP3 and MP4
  `[PASS] ... June 2026 fuel gate holds` lines differ from the "before" log; no
  other marked line may. This is the first time `require_true_false` (patch 15)
  runs on the 2022.1.1 CSVs; a difference or a stop at load may be a real finding.

### Session 5b -- cleanup, one commit (no patch)

- **Why:** decision (R). Three things the patches add are larger than they need to
  be. They are ported as written first, so each file can be checked against its
  patch, and made smaller here. Item 4 is a comment only (decision (i)). Items 5
  to 8 are the small items noted while Session 5's patches went in; the
  researcher added them on 7 Oct 2026 (decision (5b)).
- **Decide first:** the module that takes `is_dual_fuel_package` (item 2). Taken:
  decision (5b).
- **Modeled values:** none move, on either release. No calculation changes. Item 5
  removes lines that cannot run, and item 6 changes the text of one error message.
- **Start line:**
  `Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 5b of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them. Stop at every gate. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records. Suggest each commit message in my bracketed-bullet format, as decision (C) in the plan records.`

Start with an inventory: for each item, where the code is defined and every place
that names it (file and line). Show it to the researcher before the first diff.

| Item | Change |
|---:|---|
| 1 | Fold `cmu_tare_model/utils/efficiency_ratings.py` into its one caller, `add_dual_fuel_spec_columns` in `process_euss_data.py`: the two conversions become two lines that divide by `SEER2_PER_SEER1` and `HSPF2_PER_HSPF1`. Remove the file, its import, the test of its two functions, and the comments that name the file. |
| 2 | Move `is_dual_fuel_package` out of `cmu_tare_model/utils/measure_packages.py` into an existing module that reads no data file when imported. Candidates: `utils/resstock_schema.py`, `utils/calculation_utils.py`. Not `utils/modeling_params.py`, which loads the lookup tables when imported. The researcher chooses from the inventory. Update every import, the two scenarios-notebook cells that import it, and the comments that name the old file. Remove the file. |
| 3 | Move the tests of `cmu_tare_model/tests/private_impact/calculations/test_backup_furnace_cost.py` into the existing test files of the two modules they test: `test_calculate_equipment_installation_costs.py` in the same folder, and `cmu_tare_model/tests/utils/test_remdb_v4_installed_cost_utils.py`. Remove the file. |
| 4 | Reword the comment above `SEER2_PER_SEER1` and `HSPF2_PER_HSPF1` in `cmu_tare_model/constants.py` (patch 07): drop "PROVISIONAL (session decision P1)" and "not checked against the DOE Appendix M1 text", and give the researcher's reason from decision (i). The two values do not change. Ask the researcher whether to also move the block below `EFFICIENCY_FLOORS_PM2`, so that the efficiency-floors comment sits on its table again. |
| 5 | In `visuals_adoption_dotplot.py` (patch 15), remove the part of `build_econ_plot_df` that cannot run: the no-weight fallback of its inner helper, with the `ValueError` patch 15 added to it, and the `scaling_factor` argument that only that fallback read. A frame with no `weight` column already stops earlier, in `adoption_reconciliation_table`. In `test_adoption_reconciliation.py`, give `test_prepare_plot_data_reads_the_weight` a frame laid out as `prepare_plot_data` expects, so that it reaches the weight lines, and make the no-weight test say what it checks (the `KeyError` naming `weight`). |
| 6 | In `cmu_tare_model/utils/column_names.py` (patch 12): drop "(NEXT_STEPS, Phase 2 Task 7)" from the comment, and make `create_peak_electricity_col` stop on an unknown `end_use` with a message that names it and the expected values (a `KeyError`, as now). |
| 7 | In `cmu_tare_model/utils/export_tepper_csv.py` (patch 13): the four Args lines that say "(3 or 4)" name package 5 too. The test `release == "2025.1"` for the panel columns stays as it is: `process_euss_data.py` uses the same test at the two places that create those columns. |
| 8 | In the tests Session 5 added (`test_column_names.py`, `test_process_euss_data.py`, `test_adoption_reconciliation.py`): one-line docstrings, descriptive names in place of `c` and `mp`, and no decision ID "(G4)" in the heading comment. |
| 9 | Added by the researcher at the commit point (7 Oct 2026): a comment tagged `TODO (grid impact, 2025.1)` wherever the grid impact analysis is tied to ResStock 2022.1.1, saying it works on that release only for now and will be updated for the dual-fuel analysis (2025.1). Places: `GRID_IMPACT_ANALYSIS` in `constants.py`, `grid_impact/peak_load_functions.py`, `grid_impact/build_parcel_frame.py`, and cells 32 and 33 of `tare_model_main_v3_0.ipynb`. Comments only. |

- **How:** these are hand edits with the Edit tool, not patches, so C2 to C4 do not
  apply. Each file is one gated diff, shown with 3 to 5 lines of context, and keeps
  its line endings. A notebook cell changes only through the notebook-cell-edit
  skill: `check`, approval, `apply`. Removing a file is the researcher's step
  (`git rm`), made once nothing names the file any more.
- **Checks before the commit:** the test count is the count after patch 15 less the
  one test removed with item 1 (+46: 398 passed, 1 skipped), since items 5 to 8 add
  and remove no test; every moved test still runs; and a search finds no
  `efficiency_ratings` or `measure_packages` outside `cmu_tare_model/docs/`.
- **One commit**, in the format of decision (C).

**Runs and checks after the commit**

- 2025.1 Pennsylvania, log `s5b_PA_2025.log`: every output identical to Session 5's
  2025.1 Pennsylvania run, compared as step 2.5.2 says.
- 2022.1.1 Pennsylvania, log `s5b_PA_2022.log`: identical to "before".

### Session 5c -- second cleanup, one commit (no patch)

- **Why:** decision (5c). Four small things Session 5b left as they were. Comments,
  docstrings and tests only: apart from the text of one error message (item 4) and
  one line of the baseline notebook's printed banner (item 1), no line that runs
  in the model changes.
- **Decide first:** whether `build_parcel_frame` (item 4) is still wanted. Nothing in
  a module, notebook, script or test calls it; the main notebook imports only
  `build_weight_dict_from_mapping` from that file. Ask the researcher at the start.
  Taken: it is kept (decision (5c)).
- **Modeled values:** none move, on either release.
- **Start line:**
  `Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 5c of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them. Stop at every gate. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records. Suggest each commit message in my bracketed-bullet format, as decision (C) in the plan records.`

Start with an inventory, as in Session 5b: search the code again for decision IDs and
planning-document names, and show every hit (file and line) before the first diff.
Item 1 lists what the search of 7 Oct 2026 found at commit `992d5c4`; the session's
own search decides.

| Item | Change |
|---:|---|
| 1 | Take decision IDs and planning-document names out of comments and docstrings: say the same thing in plain words, or drop the ID where the sentence stands without it. The reason a rule exists stays; only the ID goes. Model code: `cmu_tare_model/constants.py` lines 383 and 385 ("NEXT_STEPS D8", "researcher's decision R3"); `private_impact/calculations/calculate_equipment_installation_costs.py` line 188 ("session decision P5"); `private_impact/data_processing/determine_rebate_eligibility_and_amount.py` line 530 ("D8, R3"); `utils/remdb_v4_installed_cost_utils.py` line 940 ("R1"); and cell 30 of `model_scenarios/tare_scenarios_v3_0.ipynb` ("NEXT_STEPS D8; researcher's decision R3"), through the notebook-cell-edit skill. Tests, one heading comment or docstring each: `tests/adoption_kpis/test_data_loading.py` ("G10"), `tests/adoption_potential/test_adoption_reconciliation.py` ("G8"), `tests/energy_consumption_and_metadata/test_process_euss_data.py` ("G5"), `tests/private_impact/test_calculate_lifetime_private_impact.py` ("G7"), `tests/private_impact/test_rebate_june2026.py` ("G8"), `tests/utils/test_remdb_v4_installed_cost_utils.py` ("G6"). Added from the session's own search (decision (5c)), all older than the port: `private_impact/data_processing/determine_rebate_eligibility_and_amount.py` line 471 ("per D8 (Phase 3)"); `energy_consumption_and_metadata/process_euss_data.py` lines 252, 261 and 342 ("D3"); and cell 3 of `model_scenarios/tare_baseline_v3_0.ipynb`, line 27 ("Phase 3, D3"), through the notebook-cell-edit skill. Added by the researcher (decision (5c)), labels with no decision ID, all older than the port: `energy_consumption_and_metadata/process_euss_data.py` lines 384 and 1422 ("Phase 1 audit, Section B.7", "Phase 3's masking funnel"); `utils/resstock_schema.py` line 4 ("the Phase 1 column map"); `grid_impact/peak_load_functions.py` line 3 ("Phase 2 BSQ refactor"); `public_impact/data_processing/validate_damages_dataframes.py` lines 35 and 50 ("Session 3"); `cmu_tare_model/constants.py` line 582 ("P0.2"); and cell 3 of the baseline notebook, lines 16 (the printed banner) and 28 ("Phase 3"). |
| 2 | One-line docstrings on the five backup furnace tests that Session 5b moved (three in `tests/utils/test_remdb_v4_installed_cost_utils.py`, two in `tests/private_impact/calculations/test_calculate_equipment_installation_costs.py`), and `menu_mp` in place of `mp` in the one lambda. What the tests check does not change. |
| 3 | One new test in `tests/utils/test_column_names.py`: `build_household_column_list` holds the four electric panel columns (`panel_service_rating_amps` and the three `mp{mp}_panel_constraint_*` columns) on a 2025.1 run, and none of them on a 2022.1.1 run. |
| 4 | In `cmu_tare_model/grid_impact/build_parcel_frame.py`: two docstrings and one comment name `compute_peak_load_summary` and `already_weighted`, which no module defines; the module docstring points to a CLAUDE.md section that does not exist; and one error message names the branch `joseph-2026-nature-comms-submission`. Reword them to what exists today, as the researcher decides ("Decide first"). The `TODO (grid impact, 2025.1)` comment from Session 5b stays. |
| 5 | Added by the researcher at the commit point (7 Oct 2026): the module docstring of `cmu_tare_model/grid_impact/peak_load_functions.py` names a notebook that is no longer in the repo (`calculate_postTARE_ts_aws_peak_demand.ipynb`) and "the national loop (Step 9)", which no tracked code holds. Replace the two lines with what uses the module today, the main notebook. Docstring only. |

- **How:** as in Session 5b, and as the kickoff prompt's step 2.9 sets out: hand edits
  with the Edit tool, each file one gated diff with 3 to 5 lines of context, endings
  kept; the notebook cell only through the notebook-cell-edit skill.
- **Checks before the commit:** the test count is Session 5b's plus the one new test of
  item 3 (+47: 399 passed, 1 skipped); and the search of item 1 finds nothing outside
  `cmu_tare_model/docs/`, archived files and the `*_EXPORT_*.py` snapshots. That
  search is for decision IDs, the names of planning documents, and phase,
  session and audit labels. The changelog pointers that decision (5c) leaves
  are not part of it.
- **One commit**, in the format of decision (C).

**Runs and checks after the commit**

- 2025.1 Pennsylvania, log `s5c_PA_2025.log`: every output identical to Session 5b's
  2025.1 Pennsylvania run (`2026-10-07_14-43`), compared as step 2.5.2 says.
- 2022.1.1 Pennsylvania, log `s5c_PA_2022.log`: identical to "before".
- In both logs the baseline notebook's banner line `SCOPE FILTERS: ...` no longer
  ends "(Phase 3)" (item 1), so that one line differs from every earlier log. It
  is not a marked line, and no output file changes.

### Session 6 -- national runs and records (no patches)

- **Decide first:** (m). Ask it at the start. The two national runs and the final test
  run do not depend on it; the documentation edits do. Taken: yes to every part
  (decision (m)).
- **Modeled values:** none move; documents only.
- **Auto-approve never covers this session's documentation edits.**
- **Start line:**
  `Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 6 of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them. Stop at every gate. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records. Suggest each commit message in my bracketed-bullet format, as decision (C) in the plan records.`

Open `cmu_tare_model/docs/REFERENCE_VALUES.md` before any comparison. Then, one run at
a time:

1. 2025.1 national, log `s6_national_2025.log`: section 11.4, Final column, and the
   "Final run in detail" tables of `PROVISIONAL_RESULTS_2026-10-07.md`.
2. 2022.1.1 national, log `s6_national_2022.log`: every output identical to Session 1's
   national "before" run (Step B), Tepper files included. Cross-check against the
   REFERENCE_VALUES rows listed in section 11.5.
3. Final test run: tests +47, the count after the Session 5c cleanup.
4. Then the gated documentation edits, one at a time:
   1. new rows in `cmu_tare_model/docs/REFERENCE_VALUES.md` for 2025.1 package 5, from
      this local run. Add rows only; never change an existing row; one approval per
      group of rows.
   2. CLAUDE.md: the Program Notice 26-3 and dual-fuel block.
   3. CLAUDE.md: the `TARE_RESSTOCK_RELEASE` and runner block.
   4. CLAUDE.md: Limitation 14.
   5. CLAUDE.md: the winter and summer peak-name block.
   5a. CLAUDE.md, Limitation 2: its last sentence says the dual-fuel package's
       June 2026 HEEHR treatment "is not yet set (Phase 7)". Decision (k) set
       it; say what it is (decision (m)).
   5b. CLAUDE.md: an entry for the port in the dated "Last updated" block at
       the top (decision (m)).
   6. a `cmu_tare_model/docs/SESSION_LOG.md` entry.
   7. `cmu_tare_model/docs/tare_tepper_exports_data_dictionary.md`: rows for
      package 5, the 13 new columns and the winter and summer peak names, from
      this local run (decision (5b)).

   Units 2 to 5 are the guide's "Proposed CLAUDE.md text", adjusted to the decisions
   actually taken and to the Session 5b cleanup (the guide's text names
   `utils/measure_packages.py`, which that cleanup removes).
5. Ask the researcher what becomes of `cloud_run/` (decision (b)), and list the by-hand
   items of (n).

### Session 7 -- after the port: grid impact for 2025.1, ResStock naming, cleanup

- **Why:** decision (S7). Three pieces of work the port left for later. Not started.
- **Before it starts:** the working tree is clean. On 8 Oct 2026 it held four things
  to settle first: the simulation notebook with its package cells grouped by
  release and the main notebook's opening text, both not yet committed; saved
  outputs in the main notebook, to clear; the notebook snapshots of decision (n),
  still to export; and an edit to `constants.py` that makes 2025.1 the default
  release, which the researcher puts back to 2022.1.1 for now (item 10).
- **Decide first:** one question for each part, asked when that part starts. They
  are marked "Decide first" in the tables below.
- **Modeled values:** Part 1 adds results for 2025.1 (peaks) and moves no existing
  value. Part 2 moves none; if the ten columns of item 6 are renamed, the header of
  the results files changes, and new runs are no longer byte-identical to earlier
  ones. Item 9 moves 2025.1 values if the sample changes. Part 4 moves values on
  both releases, the paper's 2022.1.1 results among them: item 13 is decided, and
  items 14, 16, 17 and 18 move values if taken. Each is `VALUE-MOVING`, in its own
  commit, with new REFERENCE_VALUES rows. Item 10 moves none.
- **Start line:**
  `Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session 7 of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them. Stop at every gate. For tests, use C6 with --ignore=archived_files added, as decision (T) in the plan records. Suggest each commit message in my bracketed-bullet format, as decision (C) in the plan records.`

Each part is its own work, with its own commit or commits, and can be a conversation
of its own. Scratch files take the prefix `s8_`: `s7_` is in use in `~/tare_port` by
the conversation of 8 Oct 2026 that closed Session 6.

Order (researcher, 8 Oct 2026): the boiler fix of Part 4 comes first. The default
release changes to 2025.1 only after it (item 10).

**Part 1 -- grid impact for ResStock 2025.1**

What happened on 8 Oct 2026: a national 2025.1 run made in VS Code went on into the
grid impact cells. They sent the run's building ids to the ResStock 2022.1.1 table,
read two timeseries of 0.8 GB each from AWS, wrote them into the query cache (8.06 GiB
afterwards) and held about 76 GB of memory. Those results cannot be used. The runner
avoids this only because every 2025.1 run is given `--skip-grid-impact`.

| Item | Change |
|---:|---|
| 1 | First, and in a commit of its own: make the grid impact cells of `tare_model_main_v3_0.ipynb` stop with a clear message when the run's release is one the analysis does not support, so that a 2025.1 run cannot send its building ids to the 2022.1.1 table. |
| 2 | Decide first: the AWS table that holds the 2025.1 timeseries, and whether it holds package 5. Cell 33 sets the workgroup, database, table and schema of `BuildStockQuery` for 2022.1.1 only; choose them by release. |
| 3 | Use the release's own peak column names where the grid impact code names them. `REQUIRED_EXPORT_PEAK_COLUMN_TEMPLATES` in `grid_impact/build_parcel_frame.py` holds the 2022.1.1 names; build them with `create_peak_electricity_col`. |
| 4 | Keep the two releases' query caches apart. Decide what becomes of the two entries the run of 8 Oct 2026 added to `resstock_amy2018_release_1_1_query_cache.pkl`: leave them, or rebuild the cache without them. |
| 5 | When a 2025.1 run passes with grid impact on: take `--skip-grid-impact` out of the rule for 2025.1 runs (the runner's notes and CLAUDE.md); remove the five `TODO (grid impact, 2025.1)` comments (`constants.py`, `grid_impact/peak_load_functions.py`, `grid_impact/build_parcel_frame.py`, and cells 32 and 33 of the main notebook); and add REFERENCE_VALUES rows for the 2025.1 peaks. |

**Part 2 -- "ResStock" and the release, in place of "EUSS"**

One planned change, in commits of its own; not to be done piecemeal before then.
Counted on 8 Oct 2026 at commit `bffd8e9`: `euss` is on 430 lines of 33 tracked `.py`
and `.ipynb` files, archived files and snapshots left out. Start with a fresh
inventory, as in Session 5b.

| Item | Change |
|---:|---|
| 6 | Decide first: ten columns of the results file end `_euss` (`heating_upgrade_pm1_euss`, `heating_replacement_pm2_euss_original`, `heating_backupFurnace_pm1_euss` and the like). Renaming them changes the files' header, so runs must then be compared by value and not byte for byte. Rename them, or leave these ten as they are. |
| 7 | Decide first: the file names `energy_consumption_and_metadata/process_euss_data.py` and `tests/energy_consumption_and_metadata/test_process_euss_data.py`. A rename is the researcher's `git mv`, with every import, CLAUDE.md and the documents that name the module updated in the same commit. |
| 8 | Rename every dataframe, variable, function and constant that holds `euss` to `resstock` (`df_euss_am_mpX_home`, `df_euss_am_baseline_home`, `load_euss_baseline`, `load_euss_upgrade`, `pm1_euss`, `_EUSS_COL_ALIASES` and the rest), in the modules, the tests and the four notebooks. In printed lines, comments, docstrings and notebook markdown, write "ResStock" with the release where the text is about one release ("ResStock 2022.1.1"), and "ResStock" alone where it holds for both. Printed lines that say "EUSS" on a 2025.1 run today: the banner of simulation notebook cell 7, "Model Run Complete for EUSS Measure Package" in the six package run cells, baseline notebook cell 1, and scenarios notebook cell 23. CLAUDE.md's own "EUSS" entries follow. Names that ResStock itself publishes stay, and so do the `*_EXPORT_*.py` snapshots and the archived files. |

**Part 3 -- cleanup found after the port**

| Item | Change |
|---:|---|
| 9 | Decide first: homes that heat with electricity are in the dual-fuel sample, and the package gives them a gas backup furnace. They are 8,778 of the 161,983 sample rdu (2,228,766 homes, 5.42%), and 3,257 of the 5,294 rdu that adopt with no rebate (run `2026-10-07_19-14`). Keep them, or limit the package 5 sample to homes that heat with a fossil fuel. |
| 10 | The default release stays 2022.1.1 for now: the paper and a colleague's work use it (researcher, 8 Oct 2026). The researcher puts the uncommitted edit in `constants.py` back. It changes to 2025.1 after the boiler fix of item 13, for the Tepper result files. That change is more than one line: with the default at 2025.1 and the variable unset, the test run of 8 Oct 2026 gave 8 failed and 9 errors (382 passed, 1 skipped), from tests that take 2022.1.1 for granted when the variable is unset, in `tests/test_constants.py`, `tests/utils/test_column_names.py`, `tests/private_impact/calculations/test_calculate_equipment_installation_costs.py` and `tests/adoption_kpis/test_demand_sample.py` among them. When the time comes: change the line and the comment above it, make those tests name the release they need, and update what says "unset means 2022.1.1" (CLAUDE.md, the runner's notes). One commit. A 2022.1.1 run then needs the variable or `--release 2022.1.1`. Until then, a way to choose the release for a VS Code session without editing `constants.py`. |
| 11 | What is left of decision (n) when this session starts. Only if the researcher wants it: the order of the docstring in cell 30 of the scenarios notebook. |
| 12 | Only if the researcher wants them: the two older CLAUDE.md sentences left as they are in Session 6 ("Program rules" names `REBATE_ELIGIBLE_HEATING_MPS`; Limitation 2 on June 2026 HOMES); and in `environment-cmu-tare-model.yml`, the `prefix:` line of another machine, three older versions, and the two conda lines for typing-extensions. |

**Part 4 -- cost estimation: the closest REMDB fit**

The rule (decision (S7)): each cost uses the REMDB row that fits the equipment most
closely. How the costs are priced today, in `utils/remdb_v4_installed_cost_utils.py`:
the credit for an old heating system uses one of two rows, at the old system's own
size and efficiency: `furnaces_gas_furnace` for every fossil system (gas, propane and
fuel oil; furnaces and boilers) and `electric_baseboard_default` for every electric
one (baseboard, electric furnace, electric boiler). The counts below are study-sample
rdu of the national runs `2026-10-07_19-22` (2022.1.1) and `2026-10-07_19-14`
(2025.1). Start by measuring what each item moves, before any diff.

| Item | Change |
|---:|---|
| 13 | Decided: price the credit for an old boiler with the boiler rows the table holds (`boiler_gas_non_condensing`, `boiler_gas_condensing`, `boiler_oil`), not with the gas furnace row. Boilers on the furnace row today: 20,973 rdu in 2022.1.1 (gas 16,335, fuel oil 3,983, propane 655) and 2,528 in 2025.1 (gas 2,362, fuel oil 153, propane 13). Decide first: the row for a propane boiler (a gas boiler row is the closest), how a gas boiler is split between the two gas rows (the condensing row covers AFUE 0.91 to 0.97, the other 0.80 to 0.87; 6.5% of the 2022.1.1 boilers and 2.0% of the 2025.1 ones are published at 0.90 or more), and the lowest efficiency a replacement is priced at (the furnace floor is 0.80). An electric boiler has no row and stays on the baseboard row. `VALUE-MOVING` on both releases. |
| 14 | Decide first: an electric furnace is priced with the baseboard row, because the table has no electric furnace row: 47,155 rdu in 2022.1.1 and 7,355 in 2025.1. That row is a straight line through zero, so the credit follows the size closely: from 1,773 to 9,309 dollars between the 5th and 95th percentile in 2022.1.1, where a gas furnace's credit runs from 3,228 to 4,279. Homes that heat with electricity are most of the no-rebate adopters, so this credit matters to the headline numbers. Is the baseboard row the closest fit for a ducted electric furnace? |
| 15 | No change, a line for the records: a propane or fuel oil furnace is priced with the gas furnace row, the closest the table has (2022.1.1: 11,594 and 9,888 rdu; 2025.1: 410 and 1,270). Say so in CLAUDE.md's limitations. |
| 16 | Decide first: a size outside a row's range is priced by carrying the row's line beyond the range; nothing is held at the bound (CLAUDE.md Limitation 9). Credits on the gas furnace row below its 30,000 Btu/h: 19,994 rdu (2022.1.1) and 18,652 (2025.1); above its 156,250: 3,199 and 2,763. Central AC credits below 1.5 tons: 32,084 and 25,664; above 5 tons: 22,578 and 18,260. Room AC credits below 0.4167 tons: 15,593 and 5,402. The 2025.1 backup furnace below 30,000 Btu/h: 20,795 rdu; above 156,250: 2,992. The MP4 heat pump's SEER1 of 24 to 29.3 is above the ducted row's 24. Keep, hold at the bound, or flag. |
| 17 | Efficiency floors are uneven. A central AC credit is priced at SEER1 15 for every home (133,890 of the 177,641 central AC rdu of 2022.1.1 are published below it). A room AC credit has no floor: 2,768 rdu (2022.1.1) and 325 (2025.1) are priced below the row's lowest value, 9.4, and the value given to the row is ResStock's EER where the row takes CEER. The effect on a room AC credit is small (about 6 dollars for one point). Decide whether room AC gets a floor, and settle the floors of the boiler rows with item 13. |
| 18 | Two heat pump rows are never used: `air_source_heat_pump_centrally_ducted_with_new_circuit` and `air_source_heat_pump_non_ducted_single_zone`. Every home without ducts is priced as a multi-zone system (24,853 rdu in 2022.1.1), and no home is priced with a new circuit, though 2025.1 publishes electric panel columns. Decide whether either row is the closer fit for some homes. |

- **How:** hand edits, as the kickoff prompt's step 2.9 sets out: each file one gated
  diff from a scratch copy, endings kept; a notebook cell only through the
  notebook-cell-edit skill; a file is removed or renamed by the researcher.
- **Checks:** the test count stays 399 passed and 1 skipped unless a part adds tests.
  After each commit that touches code or a notebook, a Pennsylvania run on each
  release: every output identical to the run before it, apart from what the part
  changes on purpose (new 2025.1 peak results in Part 1; the ten column names of
  item 6 if renamed; 2025.1 values if item 9 changes the sample). A national run on
  each release closes a part that adds or moves a value.
