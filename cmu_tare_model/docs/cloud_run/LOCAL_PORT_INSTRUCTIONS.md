# Porting the cloud dual-fuel (package 5) work to your machine -- step by step

Written 6 Oct 2026 for the researcher. Your local Claude Code reads the companion file,
`LOCAL_KICKOFF_PROMPT.md`. Both files are in `cmu_tare_model/docs/cloud_run/`.

## Quick start

Each step says why, and points to the details.

1. Save `cloud_run_bundle.tar.gz` from the cloud session's chat to your Downloads
   folder. It carries the 15 changes and these instructions (section 1).
2. In Git Bash, go to the repo root, on the dev branch with nothing uncommitted, so
   every change Claude makes is one you can see (steps 3.1 to 3.3).
3. Unpack the bundle there. This adds one folder and changes no code (step 3.4).
4. Activate `cmu-tare-model`, and install pyarrow if it is missing; patch 02 needs it
   (step 3.6). Close any open notebook tabs, and make Claude's scratch folder with
   `mkdir -p ~/tare_port` (steps 3.7 and 3.8).
5. Open the repo in VS Code and start a new Claude Code conversation in auto mode.
   Type `/add-dir C:\Users\jorda\tare_port` as your first message: Claude keeps its
   scratch files there, and the access lasts for that conversation only (section 4).
6. Paste the Session 0 line. Claude audits your repo and tries every change in a
   throwaway copy (section 5).
7. Answer questions (a) to (h) in one reply, then approve the plan file Claude shows you
   (sections 5 and 10).
8. For Sessions 1 to 5, in order: start a fresh conversation, type the `/add-dir` line,
   paste that session's line from the plan, approve each file, commit at each commit
   point, and reply `committed` (section 6).
9. Session 6, started the same way (fresh conversation, `/add-dir` line first), runs
   both releases nationally and drafts the new reference rows (section 9).

If a session is cut off, see section 12.1.

## Words used here

- **Patch**: one numbered change (`NN_*.patch`). It lists lines to remove (`-`) and add
  (`+`), with a few unchanged lines around them for context.
- **Section, hunk**: a patch has one section per file it changes. A section has one or
  more hunks; each hunk is one block of nearby changed lines.
- **Changes file**: the notebook form of a patch, replayed by the notebook-cell-edit
  skill. Changes files 01 to 04 stand for patches 04, 09, 11 and 14.
- **Approval unit**: what Claude shows in one turn and you approve once. For this port
  it is one file: all of one patch's changes to that file (question d).
- **Auto-approve**: an opt-in you type, for example `auto-approve this patch`, that lets
  Claude apply the files in that scope without waiting for each `approved` (6.2).
- **Gate**: a point where Claude stops and waits for your reply.
- **Commit point**: the end of a patch, where you commit.
- **Dry run**: a check that a change fits your file, which writes nothing.
- **Checksum**: a fingerprint of a file. It proves the patches arrived unchanged.
- **Line endings**: Windows ends lines with CRLF, Linux with LF. Git for Windows often
  converts between them, so Claude picks the apply command file by file.
- **Offset**: git found a hunk's lines a little higher or lower than expected, because
  your file changed after `b17d7af`, the commit the patches were made against.
- **Untracked, stage**: an untracked file is one git does not follow (`??` in
  `git status`). Staging picks the files that go into the next commit.
- **Run stamp**: the date and time in each output file's name.
- **"Before" run**: Session 1's 2022.1.1 Pennsylvania run on your unchanged model code.
  Every later 2022.1.1 run must equal it.
- **rdu**: a representative dwelling unit, one ResStock row (CLAUDE.md, Terminology).
- **unsub / sub / sub_june2026**: no rebate / 2024 rebate guidance / June 2026 guidance.
- **heatingLCC_coolingLCC** and its two siblings: the three NPV scopes (CLAUDE.md).
- **Tepper files**: the household and county CSV exports in `tepper_export/`.
- **Grid impact**: the optional peak-demand step that needs AWS. It is off for 2025.1.

---

## 1. What is `cloud_run_bundle.tar.gz`?

- It is a compressed archive, like a `.zip` file, sent to you as a file in the cloud
  session's chat. Save it to your Downloads folder.
- It holds one folder, `cmu_tare_model/docs/cloud_run/`, which contains:
  - the five cloud documents: `IMPLEMENTATION_GUIDE_2026-10-07.md` (the list of
    changes), `DECISIONS_TAKEN.md`, `BREAKAGE_LOG.md`,
    `PROVISIONAL_RESULTS_2026-10-07.md` and `TEPPER_EXPORT_REVIEW_2026-10-07.md`;
  - this file and `LOCAL_KICKOFF_PROMPT.md`;
  - `patches/`: the 15 changes, `01_*.patch` to `15_*.patch`;
  - `notebook_changes/`: the 4 notebook changes files, `01_*.json` to `04_*.json`.
- The cloud session was not allowed to commit, so this archive is how the work gets to
  your machine.
- Unpack it at the repo root (step 3.4). That creates the one folder above and
  overwrites nothing outside it. Your code does not change until a session applies a
  patch.
- If you also have `cloud_run_bundle_stageA.tar.gz`, ignore it. It is an older,
  incomplete snapshot without these instructions.
- Use these steps instead of two older sets of commands:
  - the quick `git apply` commands near the top of the implementation guide;
  - the whole "Reproducing the files on the researcher's Windows machine" section at the
    end of the Tepper review. Do not check out `cloud/resstock2025-1-mp5`: it holds
    cloud-only setup commits (`5ae9051`, `62aa53e` and others) that are never ported,
    and none of this work. Port onto your dev branch.

  Those commands apply whole patches, notebooks included, with no review of each
  change. A bare `git apply` also follows your `core.autocrlf` setting: with it off, it
  fails on CRLF files; with it on, it rewrites an LF file to CRLF.

---

## 2. The plan in brief

| Session | Patches | What it does | Do modeled values move? | Decide first |
|---|---|---|---|---|
| 0 | none | Audit and plan; a trial of every patch in a throwaway copy outside the repo | No | (a) to (h) |
| 1 | 03, 01, 02, 04 (changes 01), 05 | The runner first, then the "before" run; 2025.1 package 5 runs end to end with no keyboard input | No | -- |
| 2 | 06, 07 | Reads the dual-fuel ratings and prices the heat pump at SEER1 | Yes, 2025.1 only (heat pump about +$754) | (i) |
| 3 | 08, 09 (changes 02) | Prices the backup gas furnace | Yes, 2025.1 only (furnace about +$4,100) | (j) |
| 4 | 10, 11 (changes 03) | June 2026 rebates for a dual-fuel retrofit | Yes, 2025.1 June 2026 columns only | (k) |
| 5 | 12, 13, 14 (changes 04), 15 | Peak column names, Tepper export columns, package 5 title, two input checks | No | (l) |
| 6 | none | National runs for both releases; new reference rows and CLAUDE.md text | No (documents only) | (m) |

- Do the sessions in order. None can be skipped: later patches build on earlier ones
  (for example, patch 15 does not apply without 10 and 11).
- You approve one file at a time: 45 approvals in all, 7 to 11 per session. For a
  closer look, say `hunk by hunk` (about 110 approvals in all) and `file by file` to go
  back. When you are comfortable, `auto-approve <scope>` lets Claude go on without
  waiting (6.2). It is never on unless you type it.
- No 2022.1.1 value may move. Claude checks this with a Pennsylvania run at the end of
  every session and a national run in Session 6.

**How long things take** (cloud machine, Linux, 4 cores; yours may be slower):

| Step | Time |
|---|---|
| Test suite | 10 to 15 seconds |
| 2025.1 Pennsylvania run | about 1.5 minutes, 3.6 GB of memory |
| 2025.1 national run | about 11 minutes, 10.9 GB |
| 2022.1.1 runs | never measured; Session 1's "before" run and Session 6 give the first times |
| Session 0 | roughly 30 to 60 minutes |
| Sessions 1 to 5 | roughly 1 to 3 hours each, mostly your reading |
| Session 6 | at least 30 minutes of runs, plus the documentation diffs |

Keep the laptop plugged in, and stop it from sleeping during national runs.

---

## 3. One-time setup (Git Bash)

1. Open Git Bash and go to the repo root, the folder that holds `config.py` and
   `CLAUDE.md`:

   ```bash
   cd "/c/path/to/cmu-tare-model"
   ```

2. Make sure you have nothing uncommitted:

   ```bash
   git status
   ```

   You want "nothing to commit, working tree clean". If you have work in progress,
   commit it or put it aside first, in whatever way you normally do.

3. Check out the dev branch and bring it up to date:

   ```bash
   git checkout resstock2025-dual-fuel-codebase-update
   git pull
   ```

   If you would rather port onto a new branch, create it now from this one. Session 0
   asks you which you chose.

4. Unpack the bundle at the repo root. Give the path in Git Bash form (`~/Downloads/...`
   or `/c/Users/...`); a `C:\...` path confuses `tar`:

   ```bash
   tar -xzf ~/Downloads/cloud_run_bundle.tar.gz
   ls cmu_tare_model/docs/cloud_run/patches/*.patch | wc -l        # 15
   ls cmu_tare_model/docs/cloud_run/notebook_changes/*.json | wc -l # 4
   ls cmu_tare_model/docs/cloud_run/*.md | wc -l                    # 7
   ls cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md cmu_tare_model/docs/cloud_run/LOCAL_PORT_INSTRUCTIONS.md
   git status --short     # only: ?? cmu_tare_model/docs/cloud_run/
   ```

   You can use 7-Zip instead: open the `.tar.gz`, open the `.tar` inside it, and extract
   the `cmu_tare_model` folder into the repo root.

5. Confirm the two ResStock 2025.1 files are present:

   ```bash
   ls -l data/resstock_2025_1/upgrade0.parquet data/resstock_2025_1/upgrade5.parquet
   ```

6. Activate the project environment and check it:

   ```bash
   source /c/Users/jorda/AppData/Local/anaconda3/etc/profile.d/conda.sh
   conda activate cmu-tare-model
   python -c "import sys; print(sys.executable)"   # should end in envs\cmu-tare-model\python.exe
   python -c "import pandas, numpy; print(pandas.__version__, numpy.__version__)"
   python -c "import pyarrow, IPython, nbformat, psutil; print('ok', pyarrow.__version__)"
   echo "${TARE_RESSTOCK_RELEASE:-unset}"           # should print: unset
   ```

   - If `pyarrow` is missing, install it now, before Session 0, without touching pandas
     or numpy: `python -m pip install "pyarrow==23.0.1"` (the cloud's version; or
     `conda install pyarrow --freeze-installed`). Install a missing `nbformat` or
     `psutil` the same way. Then rerun the pandas/numpy line: it must print the same
     versions as before (your environment file pins 2.1.4 and 1.26.4).
   - Why now: the starting test count and the "before" run should use the same packages
     as every later run. Without pyarrow nothing runs once patch 02 is in, not even
     2022.1.1.
   - If `TARE_RESSTOCK_RELEASE` is set, run `unset TARE_RESSTOCK_RELEASE`. Also remove
     it from your Windows environment variables if it is set there.

7. In VS Code, close any open tab of the three notebooks:
   - `tare_model_main_v3_0.ipynb`
   - `tare_run_simulation_v3_0.ipynb`
   - `tare_scenarios_v3_0.ipynb`

8. Make Claude's scratch folder, outside the repo:

   ```bash
   mkdir -p ~/tare_port
   ```

   Optional backstop: add deny rules to your user-level settings,
   `~/.claude/settings.json` (not the repo's). They block these commands for Claude
   only, never for you. Merge them into any `permissions.deny` list already there:

   ```json
   {"permissions": {"deny": ["Bash(git add:*)", "Bash(git commit:*)",
     "Bash(git push:*)", "Bash(git reset:*)", "Bash(git checkout:*)",
     "Bash(git switch:*)", "Bash(git restore:*)", "Bash(git stash:*)",
     "Bash(git pull:*)"]}}
   ```

---

## 4. Start Claude Code (VS Code extension) in auto mode

Open the repo folder in VS Code, start a new Claude Code conversation and set it to auto
mode. Then, in every new conversation, type this as its own first message:

```
/add-dir C:\Users\jorda\tare_port
```

The access lasts for that conversation only. Claude checks for it at the start of every
session and asks if it is missing. Three things specific to this port:

- **Auto mode removes permission prompts only.** The kickoff prompt and CLAUDE.md still
  make Claude stop after every proposed file and wait for your reply, unless you turn
  on auto-approve yourself (6.2).
- **You review in the chat, not in the extension's diff view.** Auto mode skips that
  view, and changes applied with `git apply` never appear in it. Source Control shows
  each change before you commit. If Claude ever writes a change you did not approve,
  stop it, undo the change (section 7), and tell Claude "I restored <file>". Do not
  use rewind for this: it does not undo `git apply` changes.
- **No need to start VS Code from an activated terminal.** Each of Claude's commands
  starts a fresh shell and inherits nothing from your terminal, so Claude loads the
  `cmu-tare-model` environment itself at the start of every command, from
  `~/tare_port/env.sh`. Session 0 writes that file and shows it to you.

If Claude Code ever saves a permission to `.claude/settings.local.json`, that file shows
as `??` in `git status`. Never stage it. Delete `~/tare_port` yourself when the port is
finished.

---

## 5. Session 0 -- audit and plan

Type the `/add-dir` line (section 4) as your first message. Then paste this line:

```
Read cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and do Part 1 (Session 0: audit and plan) only. Follow CLAUDE.md, write nothing in the repo except the plan file after I approve it, install nothing, and stop at every gate.
```

What Claude does (it reads and checks only; nothing in the model changes):

1. Reads the cloud documents, the skill's instructions and the reference values.
2. Checks that the bundle holds all 15 patches unchanged (by checksum).
3. Checks your branch and lists any commits on it since `b17d7af`, noting which touch a
   file the patches touch.
4. Writes `~/tare_port/env.sh`, which loads your environment for its commands, and
   checks each touched file's line endings, your Python environment and your data.
5. Tries every patch and notebook change in a throwaway copy in `~/tare_port`, and checks
   in your repo, writing nothing, that each file's first change fits.
6. Runs the test suite once, to record your starting count.
7. Reports what it found and asks questions (a) to (h) (section 10).

Answer all eight in one reply, for example `(a) new branch (b) untracked (c) yes ...`.
For (d) only confirm one approval per file, and say whether the `.patch` file in VS Code
is enough for the long changes.
Section 10 suggests an answer for each if you are unsure.

Claude then shows you the text of a new file,
`cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md`, and writes it only after you
approve. It holds the audit findings, your answers, the changes for each session in
order, and a ready-to-paste line for each of Sessions 1 to 6. Session 0 ends with the
remaining questions, (i) to (n), and the session each must come before.

**Session 0 worked if:**

- all 15 checksums matched;
- the branch is the dev branch or made from it, and `b17d7af` is in its history;
- the tree is clean apart from `?? cmu_tare_model/docs/cloud_run/` (and perhaps
  `.claude/settings.local.json`), before and after;
- `~/tare_port/env.sh` loads `cmu-tare-model`, and the environment has pyarrow;
- both 2025.1 parquet files are present;
- the trial table has 45 rows, each OK or offset, and any FAILED row is explained and
  has a hand-port step in the plan;
- the test count is recorded (the cloud had 352 passed, 1 skipped);
- the plan file exists, and `~/tare_port/PORT_PROGRESS.md` exists.

---

## 6. Sessions 1 to 5 -- applying the changes

### 6.1 Starting a session

1. Start a new Claude Code conversation in auto mode, and type the `/add-dir` line
   (section 4) as its first message. A fresh conversation keeps Claude on one session's
   instructions.
2. Paste that session's line from `LOCAL_PORT_SESSION_PLAN.md`. It reads like this,
   with the session number filled in:

   ```
   Read Part 2 of cmu_tare_model/docs/cloud_run/LOCAL_KICKOFF_PROMPT.md and Session N of cmu_tare_model/docs/cloud_run/LOCAL_PORT_SESSION_PLAN.md, then follow them. Stop at every gate.
   ```

   If a decision is due (section 2 table), add it to the line, for example
   ` Decision (i): keep 0.95 and 0.85.` Claude logs it and shows you the one-line
   update to the plan for approval.
3. Claude first checks that the last session's commits are in and that nothing is left
   uncommitted. It reports anything unexpected before going on.

### 6.2 Each change

Claude shows you one file per turn (one hunk per turn after `hunk by hunk`):

- the patch and the file (and, hunk by hunk, which hunk: for example "hunk 2 of 7");
- the exact lines that will change, with context;
- git's dry-run result, which says whether the change fits your file as it is now;
- a plain-language note on what the change does and why;
- whether any modeled value moves;
- any added line that departs from CLAUDE.md's coding standards (for example decision
  IDs in comments); it is changed only if you ask.

Then it stops and waits. For long changes, you can also open the `.patch` file in
VS Code (`cmu_tare_model/docs/cloud_run/patches/`), which colors added and removed
lines.

Reply with one of these:

- `approved`: approves this file only. "Approve the rest" is not an approval; Claude
  asks which auto-approve scope you mean.
- `auto-approve this patch`, `auto-approve until the next commit point`, or
  `auto-approve this session`: Claude still shows each file and its dry run, but applies
  it without waiting. It still stops at every commit point for you to commit (only
  `this session` carries on after `committed`), and it stops and switches auto-approve
  off at anything unusual: a dry run that is not clean, a change that would need a hand
  edit, a failed test, run or check, a decision, a plan edit, or a notebook skill stop.
  It is never on for Session 6's documentation edits, and it ends with the conversation.
  `this patch` ends after that patch's last file, even when its commit is shared with
  the next patch. Any other reply (`stop`, Esc, `skip`, `change:`, "I restored ...",
  `hunk by hunk`), a usage-limit pause or a compacted conversation also switches it
  off; type it again to turn it back on. Type `stop auto-approve` to end it sooner.
  Claude never offers it, and a "yes" to Claude's questions never turns it on.
- `hunk by hunk` or `file by file`: change the approval unit from now on.
- a question.
- `change: <what>`: Claude makes that file by hand edits instead, shown to you as a new
  diff. Later patches on the same file may then not apply, and section 11's numbers may
  not hold. It is better to approve as written and make improvements after Session 5,
  in separate commits. Ask for a change during the port only when the patch does not
  fit your branch.
- `skip`: a pause only. A half-applied patch cannot be committed, and later patches on
  that file fail. Come back to it before the commit point.
- `stop` (or press Esc).

Claude writes only what you approved (or what an auto-approve you typed covers): the
whole file once it is approved (or each unit as you approve it, if you answered no to
(c)). It then shows the line
counts, which must match the patch, and confirms that the file's line endings did not
change. VS Code's Source Control view shows the change side by
side. Until you commit, you can still undo it (section 7).

### 6.3 Each commit point (end of each patch)

1. Claude runs the full test suite and compares it with the expected count
   (section 11.1). After patch 01 it also runs one quick check. Checks that need a model
   run wait for the end of the session.
2. Claude lists exactly which files belong in this commit and suggests a message.
3. You review the changes in VS Code Source Control.
4. Commit only those files:
   - VS Code: click `+` beside each listed file; type the subject line, a blank line,
     then the body; click Commit. If VS Code offers to stage all your changes, answer
     No or Cancel.
   - Or Git Bash: `git add <those files>`, then `git commit -m "subject" -m "body"`,
     then `git show --stat HEAD` to check.
   - Never stage `cmu_tare_model/docs/cloud_run/` or `.claude/settings.local.json`.
5. Tell Claude `committed`. Claude checks the last commit holds the right files, then
   goes on.

Claude never stages, commits, restores or pushes; those steps are yours.

### 6.4 Notebook changes (patches 04, 09, 11, 14)

- These go through the notebook-cell-edit skill, using changes files 01 to 04. Claude
  shows you the skill's dry-run diff for the whole changes file in one approval (one
  cell per approval after `hunk by hunk`), and the skill writes the notebook once it
  is approved.
- After Claude applies a notebook change, close and reopen that notebook's VS Code tab
  (or use "Revert File") before you run or save it. Otherwise VS Code can save the old
  text back over the change.
- Two pairs must go into the same session, with no 2025.1 run in between:
  - patch 08 with notebook change 02 (patch 09);
  - patch 10 with notebook change 03 (patch 11).

  Between the halves of a pair a 2025.1 run is expected to fail (section 11.3).

### 6.5 End of a session

Claude runs the Pennsylvania checks for both releases, in the background, and compares
them with section 11:

- 2025.1: `python scripts/run_tare_notebooks.py --release 2025.1 --state PA --skip-grid-impact`
- 2022.1.1: the same command with `--release 2022.1.1`. Its outputs must equal the
  "before" run that Session 1 makes.

Do not run a notebook or edit files yourself while a run is going.

What the runner's exit code means:

- 0: the run and all checks passed.
- 1 or 2: a notebook cell failed, the runner could not start, or a notebook asked a
  question the runner does not know. Claude reports it, and proposes any fix as a
  normal gated diff.
- 3: the run finished but a check failed. Stop and decide with Claude.
- A run cut off by a timeout or a restart is not a result. Run it again.

A 2022.1.1 difference is always a stop. In Sessions 1 (patch 05) and 5 (patch 15) it may
be a real finding, since the cloud had no 2022.1.1 data to test those changes: decide
with Claude before committing anything more.

**A session is done when** every commit is made, the tests are at the expected count,
the 2025.1 Pennsylvania numbers match section 11.2, and 2022.1.1 equals the "before"
run.

**Where results go.** Runs write to `cmu_tare_model/output_results/` (`baseline_summary/`,
`retrofit_mpN_results/`, `supplemental_data_damages/`, `supplemental_data_fuelCosts/`,
`tepper_export/`), with names like `baseline_results_PA_2026-10-06_18-14.csv`. The
release is not in the name, so Claude records which stamp is which release in
`~/tare_port/PORT_PROGRESS.md`. Runner logs go to `~/tare_port/logs/`. The folder is
git-ignored; the cloud's ten or so runs filled 1.2 GB.

---

## 7. Undo one file before committing

Run these yourself; Claude does not run them.

- A changed file: `git restore <path>`
- A new file the patch created: delete it, with `rm <path>` or in VS Code.
- A notebook: `git restore <path>`, then close and reopen its VS Code tab.

Then tell Claude "I restored <path>", so its progress log stays right.

`git restore` puts the whole file back to your last commit. If the file holds
uncommitted changes from two patches, both go. That is one reason to commit after every
patch.

To undo a commit you have already made: `git revert HEAD` (or another git step of your
choosing), then tell Claude. Claude does not do it.

---

## 8. If a change does not apply

- Claude stops and shows you what the patch expected beside what your file holds now.
  Usually this means the dev branch changed that file after `b17d7af`.
- Claude then proposes a small hand edit that does the same thing, as a normal diff for
  your approval, and flags later patches that touch the same file.
- If git says a change applied "with offset", the file has moved since `b17d7af`. Claude
  reports this and checks that the moved lines do not conflict before asking you.
- Never force a patch. That means no `--reject`, no `--ignore-whitespace`, and no
  editing of the `.patch` files.
- If a file has mixed line endings (some lines CRLF, some LF), Claude stops and asks you
  what to do.
- The notebook skill can stop with two messages:
  - "does not read as expected": the notebook changed. Claude rebuilds the change from
    the notebook as it is now and shows it to you again.
  - "cannot be written back byte for byte": usually VS Code re-saved the notebook in a
    different layout. Claude gives you the change to make by hand (cell, line, old
    text, new text) and stops. It never applies a notebook patch with git.

---

## 9. Session 6 -- final checks, and after the port

Start a new conversation in auto mode, type the `/add-dir` line (section 4), then paste
the Session 6 line from the plan file. Claude then:

1. Runs 2025.1 nationally (about 11 minutes and 10.9 GB on the cloud machine; close
   other programs first) and compares the result with section 11.4:

   ```bash
   python scripts/run_tare_notebooks.py --release 2025.1 --skip-grid-impact --log ~/tare_port/logs/s6_national_2025.log
   ```

2. Runs 2022.1.1 nationally with `--release 2022.1.1`. Add `--skip-grid-impact` unless
   your AWS access is set up. No national 2022.1.1 run has been made through the runner,
   so its time and memory are unknown. Every output must equal your pre-port run, or
   the `cmu_tare_model/docs/REFERENCE_VALUES.md` rows for runs `2026-10-05_00-54` /
   `2026-10-05_21-33` (section 11.5).
3. Runs the test suite one last time: your starting count + 47.
4. Proposes, one gated diff at a time:
   - new rows in `cmu_tare_model/docs/REFERENCE_VALUES.md` for 2025.1 package 5, from
     your own run. Rows are only added; no existing row is changed. The cloud numbers
     are provisional and are not used for these rows.
   - the four CLAUDE.md blocks under "Proposed CLAUDE.md text" in the implementation
     guide, adjusted to your decisions;
   - a `cmu_tare_model/docs/SESSION_LOG.md` entry.

**Switching release, from Session 1 on.** Editing `constants.py` no longer switches the
release; the variable `TARE_RESSTOCK_RELEASE` does, and unset means 2022.1.1.

- For a 2025.1 run, use the runner with `--release 2025.1`.
- For 2025.1 in VS Code notebooks: close every VS Code window, then start VS Code from
  Git Bash with `TARE_RESSTOCK_RELEASE=2025.1 code .`. Start it normally to go back.
  Claude's own commands are not affected: `env.sh` unsets the variable.
- Never set it in your Windows settings or a shell profile.

**When you are done:**

- push the branch when you are ready;
- settle (b): what happens to the `cloud_run` folder;
- add pyarrow to `environment-cmu-tare-model.yml` and `requirements.txt`, as its own
  change;
- do the by-hand notebook items, (n);
- note for the 2025.1 grid-impact work: `grid_impact/build_parcel_frame.py` still lists
  the 2022.1.1 peak column names (implementation guide, patch 12);
- remove the optional deny rules (step 3.8) if you want, and delete `~/tare_port`.

---

# Reference (sections 10 to 12): for Claude, and for checking results

## 10. Decisions to make, and when

Session 0 asks (a) to (h); you can answer any question early. Each ends with a
suggestion if you are unsure. For (i) to (m), add your answer to that session's start
line (section 6.1).

**Before Session 1**

- (a) Port onto `resstock2025-dual-fuel-codebase-update` directly, or onto a new branch
  made from it? If unsure: a new branch, which is easy to throw away.
- (b) Where the `cloud_run` folder lives.
  - The guide says not to port it as model code. During the port, leave it untracked
    (it shows as `??` in `git status`; do not stage it).
  - Afterwards there are three options: commit the five documents, keep the folder
    untracked, or move it out of the repo.
  - Code comments cite R1, R3 and G-numbers (defined in the cloud session prompt,
    `cmu_tare_model/docs/session_prompts/CLOUD_SESSION_PROMPT_2026-10-06_ResStock2025_1_MP5.md`,
    which a cloud-only commit added; check whether your branch has it), D8
    (`NEXT_STEPS_2026-09-19_DualFuel_ResStock2025_1.md`), and P1 and P5 (only in
    `DECISIONS_TAKEN.md`). Whatever you commit should let a reader find them.
  - If you ever commit the `.patch` files, first add `*.patch -text` to a
    `.gitattributes` file, so git does not convert their line endings.
  - If unsure: leave it untracked until Session 6.
- (c) Applying an approved file with `git apply --include=<file>` writes exactly the
  text you reviewed. Is that acceptable as "the edit"? CLAUDE.md names the Edit tool.
  If you say no, Claude makes each approved unit with the Edit tool right away, keeping
  your line endings, then checks the file against the patch. If unsure: yes.
- (d) How you approve. Already set, by your choice: one approval per file (45 in all).
  - This loosens CLAUDE.md's "across functions" rule for this port; one approval never
    covers two files. Claude records it in the plan.
  - Say `hunk by hunk` at any time for one hunk per approval (about 110 in all), and
    `file by file` to go back.
  - Auto-approve is off unless you type it (6.2).
  - For long changes, is reading the `.patch` file in VS Code enough, or must everything
    be printed in the chat? The long ones are the runner (03, 677 new lines),
    `data_loading.py` (05, +159 / -99 lines) and its new test (181), the furnace code in
    `remdb_v4_installed_cost_utils.py` (08, 158) and its new test (129), and notebook
    change 01 (113).
  - If unsure: everything printed.
- (e) Commits:
  - One commit per patch? This is needed in Sessions 2 and 5, where two patches edit
    the same file.
  - Commit 08 and 09, and 10 and 11, separately or together? A commit holding 08 or 10
    alone is one where 2025.1 does not run.
  - If unsure: one commit per patch, but 08 with 09 and 10 with 11, so that every
    commit runs.
- (f) Is a Pennsylvania run enough each session, or should MN and FL be added? If
  unsure: Pennsylvania only.
- (g) For the 2022.1.1 national "no change" check, which comparison?
  - Compare with run `2026-10-05_21-33`. This is valid only if your branch has no
    value-moving commits since that run.
  - Or make a fresh national run in Session 1, before any model patch goes in.
  - Either way, should grid impact be on (needs AWS)?
  - If unsure: the earlier run if nothing that moves values was committed since;
    otherwise a fresh run. Grid impact off.
- (h) Are the 2025.1 data files and pyarrow in place (steps 3.5 and 3.6)? This is a
  check, not a choice.

**(i) Before Session 2 -- P1**

- Keep SEER1 = SEER2 / 0.95 and HSPF1 = HSPF2 / 0.85, or use other factors?
- These factors are provisional: they were not checked against DOE Appendix M1.
- Every 2025.1 value from Session 2 on depends on them. A different factor changes the
  expected numbers below, for example the +$754.

**(j) Before Session 3 -- the furnace decisions.** Accept these?

- P5: the furnace cost is not in the rebate base.
- R1: the furnace is priced at its own backup size.
- D-S8: only a gas backup is priced; a propane or oil backup stops the run.
- P8 / D-S7: the column name is `mp5_heating_backupFurnace_installed_cost_v4MID`.

**(k) Before Session 4 -- O-1, and the Program Notice 26-3 comment**

- With the cloud's code, the June 2026 rebate rules equal the 2024 rules for the
  dual-fuel package. Adoption is identical (29.04% nationally, heatingLCC_coolingLCC),
  and the totals differ by about $424 of rounding. Is that what you intend?
- If not, setting `dual_fuel_passes_fuel_gates` to False for June 2026 is not enough.
  Notebook change 03 (cell 30), `scripts/verify_june2026_rebate_fossil_gate.py` and the
  dot plot's rule all decide from `is_dual_fuel_package` and would need matching edits,
  so the Session 4 diffs would change.
- If you are not ready to decide, stop after Session 3, or apply patch 10 as written and
  change the rule later in its own commit. Session 5 cannot go in without Session 4.
- Also: keep the neutral Program Notice 26-3 comment that patch 10 adds to
  `constants.py`, or wait until you have reviewed 26-3?

**(l) Before Session 5.** Accept these as written?

- the 2025.1 peak names `..._peak_electricity_winter_kw` / `..._summer_kw` (P7);
- the 13 extra Tepper columns for package 5 (D-S11);
- the title "Dual-fuel heat pump with gas backup furnace" (D-S12).

**(m) Before Session 6.** Apply the four proposed CLAUDE.md blocks? Add new
REFERENCE_VALUES rows for package 5 from your national run?

**(n) After the port, by hand:**

- move the package 5 block out of the MP4 export cell of the simulation notebook into
  cells of its own;
- update the notebook text the cloud left alone;
- regenerate the `*_EXPORT_*.py` snapshots. Claude never edits those.

---

## 11. Expected numbers

Cloud numbers, measured on Linux. If your branch has new commits since `b17d7af`, the
absolute test counts shift; compare the change from your Session 0 count instead.

### 11.1 Test suite after each patch, in the order applied

Command, from the repo root:
`env -u TARE_RESSTOCK_RELEASE python -m pytest -q -p no:cacheprovider`

Warnings are not compared; the cloud had 19. BREAKAGE_LOG's Stage B "Suite" column
counts `cmu_tare_model/tests` only, 5 fewer; use this table.

| After | Passed (cloud) | Skipped | Change from start |
|---|---:|---:|---:|
| start (Session 0) | 352 | 1 | +0 |
| 03 (first in Session 1) | 352 | 1 | +0 |
| 01 | 355 | 1 | +3 |
| 02 | 359 | 1 | +7 |
| 04 (notebook change 01) | 359 | 1 | +7 |
| 05 -- end of Session 1 | 367 | 1 | +15 |
| 06 | 370 | 1 | +18 |
| 07 -- end of Session 2 | 375 | 1 | +23 |
| 08 | 385 | 1 | +33 |
| 09 (notebook change 02) -- end of Session 3 | 385 | 1 | +33 |
| 10 | 389 | 1 | +37 |
| 11 (notebook change 03) -- end of Session 4 | 389 | 1 | +37 |
| 12 | 392 | 1 | +40 |
| 13 | 393 | 1 | +41 |
| 14 (notebook change 04) | 393 | 1 | +41 |
| 15 -- end of Session 5 | 399 | 1 | +47 |

The one skip is `test_kpi_functions.py` ("Stale: ... rewrite pending") and appears every
time.

### 11.2 Pennsylvania, 2025.1, at the end of each session

Command:
`python scripts/run_tare_notebooks.py --release 2025.1 --state PA --skip-grid-impact`

Every run should:

- exit 0 and print `[OK] All 20 checks passed.`;
- show a study sample of 6,469 rdu (1,642,503 homes) in 66 counties, at weight
  253.90367272727272;
- print `home_count agrees in all 66 counties`;
- show a mean heating replacement cost of $3,708.01.

Adoption is shown as unsub / sub / sub_june2026, in %.

| | Session 1 (01-05) | Session 2 (06-07) | Session 3 (08-09) | Session 4 (10-11) | Session 5 (12-15) |
|---|---|---|---|---|---|
| "Backup furnace:" line | none | none | 6,469 valid rdu, mean $4,153.36 | same | same |
| Mean heat-pump upgrade cost | $13,690.29 | $14,444.36 | $14,444.36 | $14,444.36 | $14,444.36 |
| heatingLCC_coolingLCC | 7.51 / 50.53 / 7.70 | 5.33 / 47.13 / 5.52 | 2.89 / 21.61 / 3.08 | 2.89 / 21.61 / 21.61 | same as Session 4 |
| same, adopters (rdu) | 486 / 3,269 / 498 | not measured (do not compare) | 187 / 1,398 / 199 | 187 / 1,398 / 1,398 | same as Session 4 |
| heatingSavings_coolingLCC | 3.31 / 30.27 / 3.49 | 3.06 / 26.29 / 3.28 | 1.92 / 4.17 / 2.23 | 1.92 / 4.17 / 4.17 | same as Session 4 |
| heatingLCC_coolingSavings | 2.80 / 15.01 / 3.03 | 2.60 / 11.24 / 2.89 | 1.59 / 3.01 / 1.90 | 1.59 / 3.01 / 3.01 | same as Session 4 |
| "no blank ... value" check | 37 columns | 37 columns | 38 columns | 38 columns | 38 columns |
| June 2026 check line | old wording (1) | old wording (1) | old wording (1) | new wording (2) | new wording (2) |
| June 2026 rebate funding (weighted) | not measured (do not compare) | not measured (do not compare) | Electricity baselines only: HEEHR $105,961,414, HOMES $1,015,615 | HEEHR $7.573 billion, HOMES $475.3 million (2) | same as Session 4 |
| Tepper PA files, columns (main / detailed) | 169 / 319 | 169 / 319 | 169 / 319 | 169 / 319 | 182 / 332 |

Notes to the table:

- (1) Old wording:
  `[PASS] MP5 June 2026 fuel gate holds: fossil baselines $0, electric-resistance baselines funded.`
- (2) New wording:
  `[PASS] MP5 June 2026 fuel gate holds: fossil baselines funded (dual fuel), electric-resistance baselines funded, non-participating states $0.`
- (2) By baseline fuel, the Session 4 June 2026 funding is: natural gas $7.677 billion,
  fuel oil $260.8 million, propane $3.66 million, electricity $107.0 million.
- The Tepper PA files have 6,469 rows each. The Allegheny files have 1,091 rows.
- After patch 06 alone, every number equals Session 1.
- On the cloud each PA run took about 80 seconds and peaked near 3.6 GB.

### 11.3 Runs that are expected to fail

Do not run 2025.1 at these points, and do not treat these failures as bugs.

| State | What happens |
|---|---|
| 08 in, 09 (notebook change 02) not yet | Exit 1. `tare_scenarios_v3_0.ipynb`, cell 24: `KeyError ... ['mp5_heating_backupFurnace_installed_cost_v4MID']` |
| 10 in, 11 (notebook change 03) not yet | Exit 1. `tare_scenarios_v3_0.ipynb`, cell 30: `AssertionError: MP5 June 2026 fuel-gate regression ...` |

The reverse orders of these pairs fail too.

### 11.4 National, 2025.1

Every national run should show 161,983 rdu (41,128,079 homes), exit 0 and
`All 20 checks passed`. On the cloud each took about 11 minutes, with a peak of about
10.9 GB in main notebook cell 23.

| | After Session 1 (optional) | After Session 3 (optional) | Final (Session 6) |
|---|---|---|---|
| Mean heat-pump upgrade cost | $14,957.16 | $15,711.23 | $15,711.23 |
| Mean backup furnace cost | not priced | $4,112.21 (median $4,069; $3,646-$6,567) | $4,112.21 |
| Mean total capital (2024 rebate netted) | $10,082 | $14,864 | $14,864 |
| heatingLCC_coolingLCC adoption (unsub / sub / sub_june2026) | 11.19% / 60.23% / 12.42% | 3.27% / 29.04% / 4.71% | 3.27% / 29.04% / 29.04% |
| June 2026 rebate recipients | 5,732 rdu | 5,732 rdu | 121,432 rdu |

For the final run, `PROVISIONAL_RESULTS_2026-10-07.md` (section "Final run in detail")
has every other number: the funnel, costs, rebates by fuel, all nine NPV cases and
climate.

### 11.5 2022.1.1 (must not move)

No 2022.1.1 run of the runner has been made anywhere: the cloud had no 2022.1.1 data.
The figures below for the runner are read from its code, not measured.

- **Pennsylvania, every session.** Every output must be identical to the Session 1
  "before" run (rows, columns and values; only the run stamp in file names differs).
  - The copies in `tepper_export/source_data/` are inputs with no stamp; they are not
    compared.
  - The check count should be 20 per package, so about 40 for MP3 + MP4. The "before"
    run's count is the one to match.
  - From Session 4 on, the MP3 and MP4 `[PASS] ... June 2026 fuel gate holds` line ends
    with ", non-participating states $0.". The values behind it are unchanged.
- **National (Session 6).** Every output must equal your pre-port run, or these
  `cmu_tare_model/docs/REFERENCE_VALUES.md` rows:
  - 221,205 rdu;
  - heatingLCC_coolingLCC adoption: MP3 18.8088% / 45.8308% / 26.3394%; MP4 18.9914% /
    47.8407% / 28.4189%;
  - demand change at 100% adoption: +278,176.6 GWh (MP3) / +1,278.6 GWh (MP4);
  - with grid impact on: Allegheny baseline peak 714.51 MW.

---

## 12. If something goes wrong

### 12.1 A session is interrupted

- **The conversation is still open** (a usage limit, or you pressed Esc): type
  `Re-read the kickoff prompt (Part 1 for Session 0, Part 2 otherwise) and ~/tare_port/PORT_PROGRESS.md, then continue from the last gate.`
  Any auto-approve you had typed is off; type it again if you want it.
- **Claude says it compacted the conversation:** type the same line before the next
  approval.
- **The computer restarted or the conversation closed:** open the repo in VS Code, look
  at `git status`, start a new conversation, type the `/add-dir` line (section 4), and
  paste the session's line with ` This is a resume.` added. Claude checks that the
  uncommitted files are exactly the ones its progress log records, and goes on from
  there. Any auto-approve you had typed is off again.
- **A run was cut off:** run it again. Its partial outputs carry their own stamp and are
  ignored.

### 12.2 Other problems

- **The runner fails at start-up on Windows.** It has never run on Windows. Try these
  in order:
  - If you get a console or prompt error, run it from VS Code's terminal or Anaconda
    Prompt, or put `IPY_TEST_SIMPLE_PROMPT=1 ` in front of the command.
  - If you get a text-encoding error, put `PYTHONUTF8=1 ` in front of the command.
  - If `%run` cannot find the notebook, Claude proposes a one-line fix to the runner
    as a normal gated diff.
- **The "before" run fails.** It is the first 2022.1.1 run of the runner anywhere, so
  the cause is the runner or Windows, not a patch. Claude reports it and stops before
  unit 2.
- **`No module named 'pyarrow'`** after patch 02: install pyarrow (step 3.6).
- **The wrong release runs.** `TARE_RESSTOCK_RELEASE` is set somewhere. The runner's
  `[runner] release ...` line says which release it is using. Tests assume the variable
  is unset. A Jupyter kernel reads it when it starts, so restart the kernel after
  changing it.
- **Out of memory.** Close other programs and run one thing at a time.
- **Grid impact on 2025.1.** It is not supported yet: always pass `--skip-grid-impact`
  for 2025.1.
- **Run outputs.** `output_results/`, `figures/` and the PNG files a run writes are in
  `.gitignore`, so they do not show up for commit.
