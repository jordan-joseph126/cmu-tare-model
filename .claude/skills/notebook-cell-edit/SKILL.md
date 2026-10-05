---
name: notebook-cell-edit
description: Change lines inside Jupyter notebook (.ipynb) cells exactly and safely, using the bundled script that edits only the named lines and stops unless it can prove nothing else in the file moves. Use this skill for EVERY change to a notebook cell in this repo -- removing an override such as VERBOSE = True, fixing a printed label, changing or dropping an argument, adding or deleting a line, carrying a module change over into a notebook, or anything the researcher calls a "paste cell" or a "backport". Use it instead of retyping a cell by hand, replacing a whole cell with the NotebookEdit tool, or editing the notebook JSON with a one-off script. Also use it to find which cell and line of a notebook holds some text.
---

# Notebook cell edit

## Why this exists

A notebook is one large JSON file. Two things go wrong when a cell is changed
the obvious ways:

- **Retyping a cell to change one line.** A 100-line cell means retyping 99
  lines that were not supposed to change. One slip in a divider line or an
  indent goes into the notebook unseen.
- **Rewriting the file with a general tool.** If the tool lays the JSON out
  differently, git shows hundreds of changed lines and the real change is
  buried.

The bundled script avoids both. It changes only the lines it is told to, and
it refuses to write unless it can show that nothing else in the file moves.
The researcher reviews notebook changes line by line in `git diff`, so "only
the intended lines changed" is the property that matters.

The researcher allowed direct notebook edits on 4 Oct 2026. The earlier
"never edit a notebook" rule was written for a different assistant that had
trouble reading and editing notebook cells.

## The procedure

Run every command from the repo root. The script uses the standard library
only, so any Python 3.9+ works; on the researcher's machine use
`C:\Users\jorda\AppData\Local\anaconda3\python`.

```
SCRIPT=.claude/skills/notebook-cell-edit/scripts/edit_notebook_cells.py
```

**Step 1 -- find the cell and the line.** Do not guess positions and do not
read the whole notebook into context.

```
python $SCRIPT list  NOTEBOOK                 # one row per cell
python $SCRIPT list  NOTEBOOK --grep "TEXT"   # only cells and lines holding TEXT
python $SCRIPT show  NOTEBOOK CELL --json     # one cell, each line as a JSON string
```

Cells are numbered from 1, counting markdown and code cells together. This is
how the researcher counts ("the 16th cell"). Lines are numbered from 1 inside
a cell. `show --json` prints each line already escaped, so it can be copied
straight into the `old` field below; it also makes trailing spaces visible.

**Step 2 -- write a changes file** in the session's scratch folder, never in
the repo.

```json
{"changes": [
  {"notebook": "cmu_tare_model/model_scenarios/tare_scenarios_v3_0.ipynb",
   "cell": 16, "line": 14,
   "old": "VERBOSE = True",
   "new": null},
  {"notebook": "cmu_tare_model/tare_model_main_v3_0.ipynb",
   "cell": 9, "line": 56,
   "old": "print(f\"Baseline: {len(df_baseline):,} occupied SF homes\")",
   "new": "print(f\"Baseline: {len(df_baseline):,} occupied SF rdu\")"}
]}
```

- `old` is the line exactly as it reads today, without its line break.
- `new` is the replacement line, `null` to delete the line, or a list of
  lines to put in its place. To add a line after line N, replace line N with
  `[line N as it is, the new line]`.
- Every `cell` and `line` refers to the notebook as it is on disk now. Several
  changes in one cell do not shift each other's numbers.
- One file can hold changes for several notebooks.

**Step 3 -- dry run, then stop for approval.**

```
python $SCRIPT check CHANGES.json
```

This prints a diff for each changed cell and writes nothing. Show that diff
to the researcher and wait for approval, as CLAUDE.md's stop-gate rule asks
for every edit. Notebook edits are not exempt.

**Step 4 -- apply.**

```
python $SCRIPT apply CHANGES.json
```

It runs the same checks again, copies each notebook as it was to a dated
folder in the system temp folder (or `--backup-dir DIR`), then writes. Either
every notebook in the file is written or none is.

**Step 5 -- confirm with git.**

```
git diff --stat -- '*.ipynb'
```

The counts should match the changes file: one line added and one removed for
each replaced line, one removed for each deleted line. Report them.

**Step 6 -- tell the researcher to reload open tabs.** VS Code keeps an open
notebook in memory. If a tab of the edited notebook is open, it may still
show the old text, and saving that tab writes the old text back over the
edit. Ask the researcher to close and reopen the tab, or use "Revert File",
before running or saving. This cannot be detected from disk, so say it every
time.

## What the script checks before it writes

1. **The file writes back byte for byte.** It rewrites the unchanged notebook
   in memory and compares it with the file. Only if they are identical is an
   edit safe, because then the only bytes that can differ afterwards are the
   edited lines.
2. **Each named line reads exactly as `old` says.** This catches a notebook
   that changed since it was last looked at.
3. **A code cell that is valid Python stays valid Python.** Lines starting
   with `%` or `!` are set aside for this check.
4. **Nothing else differs.** It re-reads the planned file and confirms every
   other cell, and everything outside the cells, is equal to the original.

It also notes any new line that holds non-ASCII characters, since CLAUDE.md
asks for ASCII only in notebook cells.

## When the script stops

A stop always means nothing was written.

- **"cannot be written back byte for byte"** -- do not work around it. Give
  the researcher the changes to make by hand (cell, line, old text, new text)
  and say why.
- **"does not read as expected"** -- the notebook is not what you thought.
  Run `show` on the cell again and rebuild the changes file from what is
  there. Never loosen `old` just to make it match.
- **"would not be [valid Python] after the change"** -- usually the change is
  wrong. Pass `--skip-python-check` only when the cell is meant to hold
  something that is not plain Python.

## What this skill does not cover

- **Adding, deleting, or moving a whole cell.** The script changes lines in
  cells that exist. For a new cell, hand the researcher the cell to paste.
- **Cell outputs.** They are left exactly as they are.
- **`*_EXPORT_*.py` files.** These are read-only snapshots of notebooks and
  are still never edited. Change the notebook or the module instead.

CLAUDE.md's coding standards apply to the new lines as to any other code:
ASCII only, plain-language comments, no hardcoded scenario prefixes, row
counts labeled rdu.
