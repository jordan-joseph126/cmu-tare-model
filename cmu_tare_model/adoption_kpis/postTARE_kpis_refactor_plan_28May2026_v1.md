# Post-TARE KPIs Notebook — Refactor Plan (v1)

> **Version notes.** First version. Scopes a VSCode/Copilot refactoring session to the four
> executable refactor tasks (label edits + light cleanup) and defers the two new-build tasks
> (new replacement-scenario dot plot; county-level CSV export) to interactive design in Claude
> web, because both have model-side column blockers that must be resolved before any code is
> written.

## Your Role

You are a technical research mentor supporting a researcher with intermediate Python experience
on the TARE model. You prioritize the WHY behind design decisions alongside the WHAT, but you are
efficient — the researcher is on a one-week deadline. This document is the durable reference; the
companion `copilot_prompt_postTARE_kpis_refactor.md` is the execute-now artifact for a single
Copilot Chat session.

## Project Context

Joseph et al. 2026 evaluates technology-differentiated heat pump electrification across U.S.
counties using ResStock 2022.1.1 (EUSS, AMY2018) for the building stock, augmented with the TARE
model (Carnegie Mellon) for adoption potential. Two measure packages are in scope: **MP3**
(standard air-source heat pump, HSPF1 = 9, HSPF2 ≈ 7.5) and **MP4** (high-efficiency
variable-speed ASHP, HSPF1 = 14, HSPF2 ≈ 11.5). The Pittsburgh / Allegheny County, PA case study
(FIPS 42003) anchors county-level validation before national scale-up. This refactor works only
in the KPIs notebook: `calculate_postTARE_am_kpis_demand_bill_savings.ipynb` — bill savings, NPV,
adoption rate, choropleth maps, and the adoption-potential dot plot.

## Scope Constraint — CRITICAL

- **In scope:** ResStock 2022.1.1 (EUSS), measure packages MP3 and MP4.
- **Out of scope:** ResStock 2025.1, cold-climate heat pump (MP8), dual-fuel. If a task touches
  any of these, stop and flag.
- **This session, in scope:** Tasks 1, 2, 3, 5 (label edits, figure simplification + extraction,
  light cleanup).
- **This session, deferred:** Tasks 4 and 6 (see Appendix B). Both are new builds, not refactors,
  and both have a column-availability question that must be answered first.

## Key Principle (Non-Negotiable)

**Cosmetic and cleanup edits must not change a single numerical result.** Every task in this
session touches labels, dead code, or extraction of already-computed values. None of them touches
a `column=`, a `norm=`, a filter, or a computation. The verification contract for the whole
session is therefore simple and strong: **the printed summary blocks in the notebook must be
byte-for-byte identical before and after.** If any golden value (see Reference Values) moves, a
refactor edit accidentally reached into the data path — stop and revert.

## What Was Done Before

### Prior session (figure cleanup, pre-handoff)
- Removed duplicated dot plot and adoption-rate choropleth cells.
- Hardcoded `cost_scenario='v4MID'` in the relevant lookup.
- Moved dot plot legend from a shared `fig.legend()` to per-panel `ax.legend()` (upper right).
- Built the NPV histogram as a per-MP × 2 grid (pre-IRA vs IRA) with U.S./PA median & mean
  reference lines and a break-even line at zero.

### This planning session (Claude web)
- Read the exported `.py` + `.pdf` and located every task's exact edit site.
- Confirmed Task 1 targets the `pct_bill_change` map (Map 2), not the ratio map (Map 1).
- Discovered that the dot plot y-tick label suffixes are built **inside the imported module**
  `visuals_adoption_dotplot`, not in the notebook — this changes Task 3's mechanics (Appendix A).
- Discovered that `HEATING_MP_SUBTITLES` is the single source of SEER text feeding three figures
  plus one print block, so Task 2 is a one-dict edit.

## Attached Files

- `calculate_postTARE_am_kpis_demand_bill_savings_<latest>.py` — exported notebook (the edit target).
- `calculate_postTARE_am_kpis_demand_bill_savings_<latest>.pdf` — same notebook with cell outputs
  (source of all golden reference values below).
- `data_dictionary_resstock_2022_amy2018_release1_1.tsv` — column-name lookups (needed for the
  deferred Task 6, not for this session).

## Current Implementation Status

| Step / Cell | Refactor task | Status | Notes |
|---|---|---|---|
| Cell ~`[11]` line 338 | Task 1 — colorbar rename | Ready | One-line `cbar_label=` edit on Map 2 (`pct_bill_change`). |
| Cell `[15]` lines 523–526 | Task 2 — SEER→HSPF | Ready | Single dict `HEATING_MP_SUBTITLES`; feeds 3 figures + 1 print. |
| Cell `[16]` dot plot | Task 3a — simplify y-labels | Blocked-confirm | Label suffix built in imported module (Appendix A). |
| Cell `[16]` lines 623–633 | Task 3b — reference table | Ready | `sample_info` already holds `pct_of_sample`, `weighted_homes_millions`. |
| Whole notebook | Task 5 — light cleanup | Ready | Redundant imports, dead `cbar_ticks`, diagnostic cell `[21]`, stale comment. |
| — | Task 4 — replacement-scenario dot plot | Deferred | Cooling-only `moreWTP` column not found in dump (Appendix B). |
| — | Task 6 — county CSV export | Deferred | Per-county health damages & per-equipment retrofit counts may not exist (Appendix B). |

## Required First Action

Before editing anything: open the `.py` in VSCode, run a workspace search for `SEER` and for
`Median Bill Change`, and confirm the hit counts match this plan (one `SEER`-bearing dict at
lines 523–526; one `cbar_label='Median Bill Change (%)'` at line 338). If the counts differ, the
notebook has drifted from this plan — reconcile before proceeding.

## Tasks

### Task 1 — Rename operational-savings colorbar

**Goal.** Map 2 (the `pct_bill_change` county choropleth) has the ambiguous label
`'Median Bill Change (%)'` — a reader can't tell whether it's an annual or a lifetime quantity.
The value is in fact a lifetime operating-cost change (it derives from `median_bill_savings_ratio`,
which is built from per-building lifetime fuel costs). Rename to make the time horizon explicit.

**Edit (line 338).**
```diff
-        cbar_label='Median Bill Change (%)',
+        cbar_label='Percent change in lifetime operating costs',
```

**Why it's safe.** `cbar_label` is handed to the colorbar's `.set_label()`; there is no data path
from it back to the plotted `column`, the `norm`, or the `cmap`.

**Validation gate.** Re-run the map cell. The Map 2 colorbar reads the new label; the
`--- Summary: pct_bill_change ---` block is unchanged (MP3 `med=-38.5%`, MP4 `med=-60.6%`).

### Task 2 — Equipment label sweep (remove SEER, use HSPF)

**Goal.** Apply the no-SEER-on-heating-figures rule. The only SEER text in the notebook is the
`HEATING_MP_SUBTITLES` dict (lines 523–526). It feeds the dot plot panel titles (line 578), the
bill-savings verification print headers (line 706), and the NPV histogram titles (line 746) — so
editing the dict once updates every heating figure.

**Edit (lines 523–526).**
```diff
 HEATING_MP_SUBTITLES = {
-    3: 'Single-stage, min-efficiency ASHP (SEER 15, HSPF 9)',
-    4: 'Variable-speed, high-efficiency ASHP (SEER 24-29.3, HSPF 13-14)',
+    3: 'ASHP (MP3 – Standard, HSPF1 = 9, HSPF2 ≈ 7.5)',
+    4: 'ASHP (MP4 – High Efficiency, HSPF1 = 14, HSPF2 ≈ 11.5)',
 }
```

**Why these values.** Rule 5 mandates HSPF1 (and HSPF2 where available) and forbids SEER on
heating figures. The canonical project values are MP3 HSPF1 = 9 / HSPF2 ≈ 7.5 and
MP4 HSPF1 = 14 / HSPF2 ≈ 11.5. Note this also corrects the current MP4 subtitle, which said
"HSPF 13-14" rather than the canonical HSPF1 = 14.

**Validation gate.** Workspace search for `SEER` returns zero hits in the notebook. Re-run the
dot plot, the bill-savings verification print, and the NPV histogram; confirm all three now show
the HSPF-only subtitles and no numbers changed.

### Task 3 — Dot plot: simplify y-axis + extract a reference table

**Goal.** Strip the dot plot y-tick labels down to `fuel + income group` (e.g. `Natural Gas — LMI`)
and move the sample-count breakdown (`31.1/43.8 M Homes (71.1% Fuel)`) into a standalone reference
table for the SI / scope section.

**Critical mechanics (read Appendix A first).** The sample-count suffix is appended **inside the
imported module** `visuals_adoption_dotplot`, not in the notebook. The two clean options are a
notebook-side `ax.set_yticklabels(...)` override (recommended, non-invasive) or a module-side edit
(larger refactor, affects other notebooks). Confirm the module's label construction before choosing.

**Reference table (notebook-side, already feasible).** The data lives in `sample_info` at lines
625–632 (`grouping`, `pct_of_sample`, `weighted_homes_millions`). Collect it across MPs into one
table keyed by `fuel + income group`. Output format (CSV vs markdown) is a researcher decision —
this is a 🛑 STOP gate in the prompt.

**Validation gate.** Dot plot y-axis shows no `\n` line breaks and no counts; the reference table
round-trips and is cross-referenceable by `fuel + income group`; dot point positions (adoption %)
are unchanged.

### Task 5 — Light codebase cleanup

**Goal.** Remove dead code, redundant imports, and stale comments without restructuring cells or
changing logic. Specific, verified targets (see Appendix C for the full list):

- Redundant re-import of `aggregate_bill_savings, aggregate_demand` inside the diagnostics cell
  (line 208) — already imported at the top.
- `from matplotlib.colors import Normalize` imported twice inside cells (lines 229, 466).
- Likely-unused top-level imports: `importlib`, `mcolors`, `prepare_state_geodataframe`,
  `print_adoption_decision_percentages`, `subplot_grid_adoption_vBar` — **verify each is unused
  before deleting.**
- Commented-out dead code: `_adopt_cbar_ticks` (lines 471, 488), `suptitle=` (line 774).
- Diagnostic-only cell `[21]` (lines 669–676) — a column-existence probe; demote or delete.
- Stale comment header `# TASK 1 DIAGNOSTICS` (line 183) — predates this session's task numbering;
  rename to something descriptive (e.g. `# County sample-size & coverage diagnostics`).

**Note on inline functions.** There are **no** function definitions inside notebook cells — all
functions are imported from `cmu_tare_model`. The "type hints + docstrings on inline functions"
criterion is therefore vacuously satisfied; do not invent functions to document.

**Validation gate.** Notebook runs top to bottom with no `NameError`/`ImportError`; all golden
values unchanged; no commented-out block longer than a few lines remains.

## Reference Values (golden)

Pulled from the attached PDF. **None of these may change** in this session.

| Quantity | MP3 | MP4 |
|---|---|---|
| Median bill-savings ratio | 0.615 | 0.394 |
| Counties saving money (ratio < 1) | 2934 / 3098 | 3082 / 3098 |
| Median `pct_bill_change` | −38.5% | −60.6% |
| Total elec demand change | +427,043.7 GWh | +30,618.4 GWh |
| Median `pct_elec_demand_change` | +22.5% | −8.1% |
| Mean adoption rate | 20.8% | 20.5% |
| NPV moreWTP, National median (pre-IRA / IRA) | −8,271 / −8,144 | −14,455 / −7,420 |
| NPV moreWTP, PA median (pre-IRA / IRA) | −7,024 / −6,923 | −11,848 / −5,020 |

Shared norms (must not move): ratio `[0.186, 1.814]`; `pct_bill` `[−81.4, 81.4]`; demand GWh
`[−1038.3, 1038.3]`; demand % `[−217.5, 217.5]`. Sample sizes: 331,531 baseline; 331,526
applicable; 3,098 counties (3,083 with non-null bill ratio). National fuel counts (M homes):
Electricity 25.6, Natural Gas 43.8, Fuel Oil 4.2, Propane 4.2, Other Fuel 2.2.

## Code Standards

- Type hints on all new function signatures; Google/NumPy docstrings.
- Build column names with `f'mp{mp}'`, never hardcode `'mp3'`/`'mp4'`.
- `v4MID` cost basis everywhere; never `v3`. `moreWTP`, never `lessWTP`, in economic figures.
- TARE data lives at `DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']`.
- Allegheny filter is `df['county_fips'] == '42003'`, never `df['county']`.
- Interactive `input()` calls at lines 68, 93, 94 are intentional — do not remove.
- Fail fast with specific `ValueError`/`KeyError`. Pandas over Dask unless memory forces otherwise.

## Known Anti-Patterns (do NOT suggest)

| Anti-pattern | Why it's wrong here |
|---|---|
| "Keep SEER alongside HSPF for completeness" | Rule 5 forbids SEER on heating figures outright. |
| Hardcoding `'mp3'`/`'mp4'` while editing labels | Violates the `f'mp{mp}'` convention (Rule 3). |
| Editing the `visuals_adoption_dotplot` module to simplify y-labels without flagging | It is a shared module; this is a larger refactor, not the light cleanup in scope. |
| Removing the `input()` calls "since they block batch runs" | They are intentional; cells guard them with `try/except NameError`. |
| "Recompute `pct_bill_change` here for clarity" | It is already computed in `aggregate_bill_savings`; recomputing risks drift. |
| Deleting imports without checking usage | `prepare_state_geodataframe` etc. may be used; verify first. |
| Switching `v4MID`→`v3` or `moreWTP`→`lessWTP` "to match older columns" | Hard rule violation; would change every economic number. |
| Touching `column=`/`norm=`/filters during a label edit | Breaks the no-numerical-change contract. |

## Appendix A — Task 3 y-label mechanism (the fork)

The notebook calls `plot_adoption_panel(plot_df, ax, ...)` (line 637), and the y-tick labels with
embedded counts are produced inside that function (module
`cmu_tare_model.adoption_potential.data_processing.visuals_adoption_dotplot`). Two clean paths:

- **Option A — notebook-side override (recommended).** After the `plot_adoption_panel` call,
  derive simplified labels from `GROUPING_ORDER` / `plot_df['grouping']` and call
  `ax.set_yticklabels(simplified_labels)`. Pros: module untouched, reversible, fits "light
  cleanup." Cons: the override list must match tick order exactly — verify against the current
  figure.
- **Option B — module-side edit.** Change the label construction in `visuals_adoption_dotplot`.
  Pros: single source of truth. Cons: affects every notebook importing it; out of scope for a
  light cleanup; needs its own review.

**Action before coding:** open `visuals_adoption_dotplot.py`, find where the
`"… M Homes (… Fuel)"` suffix is assembled, and confirm whether `grouping` already equals
`fuel + income group` (so the override is just the bare `grouping` string). Present the finding,
then choose A or B with the researcher.

## Appendix B — Deferred tasks (4 and 6): the blockers

**Task 4 (replacement-scenario dot plot).** The paper is moving to financial feasibility under
heating-only / cooling-only / heating-and-cooling replacement scenarios. From the column dump:
`*_heating_*_moreWTP_*` and `*_heating_and_cooling_*_moreWTP_*` columns **exist**, but a
**cooling-only** `moreWTP` column does **not** appear. Before any plotting code, run
`[c for c in df.columns if 'moreWTP' in c]` and confirm whether cooling-only exists. If it does
not, that is a model-side (TARE output) blocker, not a notebook task. Plus the new figure needs
2–3 layout options presented before committing. This is design work — handle it interactively in
Claude web.

**Task 6 (county-level CSV export, ~3,098 rows).** Several requested columns may not exist in the
current analysis output: **per-county health-damage reduction** and **per-equipment-type retrofit
counts** (ducted vs ductless adopters). These would need new computation steps, not extraction.
Before writing the export, confirm which columns already exist vs. need new computation, and lock
the column list + units in a `column_dictionary.md`. This is scoping work — handle it
interactively in Claude web.

## Appendix C — Full Task 5 cleanup checklist

Verified from the `.py`:

1. Line 9 `import importlib` — appears unused; verify, then remove.
2. Line 14 `import matplotlib.colors as mcolors` — appears unused (code uses
   `from matplotlib.colors import Normalize` locally); verify, then remove.
3. Line 29 `prepare_state_geodataframe` — appears unused; verify, then remove.
4. Lines 39–40 `print_adoption_decision_percentages`, `subplot_grid_adoption_vBar` — appear
   unused; verify, then remove.
5. Line 208 redundant re-import of `aggregate_bill_savings, aggregate_demand`.
6. Lines 229 & 466 duplicate `from matplotlib.colors import Normalize` — keep one (top of cell),
   or move to the top-level import block.
7. Lines 471, 488 commented `_adopt_cbar_ticks` — remove if not being reinstated.
8. Line 774 commented `suptitle=` — remove if not being reinstated.
9. Cell `[21]` (lines 669–676) column-existence probe — demote to a comment or delete; it is a
   leftover diagnostic, not a result.
10. Line 183 `# TASK 1 DIAGNOSTICS` — stale label; rename descriptively.

**Constraint:** light cleanup only. Do not merge cells, extract functions, or change analytical
logic. Flag anything larger as a follow-up.

## Session Summary Template

At the end of the Copilot session, produce a summary covering: (1) each task and whether its
validation gate passed; (2) the before/after for every label edited; (3) the Task 3 mechanism
chosen (A or B) and why; (4) the exact list of imports/lines removed in Task 5, with confirmation
the notebook still runs top-to-bottom; (5) confirmation that every golden value is unchanged;
(6) any item deferred or flagged for follow-up.
