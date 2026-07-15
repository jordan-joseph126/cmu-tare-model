# Session Changelog -- 2026-07-14 (Adoption Dotplot Revisions)

## Plain-language summary

This session reworked the economic-adoption dot plot and nothing else. It was a
visualization-only session: no model math, no NPV column, and no adopter or
rebate column was changed. Every change only *reads* columns that the model
already produces.

There were three requests:

1. Show three replacement-credit scopes on the first dot plot instead of two.
2. Draw one designated "headline" marker as a filled star, and draw every other
   marker as an empty outline.
3. Go back to the earlier left/right label placement, which read more cleanly
   than the vertical stack that had crept in.

Along the way one deliberate change *does* move the numbers a figure reports:
the first dot plot now plots the June 2026 subsidized adoption rate instead of
the December 2024 rate. That was done on purpose, kept in its own commit, and
flagged for review. Everything else is cosmetic.

All code changes landed in the plotting module
(`cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py`)
and in the main notebook export
(`cmu_tare_model/tare_model_main_v2_3_EXPORT_14July2026.py`). Per the project
rule, the `.ipynb` itself is never edited by the assistant; the researcher
backported the two plot cells and the markdown cell by hand (done during this
conversation).

## Where the label/marker constants live (correction)

The dot plot's label and marker constants (`REPLACEMENT_CREDIT_*`,
`REBATE_POLICY_SCENARIO_*`, and the legend-handle builders) live in
`visuals_adoption_dotplot.py`, NOT in `constants.py`. The session prompt asked to
"update constants"; the single true home is the plotting module, so that is where
the edits went. No duplicate copy was created in `constants.py`. `constants.py`
only holds `REBATE_POLICY_SCENARIOS` (guidance strings), which is unrelated to
the plot.

## Task 1 -- Audit (no edits)

- Confirmed the nine economic-adopter columns exist for MP3 and MP4: three
  replacement-credit scopes (`heatingSavings_coolingLCC`,
  `heatingLCC_coolingSavings`, `heatingLCC_coolingLCC`) times three rebate
  vintages (`_unsub`, `_sub`, `_sub_june2026`). These come from
  `NPV_CASE_CATEGORIES` in `column_names.py`, so the third marker only reads an
  existing column.
- Found the true home of the constants (the plotting module, not `constants.py`).
- Worked out which figure gets the star, and flagged that fully honoring the
  "designated case" in the first cell means pointing it at the June 2026 rate --
  a value move.
- Got the researcher's decisions before any edit:
  - Both figures get a filled star for their headline case, and the first cell
    switches to the June 2026 rate (treated as in-scope this session, its own
    commit, flagged for later review).
  - Marker order and shapes: heating = circle, cooling = triangle,
    heating + cooling = star (the star is the one drawn filled). Order low to
    high credited replacement: heating, cooling, both.

## Task 2 -- Third replacement-credit scope

Commit: `093b0a9` "Adoption dotplot: add cooling-only credit scope + land prior
uncommitted work".

- Generalized `build_econ_plot_df`'s default `replacement_credit_scenario` mode
  from two hardcoded scope columns to a loop over three scopes. The order and
  labels are driven by a new module constant, `REPLACEMENT_CREDIT_SCOPES`, an
  ordered list of `(scope token, marker label)` pairs: heating, cooling, both.
- This adds the previously missing `heatingSavings_coolingLCC` marker (cooling
  replacement credited only), shown as a triangle.
- Added a plain `rebate_vintage` argument to `build_econ_plot_df` (`'sub'` =
  December 2024, `'sub_june2026'` = June 2026), so the plotted vintage is passed
  as an argument instead of being buried in a hardcoded column name. It defaults
  to `'sub'`, which keeps the two old scopes byte-identical.
- `REPLACEMENT_CREDIT_CASES` is now derived from `REPLACEMENT_CREDIT_SCOPES`
  (single source of order). `build_replacement_credit_legend_handles` picks up
  the third scope automatically.
- Updated the module docstring and the `build_econ_plot_df` docstring to describe
  three markers and the new argument.
- Verified: three rows now emit for MP3 and MP4 National; the two pre-existing
  scopes plot exactly their prior values (byte-identical), and the new cooling
  scope reads its own column.
- This same commit also carried earlier uncommitted dot plot work that shared the
  same lines and could not be split cleanly: the relabel to "December 2024 /
  June 2026 Rebate Eligibility", `NATIONAL_FUEL_GROUPING_ORDER`, and the
  weight-derived per-group homes count. The commit message says so.

## Value move -- first cell now plots the June 2026 rate

Commit: `6b4dbe7` "Dotplot Cell 1: plot the June 2026 subsidized rate (VALUE
MOVE)".

- The first plot cell used to show the December 2024 subsidized rate for each
  scope. It now shows the June 2026 rate, via `rebate_vintage='sub_june2026'`.
- This changes the numbers the figure reports. To make the change visible on
  every run, the cell also builds the December 2024 version and prints, for the
  National row, `December 2024 -> June 2026` for each scope with the point
  difference.
- Real before/after values need a full model run in the researcher's environment.
  The researcher should confirm these against the paper's headline numbers.

## Task 3 -- Filled-star headline, empty everything else

Commit: `f5d29f7` "Dotplot: fill only the headline marker + go back to left/right
labels".

- `plot_adoption_panel` gained a `filled_tier` argument. The one case named by
  `filled_tier` is drawn filled; every other marker in the row is drawn as an
  empty outline in the fuel color. When `filled_tier` is `None` (any other
  caller), all markers stay filled, so nothing else changes.
- The headline case uses a star shape. In the marker maps:
  `REPLACEMENT_CREDIT_MARKERS['Heating + Cooling Repl. Credit'] = '*'` and
  `REBATE_POLICY_SCENARIO_MARKERS['June 2026 Rebate Eligibility'] = '*'`.
- The two legend builders gained matching arguments,
  `build_replacement_credit_legend_handles(filled_case=...)` and
  `build_rebate_policy_scenario_legend_handles(filled_label=...)`, so the legend
  draws the headline handle filled and the rest as empty outlines -- the legend
  shapes and fills match the plot.
- Cell 1 stars "Heating + Cooling Repl. Credit"; Cell 2 stars "June 2026 Rebate
  Eligibility".
- Verified on a headless render: exactly one filled marker per row, and the
  legend faces match (headline = filled gray, others = no fill).

## Task 4 -- Left/right labels instead of the vertical ladder

Same commit as Task 3 (`f5d29f7`).

- For a close cluster of three or more markers, the labels were being stacked in
  a vertical ladder, which read poorly. They now spread horizontally again: the
  leftmost label goes left, the rightmost goes right, and any middle marker keeps
  its label centered on its own marker (a new `cluster_center` case makes the
  middle label stay put even near an edge).
- Both plot cells now pass a nonzero `annotation_x_offset_pts` (26). With the old
  value of 0 the left/right split collapsed to no separation.
- The vertical-ladder code (`stagger`) is no longer triggered by these cells; the
  `cluster_stagger_pts` argument remains on the function but is now effectively
  unused by the two cells.
- Verified on headless renders of both cells at the manuscript font size: three
  well-separated markers per row, no vertical stacking, no edge clipping.
- Watch-item: when the top National row's rightmost marker sits near 90%+, its
  label can tuck under the upper-right legend. That is the existing legend
  placement (`loc='upper right'`), not the label change. Move the legend if the
  real data puts a high rate there.

## Task 5 -- Documentation

Commit: `b46a4d3` "Docs: record the dotplot revision session".

- Updated `CLAUDE.md`: the "Last updated" header line and a new session-log row
  describing the third scope, the `rebate_vintage` argument, the June 2026 value
  move, the filled-star convention, and the label revert. Also notes that the
  constants live in the plotting module, not `constants.py`.
- Added this changelog file.
- The `CLAUDE.md` commit also carried earlier uncommitted doc edits that were
  already in the working tree; the commit message says so.

## Notebook backport (done by the researcher this conversation)

- The assistant provided copy-paste-ready versions of the three cells (the
  markdown cell, the replacement-credit plot cell, and the rebate-policy plot
  cell), with Cell 1 given a self-contained import header.
- The researcher backported all three cells into `tare_model_main_v2_3.ipynb`.
- Readiness check before re-running: the plotting module exposes every name the
  cells import, and the signatures accept `rebate_vintage`, `filled_tier`,
  `filled_case`, and `filled_label`. The main gotcha is a stale import -- the
  kernel must be restarted (Restart & Run All) so the edited module is
  re-imported, otherwise the new keyword arguments raise `TypeError`.

## Verification performed this session

- Import of the plotting module succeeds; the notebook export parses (with
  Jupyter magics stripped).
- Three rows emit per grouping in the default mode for MP3 and MP4; the two old
  scopes are byte-identical to before.
- Headless renders of both plot cells at manuscript font: one filled marker per
  row, legends match, left/right label split reads cleanly, no clipping.
- Not done here (needs the researcher's full data run): the real
  December 2024 -> June 2026 National numbers, and confirming the value move
  against the paper.

## Commits (on branch update-data-and-projections-aeo2026-cambium2024)

- `093b0a9` -- Task 2: third (cooling-only) scope + `rebate_vintage` argument;
  also lands prior uncommitted dot plot work.
- `6b4dbe7` -- VALUE MOVE: first cell plots the June 2026 rate.
- `f5d29f7` -- Tasks 3 and 4: filled-star headline + left/right labels.
- `b46a4d3` -- Task 5: docs (CLAUDE.md + this changelog).

## Working-preference notes saved this session

The researcher set two standing preferences, saved to assistant memory:

- Plain language everywhere -- comments, docstrings, commit messages, and
  messages to the researcher.
- Relaxed workflow -- do not hard-stop after every task; make value-moving
  changes in their own commit, flag them clearly, and keep going. Ask a quick
  question only when something is genuinely ambiguous.

## Deferred to the researcher

- Full model run to produce the updated figures and the real
  December 2024 -> June 2026 National numbers; confirm the value move against the
  paper's headline numbers.
- Optional cleanup: `TIER_MARKERS`, `ALL_TIER_NAMES`, `_build_legend_handles`,
  and `cluster_stagger_pts` are now effectively unused by the two cells; prune if
  nothing else needs them.
- If the top National label collides with the legend on the real data, move the
  legend (for example `loc='lower right'`).
