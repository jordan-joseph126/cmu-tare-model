# Session Changelog -- 2026-07-14 (Adoption Dotplot Revisions)

## What this session did

This was a visualization-only session. Nothing here changes any model output,
NPV column, or adopter column -- every change reads columns that already exist.
The work is confined to the economic-adoption dotplot: the module
`cmu_tare_model/adoption_potential/data_processing/visuals_adoption_dotplot.py`,
the two dotplot cells in the main notebook export
(`tare_model_main_v2_3_EXPORT_14July2026.py`), and the label/marker constants
those cells use.

One thing does change the numbers a figure reports: the first plot cell now
plots the June 2026 subsidized rate instead of the December 2024 rate. It is in
its own commit and flagged below.

## Where the constants live

The label and marker constants (`REPLACEMENT_CREDIT_*`,
`REBATE_POLICY_SCENARIO_*`, the legend builders) live in
`visuals_adoption_dotplot.py`, not in `constants.py`. The session prompt asked to
"update constants"; the one true home is the dotplot module, so that is where the
edits went. No copy was made in `constants.py`.

## Task 1 -- Audit (no edits)

Confirmed the nine adopter columns exist for MP3 and MP4 (three replacement-credit
scopes x three rebate vintages). Found the constants' true home. Worked out the
star mapping per figure and flagged that pointing the first cell at the June 2026
rate is a value move. Got the researcher's decisions before editing.

## Task 2 -- Third replacement-credit scope (commit: "add cooling-only credit scope")

Generalized `build_econ_plot_df`'s default mode from two hardcoded scope columns
to a loop over three scopes, driven by the new ordered `REPLACEMENT_CREDIT_SCOPES`
list (heating, cooling, both). This adds the previously missing
`heatingSavings_coolingLCC` (cooling replacement credited only) marker with a
triangle shape. Also added a plain `rebate_vintage` argument ('sub' = December
2024, 'sub_june2026' = June 2026), defaulting to 'sub' so the two old scopes plot
exactly their prior values. Verified byte-identical for MP3 and MP4 National, and
that three rows now emit.

This commit also carried prior uncommitted dotplot work that shared the same
hunks (the "December 2024 / June 2026 Rebate Eligibility" relabel,
`NATIONAL_FUEL_GROUPING_ORDER`, weight-derived homes); it could not be split
cleanly, so the commit message says so.

## Value move -- first cell now plots the June 2026 rate (commit: "plot the June 2026 subsidized rate (VALUE MOVE)")

The first plot cell used to show the December 2024 subsidized adoption rate for
each scope; it now shows the June 2026 rate, passed via `rebate_vintage`. This
changes the numbers the figure reports. The cell prints the December 2024 ->
June 2026 National rate for each scope on every run so the size of the move is
visible. Real before/after values need a full model run in the researcher's
environment.

## Task 3 -- Filled-star headline (commit: "fill only the headline marker + go back to left/right labels")

`plot_adoption_panel` gained a `filled_tier` argument: the one headline case is
drawn filled with a star shape, and every other marker is drawn as an empty
outline. Cell 1 stars "Heating + Cooling Repl. Credit"; Cell 2 stars "June 2026
Rebate Eligibility". The two legend builders gained a matching
`filled_case`/`filled_label` so the legend shapes match the plot. Checked on a
headless render: exactly one filled marker per row, and the legend faces match.

## Task 4 -- Left/right labels instead of the vertical ladder (same commit as Task 3)

For a close cluster of three or more markers the labels were being stacked in a
vertical ladder, which read poorly. They now spread left/right again: the leftmost
label goes left, the rightmost goes right, and any middle marker stays centered on
its own marker. Both plot cells now pass a nonzero `annotation_x_offset_pts` (26)
so the split actually separates. Checked on a headless render of both cells at the
manuscript font size.

Watch-item: when the top National row's rightmost marker sits near 90%+, its label
can tuck under the upper-right legend. That is the existing legend placement
(`loc='upper right'`), not the label change. Move the legend if the real data puts
a high rate there.

## Deferred to the researcher

- Backport the Cell 1 and Cell 2 changes to the `.ipynb` (per project rule, the
  notebook is never edited here).
- Confirm the June 2026 value move against the paper's headline numbers on a full
  run.
- Optional cleanup: `TIER_MARKERS` / `ALL_TIER_NAMES` / `_build_legend_handles`
  and `cluster_stagger_pts` are now effectively unused by the two cells; prune if
  nothing else needs them.
