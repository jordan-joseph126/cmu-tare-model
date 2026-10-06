# TARE Model -- ResStock 2025.1 Phase 2: Release-Aware Loader Session Prompt

CLAUDE.md is the source of truth for conventions, file rules, golden values, coding standards,
and all non-negotiables; it is auto-loaded, so this prompt only adds session-specific state and
the task list. Audit before editing and use one diff per stop gate, per CLAUDE.md.

## Context

**Priority: build the release-aware column-reading layer that every later phase depends on,
without moving a single 2022.1.1 value.** This is Phase 2 of 11 in
`REFACTORING_GUIDE_17Sept2026_DualFuel_ResStock2025_1.md` ("Release-aware loader"), scoped to
`process_euss_data.py` plus the constants and new schema module it needs.

Phase 1 (read-only audit, 18-19 Sep 2026) is complete and changed no code. It produced
`cmu_tare_model/docs/AUDIT_2026-09-18_ResStock2025_1_DualFuel.md` (facts, file:line citations,
measurements) and `cmu_tare_model/docs/resstock_2025_1_column_map.csv` (123 rows:
`logical_name`, `name_2022_1_1`, `name_2025_1`, `unit`, `dtype`, `status` in
{confirmed, renamed, new, missing}, `note`). The 2025.1 national AMY2018 parquet files are in
`data/resstock_2025_1/` (baseline = `upgrade0.parquet`, dual fuel = `upgrade5.parquet`,
cold-climate ASHP = `upgrade4.parquet`), git-ignored, listed in `DATA_MANIFEST.md`.

Researcher decisions (19 Sep 2026) that bear on this session:
- Complete every phase for Upgrade 05 (dual fuel) first. Upgrades 04 (cold-climate ASHP) and 03
  (reference space heating and air conditioning, circa 2025) will be added later, after Phase 11
  finishes for 05. This session loads the baseline and 05 only; `upgrade4.parquet` is on disk but
  not loaded, and 03 has not been downloaded.
- Add a `RESSTOCK_RELEASE` constant and a `RESSTOCK_RELEASE_AND_MP` registry (release -> its
  measure packages). Pick whichever shape needs the least refactoring.
- Alaska and Hawaii are excluded from the study. With them out, every `in.county` is GISJOIN and
  the baseline has 3,107 counties, matching 2022.1.1 -- no FIPS, GEA, or CT code change.
- `applicability` becomes the FIRST filter in a filter funnel built in Phase 3, which will print
  rdu, weighted homes, and share by baseline heating fuel after each stage.
- The four peak pass-through columns are a straight rename.
- Panel cost is ignored; the panel constraint flags are carried through as reporting columns.

Out of scope: masking, `include_heating`/`include_cooling`, degree-day consumption, fuel costs,
capital cost, SEER2 conversion, rebates. Do not implement them even where the audit or guide
describes them.

## Current state the tasks depend on

- **MP numbers will collide across releases once 03 and 04 are added.** 2025.1 Upgrade 03
  (reference furnace/AC) and Upgrade 04 (cold-climate ASHP) will become `mp=3` and `mp=4` -- the
  same integers as 2022.1.1 MP3 and MP4, but different packages. Only `mp=5` is live this
  session, but shape `RESSTOCK_RELEASE_AND_MP` now so 03 and 04 can be added later as a one-line
  change, and never let an MP check run without the release.
- **Do NOT drop non-applicable rows at load time.** The guide's Phase 2 text says "filter
  `applicability == True`", but the researcher's decision puts that filter at the head of the
  Phase 3 funnel, which needs the pre-filter totals. This session carries `applicability` as a
  bool column only. It is already a real `bool` in all three 2025.1 parquet files, so coercion is
  a no-op there. Today's 2022.1.1 run path never reads `applicability` (audit A.10); keep it that
  way.
- **Weight is read from the frame, never hardcoded.** 2025.1 = `253.90367272727272`;
  2022.1.1 = `242.131013`.
- **The raw file read lives in the notebook, not the module.** `tare_baseline_v3_0.ipynb` cell 2
  reads the 2022.1.1 CSV with `columns_to_string = {11: str, 61: str, ...}`, which are column
  positions in the 2022 CSV. Against the 771-column 2025.1 file those positions point at the wrong
  columns and fail silently. For 2025.1, add a module-level parquet read function and hand the
  researcher the notebook cell. Never reuse `columns_to_string` for 2025.1.
- **`df_enduse_compare` requires a 2022-only frame.** Its signature (`process_euss_data.py:410`)
  requires `df_cooking_range`, which `tare_scenarios_v3_0.ipynb` cell 2 loads from the 2022
  `upgrade07` file. Cooking is an inactive category, and 2025.1 has no `upgrade.cooking_range`
  column. Make the smallest change that lets a 2025.1 call omit it without changing any 2022.1.1
  call.
- **Do NOT run the MP3 SEER/HSPF override on the dual-fuel string.** The `menu_mp == 3` ENERGY
  STAR override does a plain substring replace (`'SEER 15'` -> `'SEER 16'`). On
  `"Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5% AFUE NG, 35F switchover"` it
  would match inside `SEER 15.2`. Write a dedicated dual-fuel parser. Also add a release guard to
  the MP3 override: it is keyed on `menu_mp == 3`, and 2025.1 Upgrade 03 will later arrive as
  `mp=3`, so without the guard it would fire on the wrong package.
- **`out.natural_gas.heating_hp_bkup.energy_consumption..kwh` carries the dual-fuel furnace gas.**
  That is 81% of dual-fuel heating site energy, non-zero on 89.79% of applicable rows. The column
  also exists in 2022.1.1, where it is always zero. Leaving it out of the 2025.1 load silently
  drops most of the retrofit's fuel use.
- **`out.params.size_heat_pump_backup_primary` is commented out** at `process_euss_data.py:450`.
  Carry it for 2025.1 as the furnace size Phase 5 will cost. Attach no cost logic to it here.
- **Leave the 20 Aug 2026 old-system-size fix alone.** `base_size_heating/cooling_system_primary_
  k_btu_h`, copied from the baseline run, is what makes MP3 and MP4 replacement costs identical.
  Extend the same pattern to 2025.1; do not change the 2022.1.1 columns or values.
- **Build every rename from `resstock_2025_1_column_map.csv`**, not by hand. The 2025.1
  convention puts the unit after a double dot (`..kwh`, `..kbtu_per_hr`).
- **New standing rule (19 Sep 2026, going forward only): a simple rename must not read as a new
  column.** Scope is the **2025.1 read path only** -- the 2022.1.1 path is untouched, per the
  researcher's "going forward only" decision. `RESSTOCK_COLUMN_MAP`'s `logical_name` keys are a
  lookup key, never a DataFrame column name -- `resstock_col()` returns a physical name only, so
  Task 2 and Task 3 already satisfy this and need no rework. What Task 4 and Task 5 must do: on
  the 2025.1 assignment step only, for every `status == 'renamed'` row, assign the result under
  the column map's `name_2022_1_1` spelling instead of the raw 2025.1 name. This is a no-op on
  the 2022.1.1 path -- `name_2022_1_1` is by definition the name 2022.1.1 already reads, and that
  path's output must still equal today's exactly (`DataFrame.equals`), unchanged by this rule. A
  `status == 'new'` row (no 2022.1.1 counterpart) keeps its native 2025.1 dot name -- there is
  nothing to canonicalize onto. `confirmed` and `missing` rows are already unambiguous and need no
  change. This only governs *raw ResStock source columns* (the `in.`/`out.`/`upgrade.` namespace);
  it does not touch the existing `ref2025_mp{mp}_`/`baseline_` prefix that already marks a
  TARE-computed column (`modeling_params.py`) -- that distinction already exists and is out of
  scope here. Net effect on the 2025.1 frame: every column is either a native source name
  (2022.1.1-shaped, even though read from 2025.1) or a `ref2025_mp{mp}_`/`baseline_` computed name
  -- never a bare `logical_name` and never a cosmetic 2025.1 rename standing in place of the source
  name it renamed.
- **Correction: `status == 'renamed'` is necessary but not sufficient for the canonicalization
  above.** It says ResStock's own crosswalk pairs two physical columns across releases; it does
  not say they measure the same thing. If a row's two physical columns are NOT the same
  measurement -- confirmed by the audit, or found while implementing -- do NOT canonicalize onto
  `name_2022_1_1`. Instead give each release's column its own distinct destination name, each
  built from THAT release's own ResStock term. "Original name" in this rule means ResStock's name
  -- ResStock is the source -- never a name TARE invents to make two different things look like
  one. **Known instance: the four peak columns (Task 7).** 2022.1.1's `peak_when_heating`/
  `peak_when_cooling` are conditioned on the equipment running; 2025.1's
  `maximum_daily_peak_winter`/`_summer` are conditioned on the calendar season (audit E.2) --
  different measurements, so they get different destination names, not one shared name. If any
  other `renamed` row in the column map turns out to have a similar hidden definitional
  difference, apply the same treatment and flag it here.

## Tasks (execute in order; stop at each gate)

**Task 1 -- Audit (no edits).** In `process_euss_data.py`, `constants.py`, and the loader cells of
`tare_baseline_v3_0.ipynb` / `tare_scenarios_v3_0.ipynb`, locate: every literal 2022.1.1 column
string; every `weight` reference; `columns_to_string`; the `df_cooking_range` dependency; the
`menu_mp == 3` override; and the commented-out backup-size line. Check that the audit's file:line
citations still hold. Report findings and stop.

**Task 2 -- Release constants.** Add `RESSTOCK_RELEASE` and `RESSTOCK_RELEASE_AND_MP` to
`constants.py` (for example `{'2022.1.1': [3, 4], '2025.1': [5]}`, or whichever shape needs
the least downstream change; 03 and 04 are appended to `'2025.1'` later). Leave
`VALID_MENU_MPS` and its consumers as they are this session.
Verify: both import cleanly and nothing else changed. Stop.

**Task 3 -- Column registry.** Create a new module (for example
`cmu_tare_model/utils/resstock_schema.py`) with
`RESSTOCK_COLUMN_MAP[release][logical_name] -> physical_name`, built from the column map's
confirmed, renamed, and new rows. The `'2022.1.1'` entries must reproduce every literal that
`process_euss_data.py` reads today, exactly. `logical_name` is a lookup key only -- `resstock_col()`
returns the physical column name, and nothing here writes a DataFrame column, so this task needs
no change under the naming rule below. Verify: every current 2022.1.1 literal resolves to itself.
Stop.

**Task 4 -- Route the baseline reads through the registry.** Change `process_euss_data.py` so
every baseline column read goes through `RESSTOCK_COLUMN_MAP` and `weight` comes from the frame.
Add the 2025.1 parquet read function. On the 2025.1 assignment step only, for every
`status == 'renamed'` column, assign it into the frame under the column map's `name_2022_1_1`
spelling instead of the raw 2025.1 name -- see the naming rule in "Current state," above -- so a
rename never reads as a new, unrelated column. **Exception, per the correction in "Current
state": if the two releases' physical columns are not the same measurement, do not canonicalize
-- give each release's column its own distinct name instead.** The 2022.1.1 assignment step is
unchanged by this rule. Verify: on 2022.1.1, the output equals today's on every column TARE uses
(`DataFrame.equals`, or a per-column diff if dtypes differ); on 2025.1, spot-check that every
renamed column landed under its `name_2022_1_1` spelling (or its own distinct name, for a
same-row-different-measurement exception) and that no column name in the resulting frame is a
bare `logical_name`. Stop.

**Task 4 correction (19 Sep 2026, after Task 4 was already applied and verified) -- fix before
Task 7.** `base_peak_electricity_heating_kw`/`_cooling_kw` were assigned via
`resstock_col(release, 'peak_electricity_heating'/'peak_electricity_cooling')`, the same physical
pair as Task 7's four peak columns. Per the exception above, these should NOT share one
destination name across releases -- 2022.1.1's `peak_when_heating`/`peak_when_cooling` and
2025.1's `maximum_daily_peak_winter`/`_summer` are different measurements (audit E.2), and the
original Task 4 diff's byte-identity check never actually validated what the 2025.1 read means,
only that 2022.1.1 was untouched. Fix as its own stop-gated diff: keep
`base_peak_electricity_heating_kw`/`_cooling_kw` populated from 2022.1.1's physical columns only,
and add new, distinctly-named 2025.1 columns (built from ResStock's own `winter`/`summer` term,
not `heating`/`cooling`) populated from the 2025.1 physical columns. Verify (lightweight, per the
narrowed regression-guarantee scope below -- a targeted check on just these two columns, not a
full-frame `DataFrame.equals`): `base_peak_electricity_heating_kw`/`_cooling_kw` still hold the
same 2022.1.1 values as before this fix; the new 2025.1 columns populate under their own names;
nothing reads `base_peak_electricity_heating_kw`/`_cooling_kw` expecting 2025.1 data (grep before
removing anything). Stop.

**Task 5 -- Dual-fuel upgrade (05) load.** Load `upgrade5.parquet` through the registry and keep
`applicability` as a bool column (no row drop). Write a typed parser for
`upgrade.hvac_heating_efficiency` that returns `hp_seer2`, `hp_hspf2`, `backup_fuel`,
`backup_afue`, and `switchover_f` for both published strings (92.5% and 95.0% AFUE). Set
`retrofit_heating_fuels = ['electricity', 'natural_gas']` for `mp=5`. Add the release guard to
the MP3 override. Resolve `df_cooking_range` per the current-state note. Apply the same
`renamed` -> `name_2022_1_1` canonicalization as Task 4 to any 05-specific column the map marks
`renamed`; a 05-only column with no 2022.1.1 counterpart (`status == 'new'`) keeps its native
2025.1 name. Verify: both strings parse with no corrupted field, the MP3 override cannot run on
this path, and the same naming spot-check from Task 4 holds for the 05 frame. Stop.

**Task 6 -- Extend `df_enduse_refactored`.** For the 2025.1 path, add: baseline heating energy by
fuel; retrofit `out.electricity.heating`, `out.electricity.heating_fans_pumps`, and
`out.natural_gas.heating_hp_bkup`; cooling energy; `base_size_heating/cooling_system_primary`
from the baseline run; `size_heat_pump_backup_primary`; `in.ashrae_iecc_climate_zone_2004`;
`in.electric_panel_service_rating..a`; and the three post-upgrade panel constraint columns as
pass-through only. Verify: on applicable dual-fuel rows the gas backup column is non-zero while
baseline is zero, and the 2022.1.1 output columns are unchanged. Stop.

**Task 7 -- Peak columns: register both releases' physical names, keep destinations separate.**
2022.1.1's `out.electricity.peak_when_heating.kw`/`peak_when_cooling.kw` (plus their two
`.savings` variants) and 2025.1's `out.qoi.electricity.maximum_daily_peak_winter..kw`/
`..._summer..kw` (plus their two `.savings` variants) are NOT the same measurement -- 2022.1.1 is
conditioned on the equipment actually running, 2025.1 on the calendar season (audit E.2) -- so per
the naming-rule exception in "Current state," they do not get a shared destination name, even
though the column map pairs them as a rename. Register both releases' physical names in the
registry (each under its own logical name; do not force one logical name to cover both). Give
2022.1.1's four columns their existing destination names, unchanged. Give 2025.1's four columns
their own new destination names, built from ResStock's own `winter`/`summer` term -- not
`heating`/`cooling`, which would misrepresent them as equipment-conditioned. "Original name" here
means ResStock's own term for each release, since ResStock is the source; TARE does not invent a
shared label that papers over the difference. Add a one-line code comment on both blocks: 2022
peaks are conditioned on the end use running, 2025 peaks on the calendar season -- this is why
they're separate columns, not a rename. Also add this same caveat to whatever reference doc
serves as the Tepper data dictionary (the audit itself flagged this as something that needs to
reach that doc, not just a code comment). Per the narrowed regression-guarantee scope below, a
2022.1.1 byte-identity proof is not required here -- verify only that the four new 2025.1 columns
populate under their own distinct names and that nothing conflates the two releases' columns.
Stop.

**Task 8 -- Full-load check and backport list.** Load the 2025.1 baseline and 05 end to end and
report: rows per file (expect 549,971 each); the `applicability == True` count for 05 (expect
245,570); the same count after dropping AK and HI (the audit did not measure it directly, so
measure it now -- the audit's figures imply 245,192). Per the dropped regression-guarantee scope
below, no 2022.1.1 MP3/MP4 byte-identity proof is needed here or anywhere later in this project.
Then list the exact notebook cells the researcher must backport by hand, with paste-ready
replacement code for each. Stop.

## Regression-guarantee scope, dropped going forward (19 Sep 2026)

The original guide (Section 1) treats byte-identical 2022.1.1 MP3/MP4 output as a hard,
per-task requirement, proven again at the end of every phase. The researcher has a separate,
already-archived branch for the 2022.1.1 analysis and does not plan to run MP3/MP4 from this
branch at all, so that proof is dropped outright -- not deferred to a single later check --
**starting with Task 7**, above. There is no "prove it once, at Phase 11" requirement either; a
final full-comparison run would still be spending real effort protecting a result nobody will
use. What stays the same: the 2022.1.1 code path is still left alone by default -- nothing here
means intentionally breaking it -- but it is not verified, proven, or reconciled at any point,
including Phase 11. Tasks 2-6 already carried their own byte-identity verification and that work
stands as-is; nothing about it needs to be undone or repeated, it simply stops being required
going forward. Future phase session prompts (3-11) should carry no per-task 2022.1.1
verification step and no full-comparison task anywhere, including at the end of Phase 11.

## Start

Do Task 1 only. Produce the audit and stop at the first gate for approval.

## Deferred to a future session (do NOT start now)

- Phase 3 -- masking: `applicability` as the first funnel filter, plus the per-stage funnel table
  (rdu, weighted homes, and share for Electricity, Electricity ASHP, Fuel Oil, Natural Gas,
  Propane). This phase also removes the cooling-technology filter on BOTH releases -- an
  intentional change to 2022.1.1 MP3/MP4 output that breaks byte-identity with
  `2026-09-02_19-04` on purpose and requires superseding reference values.
- Phase 3 also makes `VALID_MENU_MPS`, `EQUIPMENT_SPECS`, and `REBATE_ELIGIBLE_HEATING_MPS`
  release-aware.
- Phase 4 -- degree-day consumption as a list of (fuel, column) pairs; removes the
  electricity-only assumption on both releases.
- Phase 5 -- SEER2/HSPF2 -> SEER1/HSPF conversion at the REMDB boundary (confirm the Appendix M1
  factors against their source first); a furnace upgrade row id added unconditionally; a 4th
  `add_remdb_metrics` call gated on dual fuel.
- Phase 6 -- electricity + natural-gas fuel-cost streams.
- Phase 7 -- rebates: `mp=5` added to the eligible set; dual-fuel retrofits pass the June 2026
  HEEHR fossil-baseline gate.
- Phases 8-11 -- NPV/adoption, climate damages (including furnace combustion), KPIs/visuals/
  exports, full run + reference values + CLAUDE.md updates.
- After Phase 11 is complete for 05 -- Upgrade 04 (cold-climate ASHP): add it to
  `RESSTOCK_RELEASE_AND_MP['2025.1']` as `mp=4`, then run it through the same phases.
  `upgrade4.parquet` is already downloaded and measured (413,604 applicable rdu before the AK/HI
  exclusion).
- After Upgrade 04 -- Upgrade 03 (reference space heating and air conditioning, circa 2025):
  download `upgrade3.parquet`, add it as `mp=3`, and audit it first. It is a like-for-like
  furnace/AC replacement rather than a heat pump.
