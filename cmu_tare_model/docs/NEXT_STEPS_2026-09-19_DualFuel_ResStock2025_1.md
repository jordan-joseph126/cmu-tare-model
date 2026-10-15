# Next Steps — ResStock 2025.1 Dual Fuel Integration

**As of:** 19 September 2026, after Phase 1 (audit) and the researcher's D1–D10 decisions.
Revised the same day: Upgrade 05 (dual fuel) goes through every phase first; Upgrades 04 and 03
become placeholders that follow once Phase 11 is complete for 05.
**Serves:** `REFACTORING_GUIDE_17Sept2026_DualFuel_ResStock2025_1.md`.

---

## Status

- **Phase 1 (audit)** — complete, 18–19 Sep 2026. No code edited.
- **Decisions 1–3** (administrative, taken 19 Sep before this round) — exclude AK/HI; deliverables
  stay in `cmu_tare_model/docs/`. **Decision 2 (peak columns are a straight rename) superseded
  later the same day** — see "Column provenance naming," below; they are not the same measurement
  across releases and do not share a destination name.
- **D1–D10** (the guide's Section 3 decision list) — all decided (below).
- **Column provenance naming** (decided 19 Sep 2026, refined same day) — a simple rename must not
  read as a new column, and a shared name must not paper over a real difference. See below.
- **Regression guarantee dropped going forward** (decided 19 Sep 2026) — this branch won't be used
  for the 2022.1.1 analysis (already archived separately), so proving 2022.1.1 output untouched is
  no longer required on any task, including a final check at Phase 11. See below.
- **Phase 2 session prompt** — drafted: `claude/PHASE2_SESSION_PROMPT_ResStock2025_1_DualFuel.md`.

---

## Package order

1. **Upgrade 05 — Dual Fuel Heating System.** Phases 2–11 below, completed in full first.
2. **Upgrade 04 — Typical Cold Climate Ducted ASHP** (placeholder). Starts after Phase 11 is done
   for 05. `upgrade4.parquet` is already downloaded and measured (413,604 applicable rdu before
   the AK/HI exclusion; audit B.1).
3. **Upgrade 03 — Reference Space Heating and Air Conditioning Upgrade, circa 2025**
   (placeholder). Starts after 04. Not downloaded yet, so it needs its own short audit first. It is
   a like-for-like furnace/AC replacement at circa-2025 minimum efficiency, not a heat pump, so its
   fit with TARE's heat-pump framing (fuel costs, the existing-ASHP exclusion, rebates) needs
   checking before any code is written for it.

Because 03 and 04 will reuse the integers `mp=3` and `mp=4` — the same numbers as 2022.1.1 MP3
and MP4, but different packages — the release registry from D1 is built in Phase 2 even though
only `mp=5` is live at first.

---

## D1–D10, as decided 19 September 2026

| # | Decision | Implementation note |
|---|---|---|
| **D1** | Add a `RESSTOCK_RELEASE` constant plus a two-layer registry, `RESSTOCK_RELEASE_AND_MP` (release → its measure packages). Starts as `{'2022.1.1': [3, 4], '2025.1': [5]}`; 04 and 03 are appended later. Whichever shape is most minimal wins. | Makes the reused `mp=3`/`mp=4` integers safe once 03 and 04 arrive. The MP3 ENERGY STAR override is keyed on `menu_mp == 3`, so it also gets a release guard in Phase 2 — otherwise it would later fire on 2025.1 Upgrade 03. |
| **D2** | **Upgrade 05 first**, through every phase. **04 and 03 follow as deferred placeholders**, in that order. | Revised 19 Sep from "05 + 04 together." The guide defaults to 05 + 04; this runs them one after the other instead, and adds 03 as a package to analyze in its own right. |
| **D3** | Apply the `applicability` filter **first**, as a new leading filter. Every existing filter (states, housing type, technology/fuel masks, the existing-ASHP exclusion) stands unchanged on top of it. Also: print a filter-funnel table after each stage — rdu, weighted homes, and % share by baseline heating fuel (Electricity, Electricity-ASHP, Fuel Oil, Natural Gas, Propane), one row per filter applied. | This is the guide's D3 option (b), with the filter order made explicit. The funnel table is a new diagnostic that the guide doesn't have — it lands in Phase 3 (masking), where `include_heating`/`include_cooling` are built. Phase 2 therefore keeps non-applicable rows (no row drop at load time), so the funnel can show the starting totals. |
| **D4** | Remove the cooling-technology filter (`ALLOWED_TECHNOLOGIES['cooling']`) on **both** releases; rely on `applicability` instead. | **Simplified 19 Sep, after the regression guarantee was narrowed (below).** This used to be flagged as an intended, logged break of the 2022.1.1 byte-identity guarantee, needing a fresh full MP3/MP4 run and new CONFIRMED "supersedes" reference-value rows. None of that bookkeeping is needed anymore — just remove the filter on both releases and move on. |
| **D5** | Remove the electricity-only hard-wiring in the fuel-cost path; generalize to a list of (fuel, column, price) entries per retrofit, applied uniformly across releases. **Revised 19 Sep: build that list the same way the baseline side already does, by reusing/extending `get_all_possible_fuel_columns` (`calculation_utils.py`) against the retrofit's own consumption columns — not a new hand-maintained per-mp fuel table.** | Phases 4 and 6. Whichever fuel columns are zero for a given package (e.g. `out.natural_gas.heating_hp_bkup...` is always zero under MP3/MP4) drop out of the sum on their own; nothing needs to know in advance which fuels a package uses. A static `{mp: [fuels]}` table (drafted, then pulled from Task 5 — see Phase 2 session progress, below) would have to be hand-updated for every future package (03, 04, whatever comes after); reusing the existing column-enumeration approach does not. No longer need to prove this is additive for MP3/MP4 (regression guarantee narrowed, below). |
| **D6** | Add the upgrade-side REMDB furnace row-id branch unconditionally. Gate the **4th `add_remdb_metrics` call** behind a dual-fuel-retrofit check, so it fires only for `mp=5`. | Phase 5. MP3/MP4 never pass the dual-fuel check. When 03 (a new furnace) is added later, this row-id branch may be reusable for it. |
| **D7** | Ignore panel-upgrade cost for now; carry the constraint flags as pass-through reporting columns. | Guide default (a). Phase 2 (pass-through). |
| **D8** | Dual-fuel retrofits (`mp=5`) pass the June 2026 HEEHR fossil-baseline fuel gate regardless of baseline fuel. Register `mp=5` in `REBATE_ELIGIBLE_HEATING_MPS` under the 2025.1 release. | The gate tests *baseline* fuel, not "was a fossil system removed," so the exception has to be explicit. Phase 7. Rebate eligibility for 03 (a fossil furnace replacement, which is likely ineligible) is decided when 03 is added. |
| **D9** | Apply the DOE Appendix M1 SEER2/HSPF2 → SEER1/HSPF conversion factors. | The M1 factors still need confirming against their source (the guide's placeholders: SEER ≈ SEER2 / 0.95, HSPF ≈ HSPF2 / 0.85) before Phase 5. |
| **D10** | Closed by D1/Decision 1 (exclude AK/HI). | No geography code changes. |

---

## Column provenance naming (decided 19 Sep 2026, after D1–D10)

A simple rename must never look like a new column. The researcher wants it obvious, everywhere
in the codebase and on both releases, which columns are raw ResStock source data and which were
produced by TARE — a plain relabeling of a source column is exactly the case that erodes that
line, since nothing about the code marks it as "the same field, renamed" rather than "a new
field."

**Scope: going forward only (confirmed by the researcher).** The Task 1 audit's inventory of the
existing 2022.1.1 code (89 column literals across ~15 files) found no violation to fix — every
literal already carries its native `in.`/`out.`/`upgrade.` prefix, and the one existing rename
(`base_size_heating/cooling_system_primary_k_btu_h`) disambiguates baseline-vs-upgrade rather
than cosmetically relabeling a field, so it already fits the rule as stated. Nothing in the
working 2022.1.1 pipeline is touched by this decision.

**The rule, keyed off the column map's own `status` field:**

| `status` | Where it lands in the frame |
|---|---|
| `confirmed` | Identical name on both releases already — no change. |
| `missing` | 2022.1.1-only — unaffected. |
| `renamed` | **Only when the two releases' physical columns are the same measurement** — on the 2025.1 read path, assign under the column map's `name_2022_1_1` spelling instead of the raw 2025.1 name. No-op for 2022.1.1 — that release already reads the field under that exact name, so nothing there changes. |
| `new` | No 2022.1.1 counterpart — keeps its native 2025.1 dot name; there is nothing to canonicalize onto. |

**Correction, 19 Sep 2026: `renamed` is a necessary condition, not a sufficient one.** The column
map's `status == 'renamed'` says ResStock's own crosswalk pairs two physical columns across
releases — it does not say they measure the same thing. Where the audit (or later work) finds
the two releases define the field differently, canonicalizing onto one shared name would erase a
real difference instead of just hiding a cosmetic one, which is the opposite of what this rule is
for. **In that case, do not share a destination name across releases at all.** Give each release's
column its own name, taken from that release's own ResStock term — the "original name" the
researcher means is ResStock's, not a name invented by TARE, because ResStock is the source.
**Known case: the four peak columns.** Audit finding E.2: 2022.1.1's `peak_when_heating`/
`peak_when_cooling` are conditioned on the equipment actually running; 2025.1's
`maximum_daily_peak_winter`/`_summer` are conditioned on the calendar season instead — not the
same measurement, despite the column map pairing them as a rename. These must NOT land under one
shared destination name. See the Phase 2 session progress correction below — this may already be
wrong in the applied Task 4 diff, not just the still-pending Task 7 plan.

`RESSTOCK_COLUMN_MAP`'s `logical_name` keys (Task 3, `resstock_schema.py`) are a lookup key only.
`resstock_col(release, logical_name) -> physical_name` returns a physical column name and never
itself writes a DataFrame column, so **Task 2 and Task 3 need no rework** — they were already
correct under this rule by construction. It is Task 4 (baseline reads) and Task 5 (the 05 load),
neither executed yet, that must apply the `renamed` -> `name_2022_1_1` canonicalization, and only
on their **2025.1 assignment step** — the 2022.1.1 assignment step is unchanged, since it was
already required (independent of this rule) to reproduce today's output byte-for-byte via
`DataFrame.equals`. Both tasks' prompts have been updated with this instruction, scoped to the
2025.1 path only, and a matching verification step.

This is separate from, and does not change, the existing TARE-computed-column convention
(`ref2025_mp{mp}_` / `baseline_`, from `modeling_params.py`) — that already marks a column as
model output rather than source data. The new rule only concerns the raw source side: it keeps
every renamed source column landing under one stable, recognizable source name instead of a
cosmetic 2025.1 relabeling standing in for it.

**Follow-up, not yet done:** add a short subsection under CLAUDE.md's existing "Column Naming
Conventions" section recording this rule, so it is auto-loaded into every future phase session
rather than repeated in each one. Proposed text is in the reply that delivered this decision;
Claude Code should apply it as its own one-diff stop-gated edit.

---

## The regression guarantee — scope narrowed 19 September 2026

**Revised.** The guide (Section 1) treats byte-identical MP3/MP4 output as "a hard requirement
of every phase, not a final check" — a `DataFrame.equals` proof on every task. The researcher has
a separate, already-archived branch for the 2022.1.1 analysis and does not plan to run MP3/MP4
from this branch, so paying that proof on every task protects a result nobody will use from here,
at exactly the point in the project (Phases 4-9: fuel costs, capital cost, rebates, NPV) where the
verification effort is better spent catching mistakes in the new, unverified dual-fuel logic —
which has no prior oracle run to check against at all.

**Dropped outright, not just deferred to a single later check.** Starting with Task 7, a task's
prompt no longer requires proving 2022.1.1 output is byte-identical to `2026-09-02_19-04`, or to
any prior task's output, as a stop-gate condition — and there is no compensating "prove it once,
at Phase 11" requirement either. The researcher does not plan to run MP3/MP4 from this branch at
all, so a final full-comparison run would still be spending real effort protecting a result
nobody will use. **What doesn't change:** the 2022.1.1 code path is still left alone by default
— nothing here means intentionally breaking it for no reason — but it is no longer verified,
proven, or reconciled at any point, including Phase 11. D4 no longer needs fresh MP3/MP4 rows or
"supersedes" bookkeeping (see D4, above); Phase 11 covers only the 2025.1 dual-fuel package.

**Tasks 2-6, already applied, keep the byte-identity verification they already did** — that work
is done and doesn't need to be undone or repeated; it simply stops being required going forward.
This applies from Task 7 onward, and to every future phase session prompt (3-11): no per-task
2022.1.1 verification step, and no full-comparison task anywhere, including at the end of
Phase 11. If a quick sanity check is nearly free in the course of other work, there's no harm in
it, but it should never be the reason a task is broken up, delayed, or re-scoped.

---

## Phase sequence for Upgrade 05 (the guide's Section 4)

1. **Phase 1 — Acquire and audit** — COMPLETE.
2. **Phase 2 — Release-aware loader** — **prompt drafted, next to run.** Release constants and
   registry (D1); a `RESSTOCK_COLUMN_MAP` that every loader read goes through; `weight` read from
   the frame; `applicability` carried as a column; dual-fuel option-string parser; MP3 override
   release guard; `df_enduse_refactored` extended with gas backup energy, backup-furnace size,
   IECC zone, and panel pass-through columns (D7); peak column renames.
3. **Phase 3 — Measure registry and masking**: release-keyed `VALID_MENU_MPS`/`EQUIPMENT_SPECS`;
   `applicability` as the first funnel filter plus the funnel table (D3); the cooling filter
   removed on both releases (D4); `REBATE_ELIGIBLE_HEATING_MPS` release-aware, adding `mp=5`.
4. **Phase 4 — Consumption and degree-day adjustment**: retrofit heating as a list of
   (fuel, column) pairs (D5, part 1).
5. **Phase 5 — Capital cost**: SEER2/HSPF2 conversion (D9); ASHP + furnace upgrade cost, with
   the furnace call gated on dual fuel (D6).
6. **Phase 6 — Lifetime fuel costs**: electricity + natural-gas streams (D5, part 2).
7. **Phase 7 — Rebates**: the dual-fuel fossil-gate exception (D8).
8. **Phase 8 — NPV, adoption, ordering checks.**
9. **Phase 9 — Climate damages**, including natural-gas furnace combustion.
10. **Phase 10 — KPIs, visuals, exports.**
11. **Phase 11 — Full run, reference values, documentation** for the 2025.1 dual-fuel package.
    No MP3/MP4 re-run needed — the regression guarantee was narrowed (below) before D4 shipped, so
    there is no MP3/MP4 value move to reconcile.

Each phase becomes one Claude Code session prompt via the `vscode-refactoring-session-prompt`
skill.

---

## After Phase 11 for 05

- **Upgrade 04** — add `mp=4` to `RESSTOCK_RELEASE_AND_MP['2025.1']` and run it through the same
  phases. Most of the infrastructure will already exist, so these should be shorter sessions.
- **Upgrade 03** — download, audit, then decide on scope before running it through the phases.

## Deferred (the guide's Section 7, minus 03)

Grid-impact/peak-demand for dual fuel, the panel-upgrade cost sensitivity (D7b), the
utility-bills cross-check, AMY2012, and the v4LOW/v4HIGH cost-sensitivity refactor. The guide had
also deferred Upgrade 03 as a replacement-cost counterfactual. It is now a package to analyze in
its own right (above), and could serve both purposes.

---

## Phase 2 session progress (started 19 Sep 2026)

**Task 1 (audit, no edits) — complete.** All 89 hardcoded 2022.1.1 column literals in
`process_euss_data.py` matched a row in the column map; every 2025.1 name in the map exists in
`upgrade0.parquet` and `upgrade5.parquet`; the 8 rows marked "missing" in 2025.1 are all on
inactive paths (dryer, cooking, MP9/MP10 enclosure). Two line-number corrections to the Phase 1
audit: the peak block is at `process_euss_data.py:613-639` (not 614-638), and
`ALLOWED_TECHNOLOGIES` ends at line 60 (not 59).

**Five additional findings from Task 1, not anticipated by the session prompt, and how each was
resolved:**

1. **`preprocess_fuel_data` mutates its caller's frame in place.** Affects Task 4's before/after
   equality check — it must run on two independently loaded frames, not the same object reused
   across the old and new code paths. **Folded into Task 4.**
2. **2025.1 file names don't match `mp_to_upgrade`'s pattern** (`upgrade05` vs. the real
   `upgrade5.parquet`), and **`bldg_id` is an ordinary column in the parquet**, not a pre-set
   index the way the CSV read sets `index_col="bldg_id"`. The 2025.1 parquet read function can't
   reuse `mp_to_upgrade` as-is and must set the index itself. **Folded into Task 4.**
3. **`in.sqft` renames to `in.sqft..ft2`** in 2025.1 — already correct in the column map;
   the registry picks it up with no separate action.
4. **`mp5_modeled_savings_frac` will overstate savings until Phases 4 and 7** — it would only
   count the drop in electric heating and ignore the backup furnace's gas if computed today.
   **Not a Phase 2 task** (the guide places this computation in Phase 4). No action now; **Phase
   4's session prompt must carry this caveat** so the metric isn't treated as real until the
   fuel-cost generalization (D5) and the rebate fossil-gate exception (D8) both land.
5. **One panel column's name shape differs from its siblings**
   (`out.params.panel_constraint_breaker_space` has no `.2023_nec...` suffix) — checked against
   the parquet schema directly and confirmed correct as published. No action.

**Task 2 (release constants) — applied and verified.** `RESSTOCK_RELEASE = '2022.1.1'` and
`RESSTOCK_RELEASE_AND_MP = {'2022.1.1': [3, 4], '2025.1': [5]}` added to `constants.py` after
`VALID_MENU_MPS`. Verified: both constants import cleanly, `VALID_MENU_MPS` is unchanged at
`[0, 3, 4]`, `git diff --stat` shows only `constants.py` (+12 lines).

**Task 3 (column registry, `cmu_tare_model/utils/resstock_schema.py`) — proposed, approved.**
One deviation from the session prompt's literal wording, judged correct: the prompt said to build
the 2022.1.1 map from the column map's confirmed/renamed/new rows, which would have silently
dropped the 8 "missing" rows and broken Task 3's own check (7 of those 8 are literals
`process_euss_data.py` still reads today, just on inactive paths). Resolved by including a row in
a release's map whenever that release has a physical name for it, which reduces to the same
confirmed/renamed/new set for 2025.1. Reads the CSV at import time, matching the existing GEA
crosswalk pattern.

**Task 4 (route baseline reads through the registry) — applied and verified.** Every raw
literal in `df_enduse_refactored` replaced by a `resstock_col(release, logical_name)` lookup;
`read_resstock_2025_1_parquet(mp)` added. Verified byte-identical against two independently
loaded 2022.1.1 frames (`preprocess_fuel_data`'s in-place mutation, folded in per the Task 1
finding, made that check meaningful). At the time, judged as needing no column-provenance-rule
rework since `df_enduse`'s destination keys are TARE's own release-invariant vocabulary
(`base_heating_fuel`, `square_footage`, ...), not a raw ResStock name.

**Correction, 19 Sep 2026: that check missed the two baseline peak columns.**
`df_enduse['base_peak_electricity_heating_kw']` and `['base_peak_electricity_cooling_kw']` are
read via `resstock_col(release, 'peak_electricity_heating'/'peak_electricity_cooling')` — the
same physical pair Task 7 was about to remap (`out.electricity.peak_when_heating.kw` /
`peak_when_cooling.kw` in 2022.1.1, `out.qoi.electricity.maximum_daily_peak_winter..kw` /
`..._summer..kw` in 2025.1). Per the corrected rule above (see "Column provenance naming"), these
are NOT the same measurement across releases (audit E.2), so sharing one destination name for
both was wrong regardless of the byte-identity check passing — that check only confirmed
2022.1.1 was untouched, it never validated what the 2025.1 read actually means. **Needs its own
fix, as a new stop-gated diff**: split `base_peak_electricity_heating_kw`/`_cooling_kw` so they
are populated only from 2022.1.1's `peak_when_heating`/`peak_when_cooling`, and add new,
distinctly-named 2025.1 columns (something reflecting ResStock's own `winter`/`summer` term, not
`heating`/`cooling`) populated from 2025.1's `maximum_daily_peak_winter`/`_summer`. Not yet
applied — flagging for the next message to Claude Code.

**Task 5 (dual-fuel parser and `df_enduse_compare` changes) — revised before applying.** The
proposed diff added `RETROFIT_HEATING_FUELS = {3: ['electricity'], 4: ['electricity'],
5: ['electricity', 'natural_gas']}` to `constants.py` as a preview for Phases 4 and 6. **Pulled
from the diff**: it is out of scope for Phase 2 (the session prompt's own "out of scope" list
excludes fuel costs), and, per the researcher's 19 Sep feedback, a hand-maintained per-mp fuel
table is more hardcoding than the fuel-cost generalization (D5) needs — see D5's revised
implementation note, above, for the preferred design (reuse `get_all_possible_fuel_columns`
against the retrofit's own consumption columns, the same way the baseline side already handles
multiple fuels; zero-valued fuel columns drop out of the sum on their own). The rest of Task 5 —
`parse_dual_fuel_heating_efficiency`, the optional `df_cooking_range` parameter, the `release`
parameter and MP3-override release guard on `df_enduse_compare` — is unaffected by this and
proceeds as proposed. `backup_afue` returned as a fraction (0.925/0.95) to match REMDB v4's own
AFUE convention — confirmed, keep as proposed.

**Task 6 — applied and fully verified** (`df_enduse_refactored` and `df_enduse_compare`
byte-identity on 2022.1.1; 2025.1 dual-fuel gas-backup, backup-size, and panel-constraint columns
match the audit's figures exactly). No naming-rule issues — the three pass-through columns each
got a new, deliberately chosen TARE-side name, per Task 6's own instruction.

**Task 7 (peak column rename) — not yet run; corrected twice before any diff was written.**
First pass: read the "no behavior change, no computation" wording as meaning these four columns
should share one destination name across releases, same as any other simple rename. **The
researcher corrected this**: `renamed` in the column map is not the same as "same measurement" —
audit finding E.2 already shows 2022.1.1's `peak_when_heating`/`peak_when_cooling` (conditioned on
the equipment running) and 2025.1's `maximum_daily_peak_winter`/`_summer` (conditioned on the
calendar season) are genuinely different things that ResStock's own crosswalk happens to pair up.
Forcing them under one name would hide that difference, which is the opposite of this rule's
purpose. **Corrected plan**: each release keeps its own destination name, reflecting that
release's own ResStock term — "original name" means ResStock's term, since ResStock is the
source, not a name TARE invents to paper over a difference. 2022.1.1's existing
`heating`/`cooling`-based names are untouched; 2025.1 gets its own new names built from
`winter`/`summer` (matching ResStock's own 2025.1 term) rather than reusing `heating`/`cooling`,
which would misrepresent the 2025.1 column as equipment-conditioned when it isn't. The E.2 caveat
still goes in the code comment and the Tepper data dictionary, but now explains why there are two
names, not why one name covers two things.