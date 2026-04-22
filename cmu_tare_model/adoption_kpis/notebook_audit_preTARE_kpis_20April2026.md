# Notebook Audit — `calculate_preTARE_am_kpis_sparkGap_COP_20April2026`

**Audit date:** 21 April 2026
**Auditor:** Claude (via `notebook-refactor-audit` skill)
**Inputs:** `calculate_preTARE_am_kpis_sparkGap_COP_20April2026.py` + matching PDF export
**Scope:** Correctness, scaling, code quality, readiness for county-level extension

---

## Per-step status

| Step / Task | Status | Notes |
|---|---|---|
| 0 — Imports | ✅ | Imports `create_choropleth_map` but **redefines it locally at line 713** (shadowing) |
| 0b — MP selection | ✅ | `input()` + NameError trick works; fragile for CI/automation |
| 1 — Load EUSS | ✅ | 331,531 SF baseline; 331,526 after MP3/MP4 applicability |
| 2 — Spark gap (4 time windows) | ✅ | Mean 3.15 (2022), 3.55 (2024), 3.41 (5-yr), 3.55 (10-yr) |
| 2b — Summary table | ✅ | Top/bottom 5 match outline values |
| 3 — Thermal COP | ⚠️ | MP3 all in [1.5, 5.0]; **MP4: 8 states COP > 5.0 flagged, not resolved** |
| Task B — Zero-baseline filter | ✅ | MP3 44/49 (5 states +0.001 = float noise); MP4 49/49 |
| Task C — CZ benchmark | ⚠️ | MP3 3/3; **MP4 2/3 — CZ 1-3 at 5.30 exceeds [3.0, 4.2]** |
| Task C — PA spot check | ❌ | **PA CZ 6-7 fails both MPs (1.63 and 2.28 vs. expected)** |
| Task D — Jenkins cross-val | ⚠️ | 1/6 strict; 5/6 relaxed; **AK skipped** |
| 4 — Break-even merged | ✅ | 51 in prices, 49 in merged (HI/DC subset) |
| 4c — Jenkins revalidation | ⚠️ | Duplicates `jenkins_ref` from Task D (lines 409, 508) |
| 5 — Shapefile load | ⚠️ | **Alaska: 0 rows** — drops silently from maps |
| 5 — Spark gap map | ✅ | 2024 title correct |
| 5 — COP panel map | ✅ | Shared color scale works |
| 5 — Break-even @90% map | ✅ | Greens colormap |
| 5 — Break-even panel (80/90/95) | ✅ | Oranges with shared norm |
| Display results | ⚠️ | Uses `display()` — fails in `.py` runs |

---

## Library log audit

> `UserWarning: The GeoDataFrame you are attempting to plot is empty.`

**What.** Alaska shapefile subset returned zero rows; `gdf_alaska.plot(...)` on the inset axis throws a warning.
**Problem?** Not a crash, but Alaska silently drops from every map. Jenkins validation in Task D references AK = 5.69 and can't check it.
**Action.** Fix the Alaska filter upstream in `prepare_state_geodataframe`, OR explicitly document exclusion and drop AK from Jenkins reference.

---

## Cross-reference findings (code vs. observed output)

1. **MP4 warm-state COPs (CA 5.72, NV 6.31, AZ 5.82, FL 5.70) exceed literature ceiling.** Code flags ("⚠ N groups with suspicious COP"); does not drill in. A 24–29.3 SEER1 ductless *should* have high COP in mild climates, but 5.30+ starts to strain plausibility. Suspected causes: (a) small heating loads in warm states making `load/elec` ratio noisy, (b) ductless fan-motor heat accounting. **Must investigate before publication.**
2. **`primary_mp = selected_mps[0]`** (line 235) silently depends on input ordering. Runs with `[4, 3]` vs. `[3, 4]` use different primary MPs downstream. Low risk at present but fragile.
3. **Jenkins offset is systematic** — FL (+0.03), PA (+0.06), MA (+0.07), MN (+0.07), CA (+0.08) all positive. Pattern matches the 1020 vs. 1038 BTU/cf gas heat content delta (factor 0.9827). Applying this correction brings all 5 within ±0.05 strict tolerance. **Parameterizable** rather than accepting as a limitation.

---

## Structural findings

- **Duplicate function definition.** `create_choropleth_map` imported at line 50, redefined at line 713. Redefinition adds an unused `year` param and slightly different axis handling. Local version wins.
- **Duplicate data frames.** `df_prices_for_d` (line 415), `df_breakeven_for_d` (line 416) re-compute identical frames to `df_prices_csv`, `df_breakeven`.
- **Duplicate `jenkins_ref` dict** at lines 409 and 508.
- **Duplicate imports** at lines 706–711 inside a cell.
- **Empty cell** at lines 703–704.
- **Hardcoded PA ranges** at line 368 — belongs in `COP_BENCHMARK_RANGES`.
- **`display()` calls** (lines 845, 849, 865) break in non-notebook runs.
- **`df_upgrade_primary`** (line 237) assigned, never read.
- **Step 5 has three near-duplicate multi-panel choropleth blocks** — one helper would cut ~120 lines.

---

## Prioritized issue list

### P0 — Correctness

1. **MP4 warm-state COPs exceed literature ceiling.**
   *Why it matters:* Reviewers will flag a published Figure showing MP4 COP of 5–6 in the South.
   *Where:* Step 3 Task C, MP4 branch; `compute_thermal_cop` output for AL, AZ, CA, FL, GA, LA, MS, NV, SC, TX, UT.
   *Fix:* Decompose numerator vs. denominator by state; compare a sample home's MP4 COP against the 29.3 SEER1 ductless spec sheet; consider a per-CZ sanity cap with flagging.
   *Verify:* All state COPs in MP4 fall within [2.0, 5.0]; flagged states' raw heating-load and HP-electricity sums printed for audit.

2. **PA CZ 6-7 spot check fails both MPs.**
   *Why it matters:* Pittsburgh is the primary case study. If the CZ 6-7 sub-population is misrepresented, the PA numbers in Section 4.5 are unreliable.
   *Where:* Task C, MP3 (1.63 vs. [1.8, 2.4]) and MP4 (2.28 vs. [2.5, 3.4]).
   *Fix:* Print `home_count` for PA × CZ group; cross-reference ResStock's county→IECC CZ mapping against the IECC map; decide whether to update the benchmark range or investigate a sampling/assignment issue.
   *Verify:* PA CZ 6-7 home_count known; benchmark range revised with citation, OR data anomaly resolved.

3. **Alaska silently absent from all maps and from Jenkins validation.**
   *Why it matters:* AK is the highest-spark-gap state (6.44) and the Jenkins extreme anchor (5.69). Its absence weakens the cross-validation.
   *Where:* `prepare_state_geodataframe` returns `gdf_alaska` with 0 rows per PDF.
   *Fix:* Inspect the shapefile — does it contain AK? Check `gdf_states_raw['STUSPS'].unique()`. If present, the filter in `prepare_state_geodataframe` is dropping it; if not, obtain an AK-inclusive shapefile or document the exclusion.
   *Verify:* AK appears in spark gap and Break-Even COP maps; Jenkins AK row shows computed value.

4. **`create_choropleth_map` double-definition shadows the module import.**
   *Why it matters:* Future edits to the module function will silently have no effect when this notebook runs. Cross-notebook consistency breaks.
   *Where:* Line 50 import; line 713 redefinition.
   *Fix:* Remove the in-notebook redefinition; update the module version to accept `norm` kwarg if missing.
   *Verify:* Module version called (add a print at module-function entry); notebook output identical.

### P1 — Scaling

5. **Benchmark coverage is too thin.** `COP_BENCHMARK_RANGES` only covers 3 CZ groups; PA spot check hardcoded inline. At county aggregation, benchmark coverage is even sparser.
   *Fix:* Add MN, FL, MA, CA spot-check ranges to `COP_BENCHMARK_RANGES`; structure as `{state: {cz_group: {mp_key: (lo, hi)}}}`.

6. **Redundant price/breakeven computation in Task D.** Harmless at state level (51 rows); costly if these objects expand to county (~3,100 rows × scenarios).
   *Fix:* Reuse `df_prices_csv` and `df_breakeven` in Task D.

### P2 — Code quality

7. `jenkins_ref` defined twice → move to `constants.JENKINS_BREAKEVEN_REF_90`.
8. PA ranges hardcoded inline → move to `COP_BENCHMARK_RANGES`.
9. `display()` calls break non-interactive runs → guard or replace with `print(df.to_string())`.
10. Empty cell + redundant imports inside cell → delete.
11. `df_upgrade_primary` assigned but unused → delete.
12. Three near-duplicate choropleth-panel blocks → extract `create_choropleth_panel_map(gdfs_by_key, columns, cmap, shared_norm, titles, output_path)`.

### P3 — Nice-to-have

13. Use `pathlib.Path` instead of `os.path.join` strings.
14. Replace Unicode glyphs (`✓ ⚠ ✗`) with ASCII (`[OK] [WARN] [FAIL]`) for terminal compatibility.
15. Add `%%time` markers for perf baseline before county aggregation triples row counts.
16. 600 DPI PNG → 300 DPI PDF for Energy Policy figure submission.

---

## Recommended next steps

1. **Before anything else:** investigate P0.1 (MP4 warm-state COPs) and P0.3 (Alaska) — these affect the paper's defensibility.
2. **Resolve P0.2 (PA CZ 6-7)** since Pittsburgh is the primary case study.
3. **Consolidate P0.4 (function shadowing) and P2.7–9 (dict duplication, `display()`, PA ranges)** before county aggregation touches the same code paths.
4. **Only then add county aggregation** (the new work). The refactor cleanup must land first; otherwise county logic compounds the existing duplication.
5. **Add adoption potential visual + bill savings ratio map updates** in a separate session once county aggregation is verified against state-level reference values.
