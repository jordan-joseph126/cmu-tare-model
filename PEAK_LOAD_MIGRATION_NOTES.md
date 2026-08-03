# PEAK_LOAD_MIGRATION_NOTES.md

Change ledger for the post-TARE peak-load analysis: the move from the standalone
`grid_impact/calculate_postTARE_ts_aws_peak_demand.ipynb` (archived per the
23 July 2026 session) into a section of the main model-run notebook plus the
supporting module `cmu_tare_model/grid_impact/peak_load_functions.py`.

Documentation only. No code, notebook, or constant was changed in this session.
Where a real bug or stale reference was found it is recorded here and, if it can
misbehave, added to the deferred-fix list -- it was NOT fixed.

---

## VERDICT -- did anything important fail to survive the move?

The purpose of this ledger is to answer one question: did anything important fail
to survive the move from the standalone notebook to the live main-notebook
section? Every difference is classified as one of:

- (a) intended and complete
- (b) intended but incompletely applied
- (c) unintended loss
- (d) pre-existing issue unrelated to the migration

Lead items -- these are the only ones that need a decision from you:

| # | Class | Item | Why it needs a decision |
|---|---|---|---|
| V1 | (b) | `extract_adopter_ids` left in place: still defined (peak_load_functions.py:96), still imported by the main notebook (export:71), never called in the live path; its tier-string defaults no longer match the float-valued adoption column, and its unit tests validate the retired tier shape. | The migration replaced the CALL but not the function, import, or tests. Decide: delete the function + tests + import, or re-point them at the float column. |
| V2 | (b) | Geometry repoint incomplete: the live county-lookup cell still declares `TIGER_FIPS_COL` / `TIGER_NAME_COL` / `TIGER_STATEFP_COL` and comments "same TIGER tl_2025_us_county shapefile" while actually reading `cb_2021_us_county_500k` (export:1367-1408). The 23 Jul CLAUDE.md log says these became `CENSUS_*`. | The `CENSUS_*` rename did not reach this cell. Decide: apply the rename here, or correct the log entry. (Open item to you; CLAUDE.md not edited.) |
| V3 | (c) | Step 5 dropped the BSQ weight diagnostic: archived printed `BSQ weight : {weight_val:.6f}` read from `units_count` (archived export:496,504); the live Step 5 summary omits it (export:1202-1217). | This print was the reader's proof that the 242.131013 weight was applied by BSQ. Its loss is a candidate unintended loss. Decide whether to restore the diagnostic (do not restore this session). |
| V4 | (c) | Allegheny fuel table dropped the percentage-sum guardrail: archived asserted `abs(_pcts.sum() - 100) < 1e-6` and printed a "sums to 100.00%" confirmation (archived export:820-821,875); the live cell computes `_pcts` but has no assert and no confirmation (export:1450). | A silently mis-summing column would no longer be caught. Candidate unintended loss. Decide whether to restore (do not restore this session). |
| V5 | (d) | Duplicate `ELEC_TOTAL_COL`: `constants.py:404` (no `.kwh`, BSQ) vs `data_loading.py:211` (`.kwh`, CSV). Pre-existing, not caused by the move. | Latent footgun for future imports. On the deferred-fix list. |

Everything else is (a) intended and complete -- see the ledger and the two sweeps
below. In particular: the adopter scheme (NPV >= 0), the 2-level
`DATAFRAMES_BY_MP` indexing including the Allegheny cell, the `input()`-driven
`location_id`, the 240.0 weight removal, and the descope of Steps 9-10 are all
complete in the live path (Step 8 is preserved/planned, not descoped). No (a)
item needs a decision.

## Sources compared

- Live source modules: `cmu_tare_model/grid_impact/peak_load_functions.py`,
  `cmu_tare_model/adoption_kpis/data_loading.py`,
  `cmu_tare_model/adoption_kpis/demand.py`, `cmu_tare_model/constants.py`.
- Live main-notebook peak-load section: cited by section heading and cell name.
  A secondary line-number reference is given for convenience, explicitly stamped
  "as of tare_model_main_v2_3_EXPORT_23July2026.py". Export line numbers are NOT
  stable across re-exports of the notebook; the section heading and cell name are
  the durable reference.
- Historical reference (archived, do not treat as live):
  `grid_impact/calculate_postTARE_ts_aws_peak_demand_EXPORT_23July2026.py` and the
  archived `.ipynb` of the same stem.

## Citation convention

- Real Python modules: `file:line`.
- Main notebook: section heading + cell name, then
  "as of tare_model_main_v2_3_EXPORT_23July2026.py:LINE" as a secondary pointer.
- The archived export is cited as `archived export:LINE` for context only.

---

## Resolved question 1: which adoption scheme does the live code run?

The live pipeline uses the canonical NPV >= 0 economic-adopter scheme. The
Tier 1 / Tier 2 / Tier 3 / Tier 4 framing is superseded and does NOT run in the
live path.

Live selection (main notebook, section "GRID IMPACT ANALYSIS", cell
"build adopter building IDs by measure package and county"; as of
tare_model_main_v2_3_EXPORT_23July2026.py:1078-1099):

```python
adoption_col = find_adoption_column(df_tare, mp, cost_scenario)
county_fips  = df_tare['county'].apply(gisjoin_to_fips)
is_adopter   = df_tare[adoption_col] == 1.0
adopter_ids_by_mp[mp][str(fips)] = {
    "all_filtered": list(idx),
    "constrained":  list(idx[adopter_mask]),
}
```

`find_adoption_column` (peak_load_functions.py:74-78) builds
`ref2025_mp{mp}_heatingLCC_coolingSavings_sub_econ_adopter_fixed_base`
(default `npv_case='heatingLCC_coolingSavings_sub'`, `discount_rate_key='fixed_base'`).
There is no `iraRef`, `v4MID`, `inmap`, or WTP token. The `cost_scenario` argument
is accepted for signature compatibility and ignored when building the name.

"constrained" now means economic adopters (`econ_adopter == 1.0`, i.e. NPV >= 0),
NOT Tier 1 + Tier 2. "all_filtered" is every filtered building in the county
(the 100 percent-adoption bound).

The `iraRef_mp{mp}_heating_adoption_central_inmap_acs_v4MID_fixed_base` +
Tier 1-4 description in the session brief corresponds to an OLDER archived
notebook. Even the 23 July archived export had already moved its adopter lookup to
the canonical `find_adoption_column` (archived export:388) -- but it still fed that
canonical float-valued column into the tier-based `extract_adopter_ids`
(archived export:408) and printed a "Tier distribution" (archived export:401-405).
That archived export is therefore internally half-migrated. The live main notebook
completed the migration by replacing the `extract_adopter_ids` call with the inline
`== 1.0` mask.

## Resolved question 2: county-map geometry vintage

The live county map loads the 2021 cartographic boundary file, not TIGER 2025.

`gdf_counties_raw = gpd.read_file(COUNTY_SHAPEFILE_PATH)` (main notebook,
"County shapefile" load cell; as of
tare_model_main_v2_3_EXPORT_23July2026.py:351). `COUNTY_SHAPEFILE_PATH` derives
from `COUNTY_GEOMETRY_PRODUCT = "cb"`, `COUNTY_GEOMETRY_VINTAGE = 2021`,
`COUNTY_GEOMETRY_SCALE = "500k"` (data_loading.py:148-189), resolving to
`cb_2021_us_county_500k`. 2021 is the newest cb vintage that still carries
Connecticut's eight pre-2023 counties, matching ResStock 2022.1.1 geography.

The archived notebook hardcoded `tl_2025_us_county/tl_2025_us_county.shp` and the
`TIGER_*` column constants. Neither the `tl_2025` path nor the `TIGER_*` names are
the live data source. See the open item below: the `TIGER_*` symbol names and the
"tl_2025" comment still linger inside one live main-notebook cell even though the
file they read is now cb_2021.

---

## Change ledger

Columns: category | old symbol / location | new symbol / location | file:line |
can move a computed value?

### Moved to module

| Old (archived standalone notebook) | New (live) | Location | Value-moving? |
|---|---|---|---|
| Inline `gisjoin_to_fips` | `gisjoin_to_fips()` | peak_load_functions.py:18 | No |
| Inline adoption-column lookup | `find_adoption_column()` | peak_load_functions.py:41 | No (name builder only) |
| Inline profile aggregation | `compute_county_scenario_profile()` | peak_load_functions.py:148 | No (same math) |
| Inline plotting | `plot_demand_panel()` | peak_load_functions.py:231 | No |
| Inline adopter-ID build | now in main-notebook cell "build adopter building IDs..." | export:1066-1109 | No by itself |

### Renamed

| Old | New | Location | Value-moving? |
|---|---|---|---|
| Adoption column `iraRef_mp{mp}_heating_adoption_central_inmap_acs_v4MID_fixed_base` | `ref2025_mp{mp}_heatingLCC_coolingSavings_sub_econ_adopter_fixed_base` | peak_load_functions.py:74-78 | Yes (different column, different homes) |
| County columns `TIGER_FIPS_COL` / `TIGER_NAME_COL` / `TIGER_STATEFP_COL` | intended `CENSUS_*` per 23 Jul session log | NOT renamed in the live main-notebook cell -- see open item | No (same GEOID/NAME/STATEFP values) |

### Behavior changed

| Old | New | Location | Value-moving? |
|---|---|---|---|
| Adopter selection = tiered (`extract_adopter_ids`, Tier 1 + Tier 2) | Adopter selection = `econ_adopter == 1.0` (NPV >= 0), inline | main notebook cell "build adopter building IDs..."; export:1092 | Yes -- changes which buildings are "constrained" |
| `DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']` (3-level, RCM-nested) | `DATAFRAMES_BY_MP[mp]['fixed_base']` (2-level) | data_loading load cell + Allegheny fuel table cell; export:1437 | Yes for any cell copied forward (a stale 3-level index raises `KeyError` now) |
| `LOCATION_ID = "National"`, `MODEL_RUN_DATE_TIME = "2026-04-10_00-05"` hardcoded module constants (archived export:126-127) | `location_id` / `model_run_date_time` set by `input()` at run time | main notebook "start_new_model_run == 'N'" cell; export:155-156 | Yes indirectly -- see QA note; a wrong entry loads the wrong results file |
| Weight hardcode `240.0` | BSQ applies per-row weight in SQL (`SUM(enduse x weight)`); MW = kWh / 1000, no extra multiply | peak_load_functions.py:192-202; note at :179 | Yes vs the old code (about 0.9 percent); already corrected before this session |

### Removed (from the live call path)

| Old | Status now | Location | Value-moving? |
|---|---|---|---|
| `extract_adopter_ids` in the pipeline | Still DEFINED and still imported by the main notebook, but never CALLED in the live path; only the archived export and the unit tests call it | peak_load_functions.py:96; import at export:71 | No in live path (it is dead there). See TRAP list. |
| Tier 1-4 distribution printout | Not present in the live path | -- | No |

### Added

| New | Location | Value-moving? |
|---|---|---|
| `peak_dict` with `peak_hour_baseline`, `peak_hour_scenario`, `baseline_peak_mw`, `scenario_peak_mw`, `delta_mw`, `n_adopters`, `n_total_buildings` | peak_load_functions.py:212-226 | No (report fields) |
| Two-set county structure `{all_filtered, constrained}` built inline | main notebook cell "build adopter building IDs..."; export:1097-1100 | No |
| Peak metric `delta_mw = max(scenario) - max(baseline)` over 8,760 hours; the two maxima may fall in different hours | peak_load_functions.py:221-223 | No (definition) |

### Descoped by decision (national scaling no longer required or feasible at present)

These existed as `NotImplementedError` stubs in the archived notebook. They were
NOT ported into the live pipeline. Per the researcher, this is a deliberate
descope, not a loss and not an open TODO. The live pipeline computes peak load for
a single validated county (Allegheny, FIPS 42003) and stops.

| Step | What it would have done | Archived stub |
|---|---|---|
| Step 8 | Validate the profile-derived county peak against the EUSS metadata peak columns (`out.electricity.winter.peak.kw` / `summer.peak.kw`), flagging a mismatch above 20 percent before any scale-up. | archived export:902 |
| Step 9 | Loop the single-county peak computation over all ~3,098 counties to build a national county-level peak-change table, with per-state checkpointing and a batching decision. | archived export:955 |
| Step 10 | Export the national county table to CSV for the Section 3.6 figures. | archived export:975 |

Status distinction (per researcher, 24 Jul 2026):
- Steps 9 and 10: DESCOPED (national county-level results are not currently
  required or feasible).
- Step 8: PRESERVED -- a retained, planned validation, NOT descoped and NOT
  merely pending. It sums the EUSS metadata peak columns
  (`out.electricity.winter.peak.kw` / `out.electricity.summer.peak.kw`) across a
  county's buildings and compares that to the profile-derived peak, flagging
  divergence above 20 percent for investigation before any scale-up. It is a
  METADATA read, not a timeseries scan, so the Steps 9/10 descoping does not
  touch it. Consequence: the Allegheny peak delta currently has no internal
  cross-check and no external benchmark; Step 8 is the planned internal check.
  The guide's known-limitations section must not imply the reported number has
  been validated.

### Still stubbed in live code

None. The live pipeline has no `NotImplementedError` stubs. The stubs exist only
in the archived export / archived notebook.

---

## Stale references

### PROSE (record only -- cosmetic, cannot misbehave)

| Where | What is stale | Correct reading |
|---|---|---|
| peak_load_functions.py:3-4 (module header) | "Used by the notebook and by the national loop (Step 9)." | No national loop exists in live code (Step 9 was descoped). |
| tests/adoption_kpis/test_peak_load_functions.py:1 (docstring) | "Tests for cmu_tare_model.adoption_kpis.peak_load_functions module" | The module lives at `cmu_tare_model.grid_impact.peak_load_functions`; the test imports already use the correct path. |

### TRAP (record + on the deferred-fix list; each has a failure mode)

| Where | Problem | Failure mode |
|---|---|---|
| peak_load_functions.py:96-145 + import at export:71 + tests/adoption_kpis/test_peak_load_functions.py:110-180 | `extract_adopter_ids` is imported-but-uncalled dead code in the live path. Its defaults `tier_1_value="Tier 1: Feasible"` / `tier_2_value="Tier 2: Feasible vs. Alternative"` are string labels, but the live adoption column is float-valued (1.0 / 0.0). | If anyone re-wires the pipeline to call `extract_adopter_ids` on the live column, every `tier1`/`tier2`/`constrained` list comes back EMPTY (no float equals a tier string), silently yielding zero constrained adopters. The unit tests pass only because they feed it a synthetic tier-STRING column -- they validate a retired data shape, so they give false confidence. |
| constants.py:404 vs data_loading.py:211 | Two different constants share the name `ELEC_TOTAL_COL`: `"out.electricity.total.energy_consumption"` (no `.kwh`, for BSQ) in constants.py, and `"out.electricity.total.energy_consumption.kwh"` (for CSV) in data_loading.py. Both are correct in their own context. | A future import from the wrong module gets the wrong string. The BSQ path needs the no-suffix name (line 404); the CSV path in demand.py needs the `.kwh` name (line 211). A silent swap would either miss the column (KeyError) or read the wrong series. |

Deferred-fix entries (do NOT fix this session; one diff per gate when greenlit):
1. Reconcile `extract_adopter_ids`: either delete it and its tier-shape tests, or
   re-point it (and the tests) at the float-valued econ-adopter column. Remove the
   unused import at export:71 in lockstep.
2. Disambiguate the duplicate `ELEC_TOTAL_COL` (e.g. rename the BSQ one to
   `BSQ_ELEC_TOTAL_COL` or similar) so no module can import the wrong one.

---

## Open items addressed to the researcher

1. TIGER_* -> CENSUS_* rename discrepancy. The 23 July 2026 CLAUDE.md session-log
   entry states the county-column symbols "became `CENSUS_*` (the product is no
   longer TIGER)" in the export. The live main-notebook cell
   "GRID IMPACT -- county geography lookup" still declares
   `TIGER_FIPS_COL` / `TIGER_NAME_COL` / `TIGER_STATEFP_COL` and comments that it
   reuses "the same TIGER tl_2025_us_county shapefile"
   (as of tare_model_main_v2_3_EXPORT_23July2026.py:1367-1408). The code works
   because `cb_2021_us_county_500k` also carries GEOID / NAME / STATEFP, but the
   symbol names and the comment are stale versus the geometry repoint. The
   `CENSUS_*` rename evidently did NOT reach this main-notebook cell. Recorded per
   instruction; CLAUDE.md was not edited. Please confirm whether the rename should
   be applied here or the log entry corrected.

2. Step 8 decision. Whether to implement the peak-vs-metadata validation
   (Step 8) is left to you. It is held as pending, not descoped.

---

## Sweep 3a -- Guardrail inventory

Every assert, raise, validation guard, and diagnostic print in the archived
export, marked PRESENT / ABSENT / CHANGED in the live main-notebook section.
"CHANGED" means present but altered (e.g. glyph or wording). Live line numbers are
"as of tare_model_main_v2_3_EXPORT_23July2026.py". An ABSENT item is a candidate
unintended loss -- flagged, NOT restored.

| Guardrail | Archived loc | Live status | Live loc | Class |
|---|---|---|---|---|
| AWS creds -> RuntimeError (NoCredentialsError / ClientError) | export:212-218 | PRESENT | export:1133-1139 | (a) |
| STS get_caller_identity success print | export:205-210 | PRESENT | export:1127-1132 | (a) |
| Shapefile expected-columns KeyError guard | export:296-301 | PRESENT | export:1377-1382 | (a) |
| TEST_FIPS present-in-shapefile ValueError | export:327-331 | PRESENT | export:1411-1415 | (a) |
| Baseline 8,760-hour assert (min AND max) | export:507-508 | PRESENT | export:1215-1216 | (a) |
| float32 downcast, baseline kWh | export:483 | PRESENT | export:1189-1191 | (a) |
| float32 downcast, upgrade kWh | export:549 | PRESENT | export:1239-1241 | (a) |
| Deterministic hour index (sort by bldg_id,timestamp -> cumcount+1) | export:486-491, 552-557 | PRESENT | export:1192-1197, 1242-1247 | (a) |
| Schema parity: compute only_in_baseline / only_in_upgrade | export:560-563 | PRESENT | export:1249-1254 | (a) |
| only_in_upgrade -> ValueError ("in upgrade but not baseline") | export:581-584 | PRESENT | export:1273-1277 | (a) |
| only_in_baseline fallback-to-baseline note | export:578-580 | PRESENT | export:1268-1272 | (a) |
| Upgrade Hours/bldg min-max print | export:570 | PRESENT | export:1259-1262 | (a) |
| Step 7 profile 8,760-row asserts (both scenarios) | export:651-652 | PRESENT | export:1326-1327 | (a) |
| compute_county_scenario_profile internal 8,760 ValueError (shared module) | peak_load_functions.py:206-210 | PRESENT | same module | (a) |
| Query timing prints (perf_counter) | export:462/476/505, 531/545/576 | PRESENT | export:1171/1184/1213, 1223/1234/1267 | (a) |
| Per-step PASSED prints | export:509/589/654 | CHANGED (glyph only, "[OK] ... PASSED") | export:1217/1281/1329 | (a) |
| BSQ weight value print (units_count, ".6f") | export:496, 504 | ABSENT | -- | (c) candidate loss (V3) |
| Heating-fuel pct-sum assert (abs(sum-100) < 1e-6) | export:820-821 | ABSENT | -- | (c) candidate loss (V4) |
| "Verification: all pct columns sum to 100.00%" print | export:875 | ABSENT | -- | (c) candidate loss (tied to V4) |
| display(df.head()) after Step 5 / Step 6 | export:510, 587 | ABSENT | -- | (a) cosmetic (intended for non-interactive .py) |
| Step 8 / 9 / 10 NotImplementedError | export:902/955/975 | ABSENT | -- | (a) descoped (Steps 9,10); Step 8 PRESERVED / planned validation |

Note on "8,760 on upgrade": neither the archived nor the live code has a
standalone 8,760 assert on the UPGRADE timeseries. Both print the upgrade
Hours/bldg range, and the 8,760 guarantee for the upgrade profile is enforced
downstream by the Step 7 profile asserts and the module's internal ValueError.
This is unchanged by the migration (a).

## Sweep 3b -- Explanatory-content inventory

Markdown / prose in the archived notebook that is source material for the guide.
These are NOT defects; they are content to transcribe (verified) into
TARE_Peak_Load_Analysis_Guide.md in Task 4, rather than rewrite from memory. Each
row gives the target guide section. Items marked "do NOT carry" are retired or
unverified and must not be reproduced.

| Archived prose | Archived loc | Carry into guide? | Target guide section |
|---|---|---|---|
| Scenario definitions (100% adoption vs economically constrained) + Allegheny test case + ResStock 2022.1.1 scope, 2025.1 future | export:1-19 | YES (reword "constrained" as NPV >= 0, not Tier 1+2; state single-county scope) | "What the analysis computes / Section 3.6 scope" |
| Prerequisites: `pip install git+https://github.com/NREL/buildstock-query.git`; `aws configure` region us-west-2; IAM `AmazonAthenaFullAccess` + `AmazonS3ReadOnlyAccess` | export:162-165 | YES | "AWS account setup / IAM / region" + "Installing and verifying BSQ" |
| BSQ parameter table (workgroup / db_name / table_name / db_schema / buildstock_type) | export:167-174 | YES (confirm values against the live init at export:1141-1148) | "Athena workgroup" + "Running the section step by step (BSQ init)" |
| Weight note: uniform 242.131013, `SUM(enduse x weight)` applied by BSQ, 240.0 hardcode removed | export:176-179 | YES | "Sampling weight -- do not double-apply" (units) |
| Known issue: `split_enduses=True` -> Pydantic ValidationError; use `split_enduses=False` | export:181-183, 448 | YES | Troubleshooting table + BSQ query step |
| GISJOIN format: `G` + 2-digit state FIPS + `0` + 3-digit county FIPS; Allegheny = G4200030 -> 42003 | export:262-269 | YES | County mapping step (gisjoin_to_fips) + "What each output contains" |
| BSQ-generated-SQL explanation: join ts<-metadata, date_trunc hour rollup, SUM(enduse x weight) weight-applied, per-building when group_by=[bldg_id] | export:442-448 | YES | "Running step by step (baseline timeseries)" + units/weight |
| Upgrade query + applicability + fallback-to-baseline for buildings with no upgrade data | export:519-521 | YES | "Running step by step (upgrade timeseries)" + "What each output contains" |
| Step 7 logic: per building/hour adopter mask; two profiles; peak metric = max(scenario) - max(baseline); units note (weight-applied kWh = raw x 242.131013, /1000 -> MW) | export:599-610 | YES (reword the two profiles as all_filtered vs NPV>=0 constrained) | "How to read the demand figure and peak delta" + peak-metric definition + units |
| Heating-fuel distribution table description | export:785-790 | YES (reword Tier 1+2 -> constrained/NPV>=0) | "What each output object contains (fuel table)" |
| Tier 1 / Tier 2 / Tier 3 / Tier 4 definitions | export:341-345 | do NOT carry (retired scheme) | replace with NPV>=0 / all_filtered vs constrained explanation |
| "See methodology notes in `peak_load_methodology.md`" | export:19 | do NOT carry -- SUPERSEDED document, absent from the tree (see note below) | n/a; drop the reference |

Note on `peak_load_methodology.md`: it is a superseded project document, absent
from the repo (confirmed by a repo-wide `.md` search). Beyond being missing, its
guidance to multiply aggregated kWh by the sampling weight is WRONG for the
BuildStockQuery path, where BSQ already applies the weight in the generated SQL
(`SUM(enduse x weight)`) -- multiplying again would double-apply it. Do not cite,
absorb, or vendor it. Nothing is lost: the peak-metric definition, the
kWh-to-MW units note, and the two-scenario framing are all sourced from the
archived notebook's own markdown cells (rows above), not from this document.

## Verification of citations

Each `file:line` above was confirmed by reading the current file, not from memory:
- peak_load_functions.py:18/41/74-78/96-145/148/179/192-202/212-226/231 -- read.
- data_loading.py:148-189/211 -- read.
- constants.py:404/407/410-424 -- read.
- tests/adoption_kpis/test_peak_load_functions.py:1/110-180 -- read.
- Live main-notebook cells at export:155-156/351/1066-1109/1367-1408/1437 -- read.
- Archived export:126-127/381/388/401-408/811/902/955/975 -- read.
- Archived export guardrail + prose lines (1-19/162-183/205-218/262-269/296-331/
  442-448/483/486-491/496/504/507-510/519-521/531-587/599-610/651-655/781-790/
  820-821/875) -- read.
- Live guardrail lines (export:1127-1139/1189-1197/1215-1217/1239-1254/1259-1277/
  1326-1329/1377-1382/1411-1415/1450) -- read.
- Repo-wide `.md` search confirmed `peak_load_methodology.md` is absent.
