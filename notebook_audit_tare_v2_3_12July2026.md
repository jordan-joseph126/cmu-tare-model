# Notebook Audit -- tare_model_main_v2_3 + tare_scenarios_v2_3 (12 July 2026 exports)

## Verdict on the prior session's implementation

Partially successful.
- DONE: REMDB v3 cleanup (loop clean, nine-case terminology refreshed in the scenarios export).
- DONE: ENERGY STAR MP3 override (MP3 draws HEEHR rebates; updated spec text in the .ipynb).
- DONE: dotplot 3-marker stagger (pages 110-111 render without marker/label overlap).
- NOT DONE: v4LOW/v4HIGH re-enable (run prints "Calculated 1 scenarios: ['v4MID']").
- UNRESOLVED: negative cooling savings unchanged (549993: -1890.05; 549999: -840.47) and no
  accept/floor/flag decision recorded anywhere visible.
- SUSPECT: June 2026 rebate fuel gate (P0-1 below) -- the [PASS] is a self-confirming test.

## Per-step status

| Section | Status | Exec time | Notes |
|---|---|---|---|
| Scenarios: fuel costs, capital (v4MID), rebates, NPV, adoption | Done | MP0 1:40; MP4 5:36 | Nine adopter columns confirmed |
| Scenarios: v3 cleanup | Done | - | One stale "less WTP and more WTP" string (L315) |
| Scenarios: tail verification cells | Partial | - | Near-duplicates; docstrings claim "NOT part of the .ipynb" yet both executed |
| Main: adoption rates, choropleths, dotplots | Done | - | Choropleth cell duplicated (vmax 50 then 100) |
| Main: rebate funding summary (cell 19) | Suspect | - | Runs only via %run -i leakage; output shows gate inversion fingerprint |
| Main: Tepper CSV exports | Partial | - | 5 counties wrong; WARN fires for all 3,098 counties |
| Main: grid impact (BSQ, Allegheny) | Done | 177.3 / 163.8 / 163.0 s queries | Steps 5-7 PASSED; 8760-hr asserts held |
| Main: tail diagnostics + placeholders | Stub | - | 3 PLACEHOLDER cells; dead .columns cells; inventory cell depends on leaked vars |

## Library log audit

- WARNING: 11 home(s) have no Cambium GEA region ['46102'] -- Oglala Lakota SD (FIPS changed
  from 46113 in 2015); crosswalk predates it; 11 homes get NaN climate damages. Document or patch.
- DtypeWarning at data_loading.py:188/222 -- mixed-dtype CSV columns; benign; specify dtypes to
  silence. No action required now.
- [WARN] home_count disagrees for 3098 county(ies) -- load-bearing; see P0-2 and P1-5.

## Cross-reference findings (code vs output)

1. JUNE 2026 FUEL GATE LIKELY INVERTED. June 2026 HEEHR total_eligible = 89,892,317,260;
   2024-guidance Electricity-baseline subtotal = 89,892,317,294 -- equal to within $34 of $90B
   (4e-10 relative; smallest possible per-home change is ~$970K weighted, so the home set is
   identical). June 2026 eligibility == electric-baseline homes exactly. Household samples:
   bldg 1 (propane, LMI) and bldg 8 (LMI) drop $8000 -> $0/'None'; electric-baseline LMI homes
   549995/549998 keep $8000 HEEHR; bldg 5 (MUI) gains $4000 HOMES. HEEHR (IRA 50122) targets
   LMI fossil-baseline electrification; HOMES (50121) is fuel-neutral -- $0 to every gas/oil/
   propane home under BOTH programs is the inversion signature. The verification cell asserts
   "fossil rows MUST be 0", i.e. the implementation, so it cannot catch the bug. Blast radius:
   June 2026 adoption 10-14 pp below 2024 everywhere (MP4 coolingLCC_sub 51.0% -> 37.3%; MP3
   29.7% -> 19.5%). CAVEAT: adjudicate against the actual June 2026 notice text -- if the notice
   genuinely restricts to electric baselines, the implementation is correct.
2. FIVE COUNTIES EXPORT WRONG HOME COUNTS in Tepper CSVs (both MP3 and MP4): G4200170 (Bucks PA,
   771 vs 770 homes), G4201010 (Philadelphia), G2405100 (Baltimore City), G5107600 (Richmond
   City), G3400070 (Camden NJ) -- adoption vs demand differ by exactly one household weight.
   All independent-city/city-county FIPS edge cases; a crosswalk/join drops or reassigns one
   building per county in one aggregation path.
3. Bare `df` in main cell 19 exists only via %run -i leakage; NameError in the 'N'
   (load-from-disk) path. Same for the tail inventory_many cell. importlib.reload cell is
   interactive hot-patch scaffolding.
4. FIGURE_DPI undefined in main (used at the rebate-policy dotplot save); latent NameError
   masked by SAVE_FIGURES=False.
5. Weight precision mismatch: adoption path uses weight rounded to 2dp (242.13); demand path
   uses full precision (242.131012...). WARN fires for all 3,098 counties, burying the five
   real discrepancies.
6. .py <-> .ipynb drift: .ipynb has updated MP3 header "(15 SEER1, 9 HSPF1) -> (16 SEER1,
   9.5 HSPF1) for ENERGY STAR" and an edited cbar_label; .py export, HEATING_MP_SUBTITLES, and
   grid-impact mp_labels still show the old spec.

## Structural findings

Duplicated choropleth cell (main L417-444 vs 446-474; only vmax differs, both ran); two
near-duplicate verification cells with self-contradictory docstrings; globals()-scanning
Block 1 and bare-df Block 2 diagnostics; stale "SIX economic-adopter columns" comment (main
L351); "less WTP and more WTP" (scenarios L315); VERBOSE = True hardcoded mid-notebook
(scenarios L448); grid-impact 2x2 figure hardcode breaks when MP8-10 activate; mp_labels only
covers MP3/4; 3 PLACEHOLDER cells and dead .columns cells; fuel_counts_millions hardcodes
* 242 instead of the weight column.

## Performance profile

| Step | Operation | Time | Binding resource |
|---|---|---|---|
| BSQ baseline timeseries (Allegheny) | Athena query | 177.3 s | DB compute / network |
| BSQ MP3 / MP4 upgrade timeseries | Athena query | 163.8 / 163.0 s | DB compute / network |
| Scenario runs (MP0 / MP3 / MP4) | Full pipeline | 1:40 / ~6:00 / ~11:30 | CPU (pandas) |
| Main notebook total | - | 29:50 | BSQ + scenario runs |

No pathological cells. BSQ is Allegheny-only; nationalizing would scale per county.

## Prioritized issue list

P0 -- Correctness
1. June 2026 fuel-gate direction + self-confirming test. Verify against spec; if inverted, fix
   gate AND test together; value-moving, supersede golden rows. Verify by re-running the
   funding summary: by_fuel must match the adjudicated spec.
2. Five-county one-home Tepper discrepancy. Trace the dropped/reassigned bldg_id per county;
   one exported table is wrong. Verify by re-export: five counties agree, others unchanged.

P1 -- Scaling / reliability
3. Fresh-run NameErrors in main 'N' path (bare df, hardcoded adopter string, inventory cell,
   reload scaffolding). Verify: top-to-bottom 'N' run succeeds.
4. FIGURE_DPI undefined. Verify: SAVE_FIGURES=True path executes.
5. Dual weight sources + tolerance-free WARN (fires 3,098 times). Unify weight source; WARN
   only above ~0.5 home-weight. Verify: WARN lists zero counties post-fix.

P2 -- Code quality
6. Duplicated choropleth cell. 7. Stale strings + backport drift (MP3 spec text, cbar_label).
8. Tail scaffolding -> one spec-driven test. 9. 2x2 grid + mp_labels hardcodes. 10.
Placeholders / dead cells. 11. VERBOSE hardcode. 14. * 242 hardcode.

P3 -- Nice-to-have
12. Document "2024 = HEEHR only" asymmetry in the comparison cell. 13. 46102 GEA crosswalk
entry or documented exclusion. 15. Dtype specs in data_loading.py CSV reads.

## Recommended next steps

1. Run the session prompt's Task 1 audit and Task 2 adjudication (rebate gate) -- everything
   downstream of rebates is provisional until this lands.
2. Fix the Tepper five-county alignment and weight unification (Task 4).
3. Reproducibility + cleanup (Tasks 5-6), then backport to the .ipynb and update CLAUDE.md +
   SESSION_CHANGELOG.md as part of session completion.
