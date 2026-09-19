# Phase 1 Audit -- ResStock 2025.1 Dual Fuel (Upgrade 05)

**Session:** 18-19 September 2026. **Plan this serves:**
`REFACTORING_GUIDE_17Sept2026_DualFuel_ResStock2025_1.md`.

**What this session did:** read the TARE code, downloaded the ResStock 2025.1 national
AMY2018 data, measured it, and wrote a column map. **No `.py`, `.ipynb`, or `*_EXPORT_*.py`
file was edited, and no git command was run.** The 2022.1.1 MP3/MP4 pipeline and its oracle
run `2026-09-02_19-04` are untouched.

**Files written by this session (new files only):**

- `data/resstock_2025_1/` -- 11 downloaded files plus `DATA_MANIFEST.md` and
  `_download_log.jsonl`. Git-ignored by the existing bare `data/` rule at `.gitignore:14`;
  `.gitignore` was not edited.
- `cmu_tare_model/docs/resstock_2025_1_column_map.csv` -- 123 rows.
- This memo.

**A note on the docs path.** The session prompt said to write into `docs/`. There is no
repo-root `docs/` folder in this project, and CLAUDE.md's own reference to
`docs/SESSION_CHANGELOG_2026-08-19.md` resolves to
`cmu_tare_model/docs/SESSION_CHANGELOG_2026-08-19.md`, so both deliverables were written
there and stay there. `docs/REFERENCE_VALUES.md` (CLAUDE.md:14, 22, 78, 455) and
`docs/SESSION_LOG.md` (CLAUDE.md:79) were cited by CLAUDE.md but missing at audit time; the
researcher added both on 19 September 2026, and this session's entry is in the session log.

**Weight convention.** Every count below is representative dwelling units (rdu) unless
labelled weighted. One 2025.1 rdu = 253.90367272727272 real dwellings, against 242.131013 in
2022.1.1.

---

## Decisions taken by the researcher, 19 September 2026

Taken after reading this audit's first draft. They close three of the six blockers that draft
listed, and they set the standard the later phases are held to.

**Overarching constraint: keep as much of the existing workflow as possible. Changes should
be minimal and as simple as possible.** Where this memo notes that something "is not
consistent with what the code does today", read that as a cost to be weighed, not a change to
be made by default.

**Decision 1 -- exclude Alaska and Hawaii.** The study keeps the 49-state plus DC footprint
the existing codebase already has. The reasoning is about the measures under study, not about
the data plumbing: natural gas infrastructure in Hawaii is limited and heating demand there is
close to nil, so a dual fuel package has almost nothing to act on; in Alaska gas is available
but fuel oil dominates heating, and a very high electricity-to-gas price ratio combined with
very high heating demand makes it a poor candidate for either a dual fuel or a cold-climate
heat pump relative to other states.

This is also the cheapest option in code terms. Excluding the two states removes 3,418
baseline rdu (867,843 dwellings) and leaves every remaining `in.county` value in GISJOIN form,
with exactly **3,107** distinct counties in the baseline -- the same count as the raw 2022.1.1
baseline. The FIPS slice at `process_euss_data.py:218`, the Cambium GEA crosswalk, the county
AMI join and the map layers then all work unchanged. See D10 and Section E.

**Decision 2 -- treat the peak columns as a straight rename.** The 2022 `peak_when_heating` /
`peak_when_cooling` pair and the 2025 `maximum_daily_peak_winter` / `..._summer` pair measure
the same kind of quantity: a per-home absolute maximum peak in kW, not time-aligned across
homes. Only the season term differs. The four peak pass-through columns are therefore mapped
as renames, not as missing columns. One caveat is carried into the column map and E.2 below:
the 2022 columns are conditioned on the end use running and the 2025 columns on the calendar
season, which shows up as a large difference in zero counts.

**Decision 3 -- deliverables stay in `cmu_tare_model/docs/`.**

---

## A. Repo audit (no edits)

Scope note: the current `*_EXPORT_*.py` snapshot
(`cmu_tare_model/tepper_mba_capstone/tare_model_main_v3_0_EXPORT_3Sep2026.py`) is the
post-TARE analysis notebook. The TARE run itself -- the raw EUSS load,
`df_enduse_refactored`, and `df_enduse_compare` -- lives in
`cmu_tare_model/model_scenarios/tare_baseline_v3_0.ipynb`,
`tare_scenarios_v3_0.ipynb`, and `tare_run_simulation_v3_0.ipynb`. Both are covered in
item 10.

### A.1 Literal ResStock column strings

| Location | Columns read | Note |
|---|---|---|
| `process_euss_data.py:202-230` | `weight`, `in.sqft`, `in.census_region`, `in.census_division`, `in.census_division_recs`, `in.building_america_climate_zone`, `in.reeds_balancing_area`, `in.state`, `in.city`, `in.puma_metro_status`, `in.county` (twice), `in.puma`, `in.county_and_puma`, `in.weather_file_city`, `in.weather_file_longitude`, `in.weather_file_latitude`, `in.geometry_building_type_recs`, `in.income`, `in.federal_poverty_level`, `in.occupants`, `in.tenure`, `in.vacancy_status`, `in.vintage`, `in.clothes_dryer`, `in.cooking_range` | Baseline metadata block. `county_fips` is built at line 218 as `x[1:3] + x[4:7]`, a positional slice of the GISJOIN code. |
| `process_euss_data.py:257-281` | `in.heating_fuel`, `in.hvac_heating_type_and_fuel`, `in.hvac_heating_efficiency`, `out.params.size_heating_system_primary_k_btu_h`, `out.electricity/fuel_oil/natural_gas/propane.heating.energy_consumption.kwh`, `in.hvac_cooling_type`, `in.hvac_cooling_efficiency`, `out.params.size_cooling_system_primary_k_btu_h`, `out.electricity.cooling.energy_consumption.kwh` | Baseline heating and cooling. Fossil heating is read at baseline only. |
| `process_euss_data.py:285-304` | Water heater, dryer and range `in.` and `out.` columns | Dead under the current `EQUIPMENT_SPECS` (`constants.py:64-70`). |
| `process_euss_data.py:320-362` | `out.site_energy.total.energy_consumption.kwh`, `out.electricity.peak_when_cooling.kw`, `out.electricity.peak_when_heating.kw`, `out.load.cooling.peak.kbtu_hr`, `out.load.heating.peak.kbtu_hr`, `out.electricity.total.energy_consumption.kwh` | Site total is the HOMES savings-fraction denominator. |
| `process_euss_data.py:441-495` | `in.hvac_has_ducts`, `in.hvac_heating_type_and_fuel`, `in.hvac_heating_efficiency`, `out.params.size_heating_system_primary_k_btu_h` (from `df_mp`), `upgrade.hvac_heating_efficiency`, `in.hvac_cooling_type`, `in.hvac_cooling_efficiency`, `out.params.size_cooling_system_primary_k_btu_h` (from `df_mp`), `upgrade.hvac_cooling_efficiency` | `df_enduse_compare`. Line 450 has `out.params.size_heat_pump_backup_primary_k_btu_h` commented out; line 463 has `..._secondary_...` commented out. |
| `process_euss_data.py:533, 552, 584, 587` | `out.electricity.heating.energy_consumption.kwh`, `out.electricity.cooling.energy_consumption.kwh` | **Post-retrofit heating is electricity only.** No `heating_hp_bkup`, no `heating_fans_pumps`, and no fossil column is read for any measure package. |
| `process_euss_data.py:536-579` | MP9/MP10 enclosure `in.`, `upgrade.` and `out.params.*_ft_2` columns | Inactive: `VALID_MENU_MPS = [0, 3, 4]` at `constants.py:75-81`. |
| `process_euss_data.py:614-638` | `out.electricity.peak_when_cooling.kw`, `peak_when_heating.kw`, both `.savings`, `out.load.cooling/heating.peak.kbtu_hr` and `.savings`, `out.electricity.total.energy_consumption.kwh` | Peak pass-through for the Tepper CSV. |
| `process_euss_data.py:524-525, 596` | `in.cooking_range`, `upgrade.cooking_range`, `out.electricity.range_oven...` from `df_cooking_range` | Dead, but the argument is still required by the signature at `process_euss_data.py:410`. |
| `user_input_geographic_filter.py:65-99` | `in.state`, `in.city` | |
| `adoption_kpis/data_loading.py:196-248` | `weight`, `out.natural_gas.heating...`, `out.load.heating.energy_delivered.kbtu`, `out.electricity.heating_hp_bkup...`, `out.electricity.heating_fans_pumps...`, `out.electricity.total...`, `in.ashrae_iecc_climate_zone_2004`, `in.county`, four `HEATING_FUEL_COLS`, plus `applicability` in `UPGRADE_USECOLS` at line 248 | The electric `hp_bkup` column is read here and nowhere else. |
| `adoption_kpis/data_loading.py:314-354` | `in.vacancy_status`, `in.geometry_building_type_recs`, `applicability == True` at line 354 | `load_euss_baseline` and `load_euss_upgrade` read the whole CSV; `USECOLS` is declared but not passed. |
| `adoption_kpis/bill_savings.py:50-51, 131, 176, 256-289` | `in.state`, `in.heating_fuel`, `weight` | |
| `adoption_kpis/demand.py:78-81, 90, 175-191` | `in.state`, `in.county`, `in.heating_fuel`, `weight` | |
| `adoption_kpis/thermal_cop.py:149-166` | `out.electricity.heating...`, `out.natural_gas.heating...`, `in.state`, `in.heating_fuel` | |
| `adoption_kpis/compute_adoption_rate.py:63-65` | Defaults `weight`, `in.county`, `in.state` | |
| `constants.py:427-436` | `ELEC_TOTAL_COL` (no `.kwh` suffix, the BSQ form), `COUNTY_COL`, `STATE_COL`, `WEIGHT_COL`; `METADATA_TABLE = "resstock_amy2018_release_1_1_metadata"` at line 433 | The grid-impact path is pinned to the 2022.1.1 Athena table. |
| `utils/tare_sample_size.py:48-56, 205-317` | `in.county`, `in.state`, `in.vacancy_status`, `in.geometry_building_type_recs`, `in.heating_fuel`, `in.hvac_heating_type_and_fuel`, `in.hvac_cooling_type`, `in.hvac_has_ducts`, `weight` | |
| `grid_impact/peak_load_functions.py:128-129`, `grid_impact/build_parcel_frame.py:91` | `in.county`; `bldg_id`, `weight` | |
| `utils/export_tepper_csv.py:232`, `utils/inventory_tare_columns.py:62`, `visuals_adoption_dotplot.py:324-325, 1224`, `determine_rebate_eligibility_and_amount.py:703, 772`, `scripts/verify_june2026_rebate_fossil_gate.py:32` | `weight`, on post-rename frames | `applicability` at `export_tepper_csv.py:262` is a Python list name, not the column. |
| `tare_model_main_v3_0_EXPORT_3Sep2026.py:615` | `df_baseline["weight"].median()` | |

`determine_economic_adoption_potential.py`, `adoption_kpis/visualize_geospatial_data.py`, and
`utils/validate_capital_costs.py` contain no `out.`, `in.`, or `upgrade.` literal. The
geospatial module keys on `state`, `county`, `GEOID`, `STATEFP`, and `STUSPS` instead.

### A.2 Literal 242 values

| Location | Form |
|---|---|
| `visuals_adoption_dotplot.py:192, 393` | `scaling_factor: float = 242.0` default parameter, plus the docstring at line 427. **The only hard-coded numeric weight in model code.** It is overridden at runtime whenever a `weight` column is present (lines 324-325). |
| `bill_savings.py:267`, `compute_adoption_rate.py:45, 176`, `demand.py:168, 198` | Comments only ("~242"). |
| `tare_model_main_v3_0_EXPORT_3Sep2026.py:615` | Comment; the value itself is measured from the column. |
| `tests/adoption_kpis/test_kpi_functions.py:104` | `assert DWELLING_UNIT_WEIGHT == 242`. **Stale**: `DWELLING_UNIT_WEIGHT` is the string `"weight"` (`data_loading.py:196`), so the assertion compares a string to an int. Fixtures at lines 280, 301, 536-597 also use `242`. |
| `cmu_tare_model/docs/BSQ_AWS_SETUP.md:222, 230`; `cmu_tare_model/docs/tare_tepper_exports_data_dictionary.md:42-756`; `CLAUDE.md:60, 68, 76, 90, 580` | Prose. |

`constants.py` contains no weight constant. The comment at `constants.py:415-418` records that
BSQ reads weights per row.

### A.3 Literal mp3 / mp4 / menu_mp comparisons

| Location | Literal | Effect |
|---|---|---|
| `constants.py:75-81` | `VALID_MENU_MPS = [0, 3, 4]` | Drives the loaders, both cost validators, and the EXPORT loop. |
| `constants.py:216` | `REBATE_ELIGIBLE_HEATING_MPS = [3, 4, 8, 9, 10]` | Rebate efficiency gate. |
| `constants.py:467-468` | `PA_COP_RANGES = {'mp3': ..., 'mp4': ...}` | KPI validation only. |
| `process_euss_data.py:475, 502` | `if menu_mp == 3:` | The ENERGY STAR string rewrite; see A.4. |
| `process_euss_data.py:531, 552` | `input_mp == 'upgrade09'` / `'upgrade10'` | Enclosure columns; reassigns `menu_mp` to 9 or 10. |
| `calculate_lifetime_private_impact.py:589, 614` | `input_mp in ['upgrade09', 'upgrade10']` | Weatherization cost. |
| `calculate_equipment_installation_costs.py:429-430` | `menu_mp == 7` | Inactive v3 path. |
| `determine_rebate_eligibility_and_amount.py:314-329` | `menu_mp in [9, 10]` | Deprecated wrapper. |
| `adoption_kpis/thermal_cop.py:33-35` | `"mp3"` / `"mp4"` keys | COP benchmark table. |
| `utils/validate_capital_costs.py:966-967` | `'mp3_heating_consumption'`, `'mp4_heating_consumption'` | Figure x-columns; frames named `df_*_mp3` / `_mp4` at lines 955-957. |
| `utils/inventory_tare_columns.py:30-31` | `DATAFRAMES_BY_MP[3]`, `[4]` | |
| `tare_model_main_v3_0_EXPORT_3Sep2026.py:272-278` | `HEATING_MP_SUBTITLES = {3: ..., 4: ..., 8: ..., 9: ..., 10: ...}` | The MP3 subtitle already reads 16 SEER1 / 9.5 HSPF1. |
| `adoption_kpis/data_loading.py:286` | `mp_to_upgrade` returns `f"upgrade{mp:02d}"` | Integer MP to file stem. Duplicated in `tare_scenarios_v3_0.ipynb` cell 4 line 10. |
| `utils/modeling_params.py:39-44` | `menu_mp == 0` gives `'baseline_'`, else `f"ref2025_mp{menu_mp}_"` | The prefix builder. |
| `tare_run_simulation_v3_0.ipynb` cells 9, 13, 16, 19, 22 | `if 3 in VALID_MENU_MPS: menu_mp = 3; input_mp = 'upgrade03'`, and the same for 4, 8, 9, 10 | One hard-coded block per 2022 package. |
| `tare_scenarios_v3_0.ipynb` cell 2 lines 10-23 | `menu_mp = 7; input_mp = 'upgrade07'`, loading `upgrade07_metadata_and_annual_results.csv` as `df_cooking_range` | A 2022-package dependency passed into `df_enduse_compare` at cell 7 line 23. |

`utils/column_names.py` contains many `'ref2025_mp3_'` and `'iraRef_mp3_'` strings, all inside
docstring examples.

### A.4 SEER/HSPF parsing and the ENERGY STAR override

| Location | Behaviour |
|---|---|
| `process_euss_data.py:464` | `upgrade_hvac_heating_efficiency` is copied raw from `upgrade.hvac_heating_efficiency`; no parsing happens here. |
| `process_euss_data.py:475-480` | The override: `if menu_mp == 3:` then `.str.replace('SEER 15', 'SEER 16', regex=False).str.replace('9.0 HSPF', '9.5 HSPF', regex=False)`. Plain substring replacement, not a regex. On the 2025.1 string `"...SEER 15.2, 7.8 HSPF2..."` the first replace would match `SEER 15` inside `SEER 15.2` and produce `SEER 16.2`; the second would not match at all. |
| `process_euss_data.py:502-507` | The same two replacements applied to `upgrade_hvac_cooling_efficiency`. A no-op today and a no-op on `Ducted Heat Pump`. |
| `remdb_v4_installed_cost_utils.py:316-320` | **The only numeric parse that feeds cost.** `str.extract(r'([\d.]+)', expand=False)` takes the first number in the string, then `/100` if the REMDB metric is AFUE (lines 329-333). On `"ASHP, SEER 15, 9.0 HSPF"` this yields 15. On the dual fuel string it would yield 15.2. |
| `calculate_equipment_installation_costs.py:346-361` | `obtain_heating_system_specs` holds regexes for `% AFUE`, `SEER`, and `HSPF`. A repository grep found **no callers** outside its own module. |

### A.5 `calculate_lifetime_fuel_costs`: post-retrofit fuel and the state key

| Location | Finding |
|---|---|
| `calculate_lifetime_fuel_costs.py:179-190` | Baseline (`menu_mp == 0`): fuel comes from `base_{category}_fuel` mapped through `FUEL_MAPPING`; `is_elec_or_gas` picks `state` or `census_division` as the price region. |
| `calculate_lifetime_fuel_costs.py:546-560` | Baseline lookup: electricity and natural gas keyed by `state`; fuel oil and propane keyed by `census_division`. |
| `calculate_lifetime_fuel_costs.py:566-578` | **Retrofit (`menu_mp != 0`): `fuel_type='electricity'` for every home, keyed by `state`.** Exactly one post-retrofit fuel is possible per home. The code comment reads "For measure packages, everything is mapped to electricity". |
| `calculate_lifetime_fuel_costs.py:582-600` | Consumption from `get_hdd_adjusted_consumption` multiplied by a single price series. |
| `degree_day_consumption_utils.py:305-308` | The retrofit consumption column is `f'mp{menu_mp}_{category}_consumption'`, which is electricity-only by construction (see A.1). Baseline sums all four fuels at line 386. |
| `calculate_lifetime_fuel_costs.py:102, 538-539` | Requires `state` and `census_division`, renamed from `in.state` and `in.census_division` at `process_euss_data.py:210, 214`. |

### A.6 Post-retrofit fossil use in the climate path

| Location | Finding |
|---|---|
| `calculate_fossil_fuel_emissions.py:56-62` | Emission series are initialised to zero and **`if menu_mp == 0:` is the only branch that computes fossil emissions.** For any measure package, fossil emissions are exactly zero for every pollutant. |
| `calculate_fossil_fuel_emissions.py:72-93` | Baseline fuels are `['naturalGas', 'propane', 'fuelOil']` read from `base_{fuel}_{category}_consumption`. |
| `calculate_lifetime_climate_impacts_sensitivity.py:157-176, 421-434` | Retrofit CO2e is electricity multiplied by the transmission-loss multiplier and the marginal emission rate, plus the zero fossil term. |
| `calculate_lifetime_private_impact.py:409-470` | `_calculate_discounted_savings` is fuel-agnostic: baseline annual fuel cost minus measure annual fuel cost. It makes no fuel assumption of its own; it inherits the electricity-only retrofit cost from A.5. |

### A.7 REMDB inputs and the capital-cost path

| Location | Finding |
|---|---|
| `remdb_v4_installed_cost_utils.py:636-660` | Heating `upgrade` reads `size_heating_system_primary_k_btu_h` (from the measure-package run, `process_euss_data.py:462`) with `upgrade_hvac_heating_efficiency`. Heating `replacement` reads `base_size_heating_system_primary_k_btu_h` (baseline run, `process_euss_data.py:265`) with `base_heating_efficiency`. Cooling replacement reads `base_size_cooling_system_primary_k_btu_h` with `base_cooling_efficiency`. |
| `remdb_v4_installed_cost_utils.py:146-171` | The upgrade row id depends only on `hvac_has_ducts`: `air_source_heat_pump_centrally_ducted` or `air_source_heat_pump_non_ducted_multi_zone`. There is no dual-fuel or furnace upgrade row. |
| `remdb_v4_installed_cost_utils.py:98-116` | The replacement row id maps Propane, Fuel Oil and Natural Gas all to `furnaces_gas_furnace`; electric non-ASHP to `electric_baseboard_default`; `Electricity ASHP` to the ASHP rows (dead, since those homes are excluded upstream). **Boilers are priced as furnaces**: REMDB's `boiler_gas_non_condensing`, `boiler_gas_condensing` and `boiler_oil` rows are never selected. |
| `data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv` rows 4, 7, 20 | The ASHP rows use pm1 = Cooling Capacity in Tons and **pm2 = SEER1** (bounds 13.8-24.0 ducted, 16.0-23.0 non-ducted). `furnaces_gas_furnace` uses pm1 = Heating Capacity in BTU/hr (bounds 30,000-156,250) and pm2 = AFUE (bounds 0.80-0.97). **A furnace cost function therefore exists**, on the replacement path, and a 0.925 or 0.95 AFUE value sits inside its pm2 bounds. |
| `remdb_v4_installed_cost_utils.py:316-320` | Confirms SEER1 only feeds the ASHP regression. HSPF is never parsed for cost. |
| `remdb_v4_installed_cost_utils.py:262-285` | pm1 unit conversion: tons means divide kBtu/h by 12; BTU/hr means multiply by 1000. |
| `calculate_lifetime_private_impact.py:293-296, 570-580` | Net capital cost is the upgrade cost minus the replacement credits, under the Case A / Case B split at `constants.py:363`. |
| `tare_scenarios_v3_0.ipynb` cell 18 lines 40-90 | Three `add_remdb_metrics` calls per measure package: heating replacement, heating upgrade, cooling replacement. There is no fourth call. |

### A.8 Constants and their consumers

| Constant | Value | Consumers |
|---|---|---|
| `EQUIPMENT_SPECS` | `constants.py:64-70`: `{'heating': 15, 'cooling': 15}` | `process_euss_data.py:8, 437`; `calculate_lifetime_fuel_costs.py:98, 120, 388`; `calculate_lifetime_private_impact.py:174-236`; both cost modules at line 45; `calculate_lifetime_climate_impacts_sensitivity.py:105-108`; `calculate_fossil_fuel_emissions.py:46`; `calculation_utils.py:44-243`; `degree_day_consumption_utils.py:301, 355`; `export_tepper_csv.py:182`; `precompute_hdd_factors.py:34`; `remdb_v4_installed_cost_utils.py:612`; `validation_framework.py:64, 105` |
| `ALLOWED_TECHNOLOGIES` | `constants.py:31-59`: heating is three electric-resistance plus six fossil furnace/boiler strings, with `'Electricity ASHP'` commented out at line 35; cooling is `['Central AC', 'Room AC']` | `calculation_utils.py:164-173` builds `valid_tech_{category}` with `.isin` against `{category}_type`; `tare_sample_size.py:233, 280, 314` |
| `REBATE_ELIGIBLE_HEATING_MPS` | `constants.py:216`: `[3, 4, 8, 9, 10]` | `calculate_lifetime_private_impact.py:320, 595, 629`; `determine_rebate_eligibility_and_amount.py:535` |
| `REBATE_RULE_CONFIG` | `constants.py:301-323`: `ira2024` has `heehr_fuel_gate` False, `homes_enabled` True, `homes_fuel_gate` False, `heehr_python_round` True; `june2026` has True, True, True, False | `determine_rebate_eligibility_and_amount.py:506-510, 827-832` |
| `REBATE_MAPPING` | `constants.py:199-201`: `('upgrade_hvac_heating_efficiency', ['ASHP', 'MSHP'], 8000.00)` | `determine_rebate_eligibility_and_amount.py:264-265, 375-378`. The cap is earned if **any substring** `'ASHP'` or `'MSHP'` appears in the upgrade string. |
| `ELECTRIC_RESISTANCE_BASELINE` | `constants.py:260`: `{"Electricity"}` | `determine_rebate_eligibility_and_amount.py:550-551` |
| `NON_PARTICIPATING_REBATE_STATES` | `constants.py:266`: `{"SD"}` | `determine_rebate_eligibility_and_amount.py:555`. No AK or HI entry. |
| `NPV_CASE_CATEGORIES` | `column_names.py:264-274`: nine tokens; `BASE_CASE_NPV_CASE = "heatingLCC_coolingLCC_unsub"` at line 283 | `column_names.py:315, 531`; `determine_economic_adoption_potential.py:11, 107`; EXPORT lines 50, 310; `peak_load_functions.py:66` |
| `VALID_MENU_MPS` | `constants.py:75-81`: `[0, 3, 4]` | Both cost modules at lines 176 and 234; `load_exported_results_to_df.py:95`; EXPORT lines 39, 195; `tare_run_simulation_v3_0.ipynb` cells 9-23 |
| `FUEL_MAPPING` | `constants.py:127`: four fuels; no Wood, Other Fuel or None | `calculate_lifetime_fuel_costs.py:186`; `calculation_utils.py:50-61`; `degree_day_consumption_utils.py:26`; `tare_sample_size.py:223, 310` |
| `UPGRADE_COLUMNS` | `constants.py:191-196`: `{'heating': 'upgrade_hvac_heating_efficiency'}` | `validation_framework.py:21` |

### A.9 Alaska and Hawaii in the state-keyed inputs

Recorded as measured. Decision 1 excludes both states, so the one gap below (the Cambium GEA
crosswalk) no longer bites; the table is kept because it is the evidence the decision rests
on.

| Input | AK | HI | Note |
|---|---|---|---|
| `data/fuel_prices/eia_fuel_price_data_2025_usd2025.csv` | Present: electricity 0.2609, natural gas 0.0426 USD/kWh; fuel oil and propane via the national fallback row | Present: electricity 0.4059, natural gas 0.1720; fuel oil and propane via the national fallback | Four rows each. |
| `data/projections/aeo2026_fuel_price_factors_2025_2050.csv` | Covered through the `Pacific` division row | Same | Keyed by census division, not state. |
| `data/projections/aeo2026_degree_day_factors_2025_2050.csv` | Covered through `Pacific`, with a `National` fallback at `degree_day_consumption_utils.py:139-146` | Same | |
| `data/ami_calculations_data/.../ACSDT5Y2024.B19013-Data.csv` | Present: state row `0400000US02` plus 30 borough and census-area rows | Present: state row `0400000US15` plus 5 county rows | |
| `adoption_kpis/data_loading.py:255-258` `STATE_NAMES` | `"AK"` present | `"HI"` present | |
| **`data/projections/county_to_gea_mapping_cambium23.csv`** | **Zero rows** | **Zero rows** | Not on the task list but state-keyed. `process_euss_data.py:238-250` maps `county_fips` to `gea_region`; unmapped homes get NaN and are excluded from climate damages. Its CT rows are the eight pre-2023 counties. |
| `adoption_kpis/visualize_geospatial_data.py:254, 588-591` | Drawn as a separate inset (`gdf_alaska`), excluded from the CONUS base layer | Excluded outright (`'15'`) | |

### A.10 Notebook load and demand cells

| File and cell | Lines | Detail |
|---|---|---|
| `tare_baseline_v3_0.ipynb` cell 2 | 20-32 | `filename = "baseline_metadata_and_annual_results.csv"`; path `data/euss_data/resstock_amy2018_release_1.1/national/csv`; **`columns_to_string = {11: str, 61: str, 121: str, 103: str, 113: str, 128: str, 129: str}`**, positional dtype indices tied to the 2022.1.1 column order; `pd.read_csv(..., index_col="bldg_id")`. Column strings at lines 35, 39, 62, 66, 85. |
| `tare_baseline_v3_0.ipynb` cells 4, 7, 9 | 21, 13, 28 | `df_enduse_refactored`; `calculate_lifetime_climate_impacts(menu_mp=0)`; `calculate_lifetime_fuel_costs`. |
| `tare_scenarios_v3_0.ipynb` cell 2 | 10-23 | Hard-coded `upgrade07` load into `df_euss_am_mp7`, used only as the cooking-range frame. |
| `tare_scenarios_v3_0.ipynb` cells 4-5 | 10, 15, 26 | `mp_to_upgrade`; `f"{input_mp}_metadata_and_annual_results.csv"`; `read_csv(low_memory=False, index_col="bldg_id")`. **No `applicability` filter appears anywhere in the TARE run path**; only the KPI loader filters it, at `data_loading.py:354`, and without a bool coercion. |
| `tare_scenarios_v3_0.ipynb` cells 7, 10, 13, 18, 26, 27 | | `df_enduse_compare(df_mp, input_mp, menu_mp, df_baseline, df_cooking_range=df_euss_am_mp7)`; climate; fuel costs; three `add_remdb_metrics` calls; `calculate_private_npv`; `economic_adoption_decision`. |
| `tare_run_simulation_v3_0.ipynb` cells 9-23 | | One `if N in VALID_MENU_MPS:` block per package with a literal `input_mp = 'upgradeNN'`. |
| `tare_model_main_v3_0_EXPORT_3Sep2026.py:180-206` | | Loads TARE output CSVs, not EUSS. |
| `tare_model_main_v3_0_EXPORT_3Sep2026.py:231-296` | 285 | `df_baseline = load_euss_baseline()`. |
| `tare_model_main_v3_0_EXPORT_3Sep2026.py:462-520` | 470-473, 513, 516, 485 | `load_euss_upgrade(mp_to_upgrade(mp))`; `compute_scenario_demand`; `aggregate_demand`; operating-cost percent from `{prefix}heating_avg_annual_fuel_cost_pct_change`. The column literals live in `demand.py:78-81`. |
| `tare_model_main_v3_0_EXPORT_3Sep2026.py:588-634` | 615 | The weight sanity check. |

---

## B. Measurements of the 2025.1 data

All three national AMY2018 parquet files hold 549,971 rows and an identical 771-column schema.
One distinct `weight` value, **253.90367272727272**. Baseline weighted total: 139,639,656.8
dwellings.

### B.1 Rows, weight, applicability

| File | Rows (rdu) | Distinct weight | `applicability` dtype | True (rdu) | Weighted |
|---|---:|---|---|---:|---:|
| upgrade0 (baseline) | 549,971 | 1 | `bool` | 549,971 | 139,639,657 |
| upgrade5 (dual fuel) | 549,971 | 1 | `bool` | 245,570 | 62,351,125 |
| upgrade4 (cold-climate ASHP) | 549,971 | 1 | `bool` | 413,604 | 105,015,575 |

`applicability` is a true boolean in all three files, so no string coercion is needed for these
files. Non-applicable rows carry `None` in every `upgrade.*` column. 245,568 of the 245,570
upgrade-05 applicable homes are also applicable under upgrade 04.

### B.2 Upgrade option strings, applicable rows only

| File | `upgrade.hvac_heating_efficiency` | rdu | `upgrade.hvac_cooling_efficiency` |
|---|---|---:|---|
| 05 | `Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5% AFUE NG, 35F switchover` | 147,523 | `Ducted Heat Pump`, all 245,570 |
| 05 | `Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 95.0% AFUE NG, 35F switchover` | 98,047 | |
| 04 | `ASHP, SEER2 17.5, 8.5 HSPF2, Typical Cold Climate` | 413,604 | `Ducted Heat Pump`, all |

Exactly the two expected strings for 05, and nothing else. The AFUE tier splits cleanly on
IECC zone with no crossover: 92.5 percent for every home in zones 1A, 2A, 2B, 3A, 3B, 3C, 4A,
4B and 4C; 95.0 percent for every home in 5A, 5B, 6A, 6B, 7A, 7AK, 7B and 8AK.

`upgrade.hvac_cooling_partial_space_conditioning` reads `100% Conditioned` for all 245,570
applicable rows under 05 and all 413,604 under 04.

`upgrade.hvac_detailed_performance_data` is `None` for every 05 row and
`Cold Climate Heat Pump Ducted` for every 04 row.

### B.3 The broken upgrade fuel columns

`upgrade.heating_fuel` and `upgrade.hvac_heating_type_and_fuel` are `None` on all 549,971 rows
under both 05 and 04. Confirmed, and not used anywhere else in this audit. Upgrade 01 was not
downloaded, so the guide's claim that 01 is the only correct one was not tested.

### B.4 States

51 distinct `in.state` values: the 50 states plus DC. Both AK and HI are present in the data
and are excluded from the study by Decision 1.

| State | Baseline rdu | Baseline weighted | 05 applicable rdu (weighted) | 05 applicable, Occupied and single-family rdu (weighted) |
|---|---:|---:|---:|---:|
| AK | 1,255 | 318,649 | 248 (62,968) | 164 (41,640) |
| HI | 2,163 | 549,194 | 130 (33,007) | 92 (23,359) |
| CT | 6,132 | 1,556,937 | 1,544 (392,027) | 1,102 (279,802) |

AK and HI both carry `in.census_division = 'Pacific'`. Dropping the two states removes 3,418
baseline rdu (867,843 dwellings), 378 rdu from the 05 applicable set (95,976 dwellings) and
604 rdu from the 04 applicable set (153,358 dwellings), and leaves 49 state keys plus DC --
the same set as 2022.1.1.

### B.5 Connecticut, and an unexpected geography finding

`in.county` for Connecticut has **eight** distinct values, all GISJOIN:
`G0900010, G0900030, G0900050, G0900070, G0900090, G0900110, G0900130, G0900150`
(Fairfield, Hartford, Litchfield, Middlesex, New Haven, New London, Tolland, Windham). These
are the same pre-2023 counties as 2022.1.1, so the CT question is settled and nothing about
the CT handling needs to change on geography grounds.

**AK and HI were the geography problem instead, and Decision 1 removes it.** `in.county` is
GISJOIN for 3,107 counties across 49 states, but for AK and HI it is a name string such as
`'AK, Anchorage Municipality'` or `'HI, Honolulu County'` -- 32 distinct values across 3,418
rdu. `in.county_and_puma` stays GISJOIN on those rows (`'G1500030, G15000308'`). Applying
TARE's FIPS slice at `process_euss_data.py:218` to `'HI, Honolulu County'` returns `'I,Hon'`.
Total distinct `in.county` in the 2025.1 baseline is 3,139, against 3,107 in the raw 2022.1.1
baseline.

With AK and HI excluded, **every remaining `in.county` value is GISJOIN and the baseline holds
exactly 3,107 distinct counties, matching the raw 2022.1.1 count**. The 05 applicable set
spans 3,064 counties and the 04 set 3,105. No change to the FIPS parse, the GEA crosswalk, the
AMI join or the map layers is needed on geography grounds.

### B.6 Upgrade 05 applicable population

245,570 rdu, 62,351,125 dwellings, which is 44.65 percent of published rows.

Weighted share by baseline heating fuel: Natural Gas 85.50, Electricity 13.27, Fuel Oil 0.93,
Propane 0.30. No Wood, Other Fuel or None baseline is applicable.

Weighted share by baseline cooling type: Central AC 75.25, Room AC 9.92, None 8.11, Ducted
Heat Pump 6.73. By baseline partial conditioning: 100% Conditioned 73.72, None 8.11, 60%
6.04, 80% 5.27, 20% 3.86, 40% 2.64, under 10% 0.36.

Existing heat pumps: 16,531 rdu (4,197,282 dwellings, 6.73 percent weighted) carry
`in.hvac_heating_type_and_fuel = 'Electricity ASHP'`, split across SEER 13 / 7.7 HSPF (7,912),
SEER 15 / 8.5 HSPF (7,819) and SEER 10 / 6.2 HSPF (800). No MSHP. `in.hvac_heating_type` reads
`Ducted Heat Pump` for exactly those rows.

Baseline technologies present among 05 applicable rows but absent from
`ALLOWED_TECHNOLOGIES['heating']` (`constants.py:31-44`): `Natural Gas Fuel Wall/Floor Furnace`
4,620 rdu, `Electricity Electric Wall Furnace` 477, `Propane Fuel Wall/Floor Furnace` 60,
`Fuel Oil Fuel Wall/Floor Furnace` 46, plus the 16,531 `Electricity ASHP`.

All 245,570 applicable rows read `in.hvac_has_ducts = 'Yes'` and
`in.hvac_shared_efficiencies = 'None'`.

### B.7 Where the dual fuel energy lands

| Column | Non-zero rdu | Share | Mean kWh |
|---|---:|---:|---:|
| `out.electricity.heating_hp_bkup...` | 0 | 0.00% | 0.0 |
| `out.natural_gas.heating_hp_bkup...` | 220,502 | 89.79% | 10,015.9 |
| `out.propane.heating_hp_bkup...` | 0 | 0 | 0 |
| `out.fuel_oil.heating_hp_bkup...` | 0 | 0 | 0 |
| `out.electricity.heating...` | 245,494 | 99.97% | 1,907.4 |
| `out.natural_gas.heating...` | 7 | 0.00% | 0.3 |

Electric backup is exactly zero everywhere. The furnace gas lands only in the natural-gas
backup column, including for the 2,282 fuel-oil and 707 propane baselines, which switch their
backup fuel to natural gas. The 25,068 rows with zero gas backup sit almost entirely in warm
zones: 1A 100.0 percent zero, 2B 72.7, 3C 53.8, 3B 46.7, 2A 4.9, and every zone from 4A
colder below 0.6 percent. The seven rows with non-zero `natural_gas.heating` are all Alaska
natural-gas furnace homes and look like secondary heating.

Weighted energy under 05: electricity heating 118.93 TWh, heating fans and pumps 10.51 TWh,
backup fans 17.40 TWh, natural gas backup 624.50 TWh. Gas is **81.0 percent** of heating site
energy. Per home the gas share has a median of 75.8 percent, with quartiles at 53.7 and 84.3.
By zone the weighted gas share runs 0.0 (1A), 52.7 (2A), 69.4 (3A), 76.5 (4A), 85.1 (5A),
91.4 (6A), 94.5 (7A). For the same homes, baseline natural-gas heating is 997.41 TWh, so the
package cuts gas use by roughly 37 percent rather than eliminating it.

### B.8 Sizes

Among 05 applicable rows, `size_heating_system_primary` equals `size_cooling_system_primary`
for 100.0 percent of rows (mean 36.4 kBtu/h, median 31.6, range 3.2 to 611.2).
`size_heat_pump_backup_primary` is positive for 100.0 percent of rows: mean 55.0 kBtu/h,
median 49.0, quartiles 30.7 and 72.5, range 0.2 to 413.7. The backup furnace is larger than
the heat pump.

Against REMDB v4 bounds (`data/retrofit_costs/remdb_v4_tare_retrofit_costs.csv`): the backup
size falls inside the `furnaces_gas_furnace` pm1 range of 30 to 156.25 kBtu/h for 74.48
percent of rows, below it for 24.09 percent, above it for 1.42 percent. The heat pump size
falls inside the ducted-ASHP pm1 range of 18 to 60 kBtu/h for 65.89 percent of rows, below for
21.02 percent, above for 13.09 percent. CLAUDE.md Documented Limitation 9 already records
that TARE prices homes outside the REMDB bounds.

### B.9 Electrical panel constraints

Under 05, `out.params.panel_constraint_overall...` reads No Constraint for 208,694 rdu,
Capacity Constrained Only 20,943, Space Constrained Only 14,008, and Capacity and Space
Constrained 1,925. That is **15.02 percent** constrained in some way (9.31 percent capacity,
6.49 percent breaker space). Under 04 the same field reads No Constraint for only 168,750 of
413,604, so **59.2 percent** are constrained (50.84 percent capacity, 34.50 percent space).

### B.10 TARE-style scope, for comparison with the 2022.1.1 frame

Applying the Occupied plus single-family filter (`ALLOWED_HOUSING_TYPES`, `constants.py:26`):

| Frame | rdu | Weighted dwellings |
|---|---:|---:|
| 2025.1 baseline, Occupied and single-family | 331,541 | 84,179,478 |
| 2025.1 upgrade 05 applicable, Occupied and single-family | 190,630 | 48,401,657 |
| 2025.1 upgrade 04 applicable, Occupied and single-family | 288,283 | -- |
| 2022.1.1 TARE baseline frame (CLAUDE.md) | 331,531 | 80,273,937 |

**The same frames with AK and HI excluded per Decision 1**, which are the denominators the
later phases will actually use:

| Frame, AK and HI excluded | rdu | Weighted dwellings |
|---|---:|---:|
| 2025.1 baseline, Occupied and single-family | 329,633 | 83,695,029 |
| 2025.1 upgrade 05 applicable, Occupied and single-family | 190,374 | 48,336,658 |
| 2025.1 upgrade 04 applicable, Occupied and single-family | 287,877 | 73,093,028 |

Within the 05 Occupied single-family set with AK and HI excluded: Natural Gas 168,848 rdu,
Electricity 19,406 (of which 9,681 are existing ASHP), Fuel Oil 1,616, Propane 504; Central AC
149,564, Room AC 16,112, None 15,017, Ducted Heat Pump 9,681. Before the state exclusion the
same figures were Natural Gas 169,020, Electricity 19,490 (9,725 existing ASHP), Fuel Oil
1,616, Propane 504; Central AC 149,578, Room AC 16,139, None 15,188, Ducted Heat Pump 9,725.

### B.11 What the crosswalk files say about the 2022 packages

Nothing. `measure_name_crosswalk_res_2025_1.xlsx` has four columns -- `measure_id`,
`measure_documentation_name`, `upgrades_lookup.json name`, and
`2025_resstock_amy2018_release_1_upgrade_id` -- across 29 rows, and zero cells mentioning 2022
or release 1.1. `Column Name Crosswalk.csv` maps 2025.1 names to **ResStock 2024 Release 2**,
not to 2022.1.1. The accompanying docx is a one-paragraph summary of 2024.2 to 2025.1 column
changes. There is no published mapping from the 2022.1.1 upgrade03 and upgrade04 packages to
any 2025.1 upgrade.

---

## C. Facts bearing on decisions D1 to D10

Stated neutrally. Each item also records whether the guide's recommended default is consistent
with what the code does today. No recommendation is made beyond that.

### D1 -- Measure-package numbering

**Facts.** `define_scenario_params` builds the prefix as `f"ref2025_mp{menu_mp}_"`
(`modeling_params.py:44`) from an integer, and `mp_to_upgrade` builds the file stem as
`f"upgrade{mp:02d}"` (`data_loading.py:286`). Both work unchanged for `mp = 5`. However
`VALID_MENU_MPS = [0, 3, 4]` (`constants.py:75-81`) is validated against in
`calculate_equipment_installation_costs.py:176`, `calculate_equipment_replacement_costs.py:234`
and `load_exported_results_to_df.py:95`, and `menu_mp == 3` at `process_euss_data.py:475` and
`502` is keyed to the MP3 identity, not to a release. There is no `RESSTOCK_RELEASE` constant
anywhere in `constants.py` today.

**Consistency with the guide's default (a).** Consistent with the column builders, which do
reuse an integer `mp`. The hard rule the guide attaches -- one release per run, declared once
-- has no existing mechanism: nothing in the codebase records which release a frame came from,
and `METADATA_TABLE` at `constants.py:433` hard-codes the 2022.1.1 Athena table name.

### D2 -- Which packages to run

**Facts.** Upgrade 05 has 245,570 applicable rdu and upgrade 04 has 413,604, overlapping on
245,568 (B.1). Both were downloaded and measured. Upgrade 03 was not downloaded. The two
files total 1.49 GB.

**Consistency with the guide's default (05 plus 04).** Consistent. Nothing in the code
constrains how many packages can be loaded; `VALID_MENU_MPS` is a list.

### D3 -- Which homes are in scope

**Facts.** ResStock's own applicability admits 16,531 rdu with `in.hvac_heating_type_and_fuel`
= `'Electricity ASHP'` under 05, 6.73 percent of the applicable population by weight (B.6).
TARE excludes those today because `'Electricity ASHP'` is commented out of
`ALLOWED_TECHNOLOGIES['heating']` at `constants.py:35`, and the exclusion is enforced at
`calculation_utils.py:164-173`. CLAUDE.md records this as a settled decision and Documented
Limitation 8. Separately, 5,203 rdu carry wall or floor furnace technologies that
`ALLOWED_TECHNOLOGIES` also excludes (B.6), which is the open question already recorded in the
`project_wall_floor_furnace_exclusion` memory note. Also relevant: **TARE's run path never
filters on `applicability` at all** (A.10), in either release.

**Consistency with the guide's default (b).** Consistent with the existing-ASHP half: option
(b) is what `ALLOWED_TECHNOLOGIES` already does. Not consistent with the `applicability` half:
adding that filter would be new behaviour, not a continuation. In 2022.1.1 the column exists
and is populated -- upgrade03 has 656 rdu at False -- and TARE has always ignored them.

### D4 -- Cooling for homes with no baseline air conditioner

**Facts.** The guide's premise is confirmed: `upgrade.hvac_cooling_partial_space_conditioning`
is `100% Conditioned` for all 245,570 applicable rows (B.2). Under 05, 8.11 percent of
applicable homes by weight have no baseline cooling and a further 18.17 percent have partial
conditioning (B.6). TARE's current rule sets both cooling savings and cooling capital to zero
when `include_cooling` is False; `include_cooling` comes only from
`ALLOWED_TECHNOLOGIES['cooling'] = ['Central AC', 'Room AC']` (`constants.py:41-44`), so both
the `None` homes and the 6.73 percent Ducted Heat Pump homes fall outside it. The cooling
replacement credit is zeroed at `calculate_lifetime_private_impact.py:288-289`. CLAUDE.md's
"negative cooling savings accepted as real" decision already accepts an uncompensated cooling
cost for Room AC homes.

**Consistency with the guide's default (b) for dual fuel, (a) retained for 2022.1.1.** Not
consistent with what the code does today: option (b) reverses the existing zeroing rule for a
subset of homes, so it is a behaviour change. Keeping (a) for the 2022.1.1 path is consistent,
and is also what the byte-identical regression requirement demands.

### D5 -- Post-retrofit fuel mix and prices

**Facts.** The post-retrofit fuel is hard-wired to electricity at three independent points:
the consumption column read (`process_euss_data.py:533, 552, 584`), the price lookup
(`calculate_lifetime_fuel_costs.py:566-578`, `fuel_type='electricity'` for every home), and the
fossil-emissions branch (`calculate_fossil_fuel_emissions.py:62`, which computes fossil
emissions only when `menu_mp == 0`). Measured, the dual fuel split is electricity heating
118.93 TWh against natural gas backup 624.50 TWh, so gas is 81.0 percent of heating site
energy and the median home is at 75.8 percent gas (B.7). Propane and fuel-oil baselines do
switch to gas backup: their propane and fuel-oil backup columns are zero on every row. The gas
share varies from 0.0 percent in zone 1A to 94.5 percent in 7A, which is the temperature
dependence the guide flags.

**Consistency with the guide's default (model both fuels, scale each by HDD).** Not consistent
with what the code does today at any of the three points above. The degree-day helper takes
one consumption column per category (`degree_day_consumption_utils.py:305-308`), and
`calculate_annual_fuel_costs` computes one price series per category
(`calculate_lifetime_fuel_costs.py:582-600`), so a two-fuel retrofit has no representation in
the current column scheme.

### D6 -- Capital cost of the new furnace

**Facts.** A furnace cost function exists. `_assign_replacement_row_id` maps Natural Gas,
Propane and Fuel Oil baselines to the REMDB row `furnaces_gas_furnace`
(`remdb_v4_installed_cost_utils.py:98-116`), whose pm1 is Heating Capacity in BTU/hr with
bounds 30,000 to 156,250 and whose pm2 is AFUE with bounds 0.80 to 0.97. The dual fuel
furnace's 0.925 and 0.95 AFUE both fall inside the pm2 bounds. The size column the guide names,
`out.params.size_heat_pump_backup_primary`, exists in both releases and is already present but
commented out at `process_euss_data.py:450`. Measured, it is positive on 100 percent of
applicable rows with a mean of 55.0 kBtu/h, and 74.48 percent of rows fall inside the REMDB
furnace pm1 bounds, 24.09 percent below and 1.42 percent above (B.8). The current
`_assign_upgrade_row_id` has only the two ASHP rows and no furnace branch
(`remdb_v4_installed_cost_utils.py:146-171`), and `tare_scenarios_v3_0.ipynb` cell 18 makes
exactly three `add_remdb_metrics` calls.

**Consistency with the guide's default (b), ASHP plus furnace.** The reusable function the
guide hopes for does exist, on the replacement path. Adding a furnace to the upgrade side is
not consistent with what the code does today: it needs a fourth cost call and a new upgrade
row-id branch, neither of which exists.

### D7 -- Electrical panel upgrade cost

**Facts.** The panel columns exist and are populated. Under 05, 15.02 percent of applicable
homes are constrained in some way: 9.31 percent on capacity, 6.49 percent on breaker space
(B.9). Under 04 the figure is 59.2 percent. The constraint fields are
`out.params.panel_constraint_overall.2023_nec_existing_dwelling_load_based` (a four-value
string), `..._capacity...` and `...breaker_space` (both boolean). No panel column is read
anywhere in the TARE codebase, and no panel cost component exists in
`calculate_lifetime_private_impact.py`.

**Consistency with the guide's default (a), ignore for the first run and carry the flag as a
reporting column.** Consistent: ignoring the panel is what the code does today. Carrying the
flag would be a new pass-through column, which is additive rather than a behaviour change.
Note that the guide cites the measure documentation's 8.0 percent constrained figure; the
measured figure on the published data is 15.02 percent (see D-item corrections in Section D).

### D8 -- Rebate eligibility for a dual fuel system

**Facts.** The HEEHR cap is gated by a substring test: `REBATE_MAPPING['heating']` holds
`['ASHP', 'MSHP']` (`constants.py:201`) and `_heehr_rebate_amount` checks
`any(cond in upgrade for cond in tech_conditions)`
(`determine_rebate_eligibility_and_amount.py:375-378`). The dual fuel option string contains
the literal substring `ASHP`, so **it would pass that test as written today** and earn the
8,000 dollar cap. The efficiency gate is separate: `REBATE_ELIGIBLE_HEATING_MPS = [3, 4, 8, 9,
10]` (`constants.py:216`) is checked at `determine_rebate_eligibility_and_amount.py:535`, and
5 is not in that list, so an `mp = 5` run would currently get zero from the earlier gate. The
June 2026 HEEHR fuel gate reads `base_heating_fuel` against
`ELECTRIC_RESISTANCE_BASELINE = {"Electricity"}` (`constants.py:260`,
`determine_rebate_eligibility_and_amount.py:550-551`), so under the current rules 86.73 percent
of dual fuel recipients by weight -- every non-electric baseline (B.6) -- would get zero HEEHR
under the June 2026 vintage regardless of the hybrid question. HOMES is fuel-neutral for the
2024 vintage (`REBATE_RULE_CONFIG`, `constants.py:301-323`).

**Consistency with the guide's default (run (a) and (b) as bounds, encode via
`REBATE_RULE_CONFIG`).** Partly consistent. `REBATE_RULE_CONFIG` is the right lever and is
already the single dispatch point. But the guide's statement that "the June 2026 HEEHR fossil
gate is moot because no fossil system is removed" does not match the code: the gate is
implemented as a test on the *baseline* fuel, not on whether a fossil system is removed, so it
fires on fossil-baseline dual fuel homes as written.

### D9 -- SEER2 and HSPF2 conversion for REMDB

**Facts.** Only SEER1 feeds cost. The ASHP REMDB rows use pm2 = SEER1 with bounds 13.8 to 24.0
(ducted), and `_convert_pm2` takes the first number in the efficiency string
(`remdb_v4_installed_cost_utils.py:316-320`). HSPF is never parsed for cost. The dual fuel
option string reads `SEER 15.2`, using the SEER1 label, while the measure documentation
describes the unit as SEER2 15.2; the string itself therefore does not say which convention
its number follows. Either 15.2 or a converted 16.0 falls inside the REMDB pm2 bounds, so
neither would be clamped by `_apply_efficiency_floor`. Separately, the 2025.1 **baseline**
efficiency enumerations are still SEER1 and AFUE style -- `ASHP, SEER 13, 7.7 HSPF`,
`Fuel Furnace, 80% AFUE`, `Room AC, EER 12.0` -- so the replacement-cost pm2 parse needs no
conversion.

**Consistency with the guide's default (Appendix M1 factors).** No conversion of any kind
exists in the code today; `_convert_pm2` reads the number as-is. The guide's own note to
confirm the factors against the M1 document before coding still stands. The guide's framing
that 2025.1 moved to SEER2 applies to the upgrade option strings only, not to the baseline
characteristics.

### D10 -- Geography

**Facts.** Every state-keyed TARE input has AK and HI rows except one: the fuel-price CSV, the
two AEO2026 projection tables (via the `Pacific` division), the ACS AMI file and `STATE_NAMES`
all cover them, but `data/projections/county_to_gea_mapping_cambium23.csv` has **zero** AK and
HI rows (A.9). Homes with no GEA match get NaN and drop out of climate damages
(`process_euss_data.py:238-250`). On top of that, `in.county` is a name string rather than
GISJOIN for exactly those two states (B.5), so the FIPS slice at `process_euss_data.py:218`
fails there, which breaks the county AMI join and the GEA lookup before the missing rows even
matter. The map code already excludes HI outright and draws AK as an inset
(`visualize_geospatial_data.py:254, 588-591`). The volume at stake is small: 164 rdu in AK and
92 in HI within the 05 Occupied single-family set, 41,640 and 23,359 dwellings respectively
(B.4). Connecticut is resolved: eight pre-2023 GISJOIN counties, unchanged from 2022.1.1
(B.5), so the `cb_2021` shapefile pin and the state-level AMI fallback both remain valid.

**Consistency with the guide's default (include AK and HI if every input table has rows,
otherwise exclude explicitly and log the weighted count).** The condition the guide sets was
not met, because of the GEA crosswalk.

**DECIDED, 19 September 2026: exclude Alaska and Hawaii.** The reasoning is about the measures
under study rather than the data plumbing -- Hawaii has limited gas infrastructure and
essentially no heating demand, and Alaska is fuel-oil dominated with a very high
electricity-to-gas price ratio and very high heating demand, which makes both states poor
candidates for either a dual fuel or a cold-climate heat pump relative to the rest of the
country. The exclusion is also the minimal-change option: it restores the 49-state plus DC
footprint the code already has, leaves every `in.county` value in GISJOIN form, and brings the
baseline county count to exactly 3,107, matching 2022.1.1 (B.5). Connecticut needs no change
either. The weighted count dropped is 867,843 dwellings at baseline and 95,976 within the 05
applicable set (B.4), which the guide's own default asks to be logged.

---

## D. Corrections to the guide

Each item quotes the guide and gives the measured correction.

**1. Section 2.5, the gas backup row.** The guide marks
`out.natural_gas.heating_hp_bkup.energy_consumption..kwh` as
"**Gas furnace backup (NEW)** | n/a" in the 2022.1.1 column. It is not new. The column
`out.natural_gas.heating_hp_bkup.energy_consumption.kwh` exists in ResStock 2022.1.1, as do the
propane and fuel-oil equivalents. Measured in the 2022.1.1 files, all three are identically
zero in the baseline, upgrade03 and upgrade04 -- no non-zero value in 548,916 rows -- because
no 2022 package had a fossil backup. The correct status is a rename plus a first population,
not an addition.

**2. Section 2.5, the backup fan column.** The guide names
`out.electricity.heating_hp_bkup_fans_pumps.energy_consumption..kwh`. No column by that name
exists in either release. The 2025.1 name is
`out.electricity.heating_hp_bkup_fa.energy_consumption..kwh`, abbreviated to `fa`, and it is
genuinely new: 2022.1.1 has no heat-pump backup fan column at all. It carries 17.40 TWh under
upgrade 05.

**3. Section 2.5, the row-identity row.** The guide says "`in.sample_weight` also exists in the
raw inputs". There is no `in.sample_weight` column in the published metadata file of either
release. The guide's advice to use the top-level `weight` column is correct; only the
parenthetical is wrong.

**4. Section 2.5, the applicability row.** The guide's 2022.1.1 cell reads "(all rows
applied)". 2022.1.1 has an `applicability` column in both the baseline and the upgrade files,
typed bool, and upgrade03 carries 656 rdu at False. The behaviour the guide describes belongs
to TARE, not to the data: TARE's run path never filters on the column in either release
(A.10).

**5. Section 2.5, the design-capacity row.** The guide offers
`out.hvac_capacity.heating..btu_h` as an "Alternative to `out.params.size_*` if those are
absent in the published file". Two corrections: the actual names are
`out.capacity.heating..btu_per_hr`, `out.capacity.cooling..btu_per_hr` and
`out.capacity.heat_pump_backup..btu_per_hr`, without the `hvac` segment; and all four
`out.params.size_*` columns are present in the published file, so no substitute is needed.

**6. Section 2.5, the panel row.** The guide names `in.electric_panel_service_rating`. The
published name is `in.electric_panel_service_rating..a`, with the amp unit suffix.

**7. Section 2.3, the crosswalk.** The guide says "`measure_name_crosswalk.csv` in the dataset
root is the authoritative cross-release mapping; download it in Phase 1 and record what it
says about the 2022 packages." The file is `measure_name_crosswalk_res_2025_1.xlsx`, an xlsx
not a csv, and it says **nothing** about the 2022 packages: its four columns are `measure_id`,
`measure_documentation_name`, `upgrades_lookup.json name` and the 2025.1 upgrade id, with zero
cells mentioning 2022 (B.11). The separate `Column Name Crosswalk.csv` maps to ResStock 2024
Release 2, not 2022.1.1. There is no published 2022-to-2025 package mapping.

**8. Section 2.1, the upgrade count.** The guide says "28 packages, **all new**". The release
now has 32 upgrades plus the baseline: `upgrades_lookup.json` holds keys `"0"` through `"32"`,
and the README's own "Updates Since November 2025" section records that upgrade IDs 29, 30, 31
and 32 were added after the original publication.

**9. Section 2.1, the efficiency metric.** The guide's row reads "SEER / HSPF (SEER1) ->
**SEER2 / HSPF2**" as a dataset-level change. It is not dataset-level. The 2025.1 **baseline**
enumerations are still SEER1, EER and AFUE style: `in.hvac_heating_efficiency` values include
`ASHP, SEER 13, 7.7 HSPF` and `Fuel Furnace, 80% AFUE`, and `in.hvac_cooling_efficiency`
includes `AC, SEER 13` and `Room AC, EER 12.0`. Only the upgrade option strings use SEER2 and
HSPF2, and the dual fuel string mixes the two conventions in one value (`SEER 15.2` with
`7.8 HSPF2`). So the replacement-cost pm2 parse needs no conversion; only the upgrade side
does.

**10. Section 2.1, the weight.** The guide gives `253.9`. The exact value in the file is
253.90367272727272, on every one of 549,971 rows in all three parquet files.

**11. Section 2.1, the sample count.** The guide says "~550,000 simulated; 'roughly 500,000'
published rows (README)". The published national file has exactly 549,971 rows, for the
baseline and for each upgrade.

**12. Section 2.1, the county count.** The guide says "3,098 in TARE's frame -> 3,100+ (data
page)". Measured, the 2025.1 baseline has 3,139 distinct `in.county` values against 3,107 in
the raw 2022.1.1 baseline. More important than the count is the format change described in
B.5, which the guide does not anticipate: `in.county` is a name string, not GISJOIN, for all
AK and HI rows. Under Decision 1 that format change is moot, and the post-exclusion county
count is exactly 3,107, the same as 2022.1.1.

**13. Section 2.1, the applicability dtype.** The guide says to "Coerce to bool explicitly on
read" because the column "was string in upgrade parquets". In these three files it is already
`bool`, so the coercion is a no-op here. The advice is harmless, but the stated reason does
not apply to this download.

**14. Section 2.2, the panel constraint share.** The guide's Section 2.1 panel row says "Dual
fuel doc: 8.0% of applicable homes constrained". Measured on the published upgrade 05 file,
15.02 percent of applicable rows are constrained -- 9.31 percent on capacity and 6.49 percent
on breaker space -- with 208,694 of 245,570 reading No Constraint (B.9).

**15. Section 2.4, Connecticut.** The guide says "CT is an open question ... Nothing in the
2025.1 README or changelog says whether county FIPS moved to the nine planning regions". It
did not move. 2025.1 carries the same eight pre-2023 CT counties, `G0900010` through
`G0900150` (B.5). The question is closed and the existing CT handling stands.

**16. Section 4, Phase 1 file names.** The guide's Phase 1 list names
`README_resstock_20251.pdf` and `measure_name_crosswalk.csv`. The published names are
`README_resstock_2025_1.pdf` and `measure_name_crosswalk_res_2025_1.xlsx`. The guide also does
not mention the metadata parquet layout: the files are at
`metadata_and_annual_results/national/full/parquet/upgrade{N}.parquet`, and the baseline is
`upgrade0.parquet`, not a file named `baseline`.

**17. Section 1, the definition of done.** It requires "A CONFIRMED reference-values row set
for the package, from a full run, in `docs/REFERENCE_VALUES.md`". That file was not in the
repository at audit time, and neither was `docs/SESSION_LOG.md`, though CLAUDE.md cites both.
The researcher added both on 19 September 2026, so this is closed. `REFERENCE_VALUES.md` now
carries a release-vintage note recording that every row in it is a 2022.1.1 measurement and
that no 2025.1 reference value exists yet.

**Claims that the audit confirmed rather than contradicted**, recorded so they are not
re-litigated: the two dual fuel option strings are exactly as published (B.2); the AFUE tier
splits on IECC zone with no crossover (B.2); applicability is 44.65 percent of the stock,
matching Table 3 of the measure documentation exactly (B.6); `upgrade.heating_fuel` and
`upgrade.hvac_heating_type_and_fuel` are `None` for 05 and 04 (B.3); electric backup is zero
under 05 (B.7); all recipients get 100 percent whole-home cooling (B.2); `HVAC Has Ducts|Yes`
and the not-shared condition both hold on every applicable row (B.6); AK does have modeled
secondary heating, in `in.hvac_secondary_heating_fuel`, `_type`, `_efficiency` and
`_partial_space_conditioning`; and `upgrade.hvac_detailed_performance_data` is populated for
04 and null for 05, as Section 2.3 states.

---

## E. Blockers

The first draft of this memo listed six. Decisions 1 and 2 close three of them. What remains
is one substantial piece of work and two narrower items.

**E.1 No two-fuel representation exists anywhere in the pipeline.** Three independent places
assume the post-retrofit fuel is electricity: the consumption column read
(`process_euss_data.py:533, 552, 584`), the price lookup
(`calculate_lifetime_fuel_costs.py:566-578`), and the fossil-emissions branch
(`calculate_fossil_fuel_emissions.py:62`). 81.0 percent of dual fuel heating site energy is
gas (B.7), so this is not an edge case. This is the largest single piece of work implied by
the data, it is what decision D5 turns on, and it is the one place where the
minimal-change constraint and the data genuinely pull against each other.

**E.2 The peak pass-through columns are a rename with a conditioning caveat.** Downgraded from
a blocker by Decision 2. The four columns map cleanly:
`peak_when_heating.kw` to `out.qoi.electricity.maximum_daily_peak_winter..kw`,
`peak_when_cooling.kw` to `..._summer..kw`, and the two `.savings` variants to the matching
`..._savings_...` columns, whose published description reads "Reduction on the maximum value
in Dec/Jan/Feb". The caveat to carry into the Tepper data dictionary: the 2022 columns are
conditioned on the end use running while the 2025 columns are conditioned on the calendar
season, so the zero counts differ sharply. At baseline, `peak_when_cooling` is zero for 88,401
rdu with no cooling while `maximum_daily_peak_summer` is zero for only 39; on the heating side
it is 25,943 against 22. The means shift accordingly (cooling 6.503 kW against 7.959, heating
8.475 against 9.554). Cross-release comparison of these columns is therefore not like for
like, though within 2025.1 they are internally consistent.

**E.3 The grid-impact path is pinned to the 2022.1.1 Athena table.**
`METADATA_TABLE = "resstock_amy2018_release_1_1_metadata"` at `constants.py:433`, and
`ELEC_TOTAL_COL` at line 427 uses the 2022 BSQ column form. The guide already defers grid
impact to Section 7, so this is a note rather than a new blocker, but it means
`GRID_IMPACT_ANALYSIS = True` (`constants.py:116`) cannot run against 2025.1 as configured.

**E.4 The raw CSV read uses positional dtype indices.**
`tare_baseline_v3_0.ipynb` cell 2 line 31 sets
`columns_to_string = {11: str, 61: str, 121: str, 103: str, 113: str, 128: str, 129: str}`.
These are column positions in the 2022.1.1 CSV. They are meaningless against a 771-column
2025.1 file, and the failure would be silent: wrong columns cast to string rather than an
error. Reading the parquet files instead avoids the issue entirely, since parquet carries its
own types.

### Closed by the 19 September decisions

**Closed -- the AK and HI county code format.** `in.county` is a name string rather than
GISJOIN for 3,418 baseline rdu across those two states, which broke the FIPS slice at
`process_euss_data.py:218` and everything keyed on `county_fips`. Decision 1 excludes both
states, after which every remaining value is GISJOIN (B.5).

**Closed -- the Cambium GEA crosswalk has no AK or HI rows.**
`county_to_gea_mapping_cambium23.csv` covers neither state (A.9), which would have left those
homes with a NaN `gea_region` and out of climate damages. Moot under Decision 1.

**Closed -- the peak pass-through columns.** Downgraded to a rename by Decision 2; see E.2.

### Not blockers, listed so they are not mistaken for blockers

Connecticut is resolved and needs no change (B.5). Excluding AK and HI leaves exactly 3,107
distinct GISJOIN counties, matching 2022.1.1, so no geography code needs to change (B.5). The
`out.params.size_*` columns all exist, including the backup furnace size, so the capital-cost
path has the inputs it needs (B.8). `applicability` is already a real boolean, so the guide's
defensive cast is a no-op on these files (B.1). The fuel-price, AEO2026 projection and ACS AMI
tables all cover AK and HI (A.9), which is now moot but was never the constraint.
