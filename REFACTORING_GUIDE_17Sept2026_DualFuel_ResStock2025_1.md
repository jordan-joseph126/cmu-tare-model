# Refactoring Guide -- ResStock 2025.1 Dual Fuel Heat Pump (Upgrade 05) in the TARE Model

**Drafted:** 17 September 2026
**Scope:** Make the full TARE pipeline (consumption -> capital cost -> rebates -> fuel costs ->
NPV -> adoption -> climate damages -> KPIs/visuals) run for the ResStock 2025.1 Dual Fuel
Heating System package, with the same outputs it produces today for MP3 and MP4 on
ResStock 2022.1.1.
**Status:** Planning document. Nothing here has been applied to the codebase.

**Superseded 19 Sep 2026:** the byte-identical-2022.1.1 regression guarantee described in
Section 1 and referenced throughout the phase task lists below no longer applies. See
CLAUDE.md's "Regression guarantee -- dropped" section and
`docs/NEXT_STEPS_2026-09-19_DualFuel_ResStock2025_1.md` for the current requirement (none)
before drafting any future phase's session prompt from this guide.

> CLAUDE.md is the source of truth for conventions, file rules, reference values, coding
> standards, and non-negotiables. This guide only adds what CLAUDE.md does not already know:
> what changed in the data, what that breaks, the decisions that must be made first, and an
> ordered, gated task plan. Each Phase below is sized to become one Claude Code session prompt.

**How this guide was built and its one big caveat.** The ResStock side is grounded in the
2025.1 sources listed in Appendix A (release README, data page, measure documentation, the
SDR upgrade YAML at the `2025_Release_1` tag, the raw input/output dictionaries, the
changelog, and the column-applicability issue report). The TARE side is grounded in
CLAUDE.md's documented architecture, NOT a read of the actual code. Every claim about what a
TARE module does today must be confirmed in the Phase 1 audit before any edit. Where a
2025.1 published column name is inferred from the naming convention rather than read from
`data_dictionary.tsv`, it is marked **(verify)**.

---

## 1. What "the same functionality as MP3/MP4" means (definition of done)

For one measure package `mp`, the current pipeline produces, per CLAUDE.md and the Session
Log. The dual fuel run is done when every item below exists for the dual fuel package with
correct values:

| Output family | Concrete artifacts (today, per MP) |
|---|---|
| Consumption deltas | Degree-day-adjusted baseline vs. retrofit heating and cooling energy; `baseline_total_site_consumption`; `mp{mp}_modeled_savings_frac` |
| Capital cost | Heat-pump upgrade cost (REMDB v4, keyed on SEER1 and `size_*`); old-system replacement cost priced off `base_size_heating/cooling_system_primary_k_btu_h`; MP3-vs-MP4 replacement-cost identity check |
| Rebates | `mp{mp}_heating_rebate_amount_{cost_scenario}`, `mp{mp}_heating_rebate_amount_june2026_{cost_scenario}`, `mp{mp}_rebate_eligibility_ira2024`, `mp{mp}_rebate_eligibility_june2026` (labels `'HEEHR'`/`'HOMES'`/`'None'`); SD participation gate |
| Fuel costs | Lifetime (2025-2039) fuel cost streams per end use, projected with AEO2026 price factors and HDD/CDD factors; `ref2025_mp{mp}_cooling_lifetime_savings_fuel_cost` and the `_negative` flag |
| NPV | Nine cases per MP (`heatingSavings_coolingLCC`, `heatingLCC_coolingSavings`, `heatingLCC_coolingLCC` x `_unsub`/`_sub`/`_sub_june2026`) x four discount rates; NPV identity (savings - net_capital_cost = NPV) and the CLAUDE.md NPV ordering checks hold |
| Adoption | Nine `ref2025_mp{mp}_{npv_case}_econ_adopter{method_suffix}` columns; `NPV >= 0` |
| Climate | `ref2025_mp{mp}_heating/cooling_avoided_mt_co2e_{lrmer,srmer}` and `_damages_climate_{lrmer,srmer}_central` |
| KPIs / visuals | County-level adoption rate and operating-cost % change; adoption dotplot; choropleths; Tepper household/county exports |
| Grid impact | County demand profiles and peak (`compute_county_scenario_profile`, `plot_county_demand_grid`) under `GRID_IMPACT_ANALYSIS` -- **deferred, see Section 7** |
| Reference values | A CONFIRMED reference-values row set for the package, from a full run, in `docs/REFERENCE_VALUES.md` |

Everything the current MP3/MP4 path produces on 2022.1.1 must keep producing **byte-identical**
output (`2026-09-02_19-04` is the oracle). That regression guarantee is a hard requirement of
every phase, not a final check.

---

## 2. ResStock 2022.1.1 -> 2025.1: what changed, and what it breaks in TARE

### 2.1 Dataset-level changes

| Change | 2022.1.1 (current) | 2025.1 | TARE impact |
|---|---|---|---|
| Weight per rdu | `242.131013` | `253.9` (README; data page). Uniform across the release. | Every hardcoded `242`/`242.13` breaks; CLAUDE.md Terminology, reference conversions, the "count under ~242 is rdu" rule, and one anti-pattern line are 2022-specific. Code that reads `weight` from the frame is fine (`fuel_counts_millions`, Tepper `home_count` tolerance already do). |
| Sample count | 331,531 baseline rdu in TARE's frame (after masking/failures) | ~550,000 simulated; "roughly 500,000" published rows (README) | All row-count reference values are 2022-specific. New baseline count must be measured, not assumed. |
| States | 48 + DC | 50 + DC -- **Alaska and Hawaii added** | New state keys `'AK'`, `'HI'`. TARE's fuel-price CSV, degree-day factors, rebate participation set, and county shapefile join must all be audited for AK/HI coverage. AK has modeled secondary heating (`in.hvac_secondary_heating_*`). |
| Counties | 3,098 in TARE's frame | 3,100+ (data page) | County count changes; CT geography must be re-audited (Section 2.4). |
| Weather years | AMY2018 | AMY2018 (Oct 2025) and AMY2012 (Nov 2025) | Keep AMY2018 for continuity. OEDI path: `2025/resstock_amy2018_release_1`. |
| Upgrade count / numbering | 10 packages; TARE uses MP3, MP4 | 28 packages, **all new**; numbering does not carry over | MP3/MP4 do not exist in 2025.1. The dual fuel package is **Upgrade 05** (`hvac_005`). See 2.2. |
| Column naming | Units implied or single-dot (e.g. `out.site_energy.total.energy_consumption.kwh`, `out.params.size_heating_system_primary_k_btu_h`) | **Units at the end, after `..`** (e.g. `out.electricity.plug_loads.energy_consumption..kwh`); "several other renames"; weighted results and intensities added to the metadata file | Every literal column string in the loader breaks. A release-aware column map is required (Phase 2). Ignore the new weighted and intensity columns (TARE weights itself). |
| Efficiency metric | SEER / HSPF (SEER1) | **SEER2 / HSPF2** | REMDB v4 cost regression is keyed on SEER1 ("only SEER1 (pm2) feeds the REMDB v4 upgrade cost"). Needs an explicit SEER2->SEER1 (and HSPF2->HSPF) conversion at the cost-model boundary. |
| Emissions factors | Cambium 2024 (TARE's own pipeline) | Cambium 2024 in ResStock too; end-use emissions removed from SDR (totals and by-fuel only) | No change to TARE's climate pipeline inputs. Do not switch to ResStock's emissions columns. |
| Utility bills | none | Published (2023 rates; electricity, NG, propane, fuel oil; fixed charges; AK-specific) | Optional cross-check for TARE's EIA USD2025 fuel-cost pipeline. Not a replacement. |
| Electric panels | none | Baseline panel rating (`in.electric_panel_service_rating`) and post-upgrade breaker-space and capacity constraint outputs | New optional capital-cost component (panel upgrade). Dual fuel doc: 8.0% of applicable homes constrained. Decision D7. |
| Failed simulations | rows removed from all upgrades | **Failures copied from baseline** and marked not applied (changelog, 2025-08-12) | Filter upgrade rows on `applicability == True`. Never assume every row in the upgrade file changed. |
| `applicability` dtype | boolean | boolean in baseline, was string in upgrade parquets (bug fixed 2025-07-08, but cast defensively) | Coerce to bool explicitly on read. |

### 2.2 The Dual Fuel Heating System package (Upgrade 05), exactly as published

Source: measure documentation NLR/TP-5500-96414 and `project_national/sdr_upgrades_tmy3.yml`
at the `2025_Release_1` tag.

**Technology**
- Single-stage ducted ASHP, **SEER2 15.2 / HSPF2 7.8** (ENERGY STAR minimum), capacity
  retention 0.5 at 5 F, sized to the cooling load per ACCA Manual S (+15% oversize in mild/humid
  climates, +15,000 Btu/h in cold-dry climates).
- **Integrated natural gas furnace backup**, condensing: **92.5% AFUE in IECC CZ 1-4, 95% AFUE
  in CZ 5-8** (the YAML says 92.5%; the README says 92% -- the YAML is what was simulated).
- **Switchover / lockout at 35 F**: furnace is the only heat source below 35 F; the ASHP is the
  only heat source above it. No overlap band.
- All recipients get **100% whole-home cooling** (`HVAC Cooling Partial Space Conditioning|100%
  Conditioned`), even if the baseline had no or partial cooling.
- Modeling limitation (their words): EnergyPlus uses the heat-pump fan airflow during backup
  operation, which understates heating-fan savings.

**Option strings written to the upgrade file** (these become the values of
`upgrade.hvac_heating_efficiency` / `upgrade.hvac_cooling_efficiency`):
```
HVAC Heating Efficiency|Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 95.0% AFUE NG, 35F switchover
HVAC Heating Efficiency|Dual-Fuel ASHP, SEER 15.2, 7.8 HSPF2, Integrated Backup, 92.5% AFUE NG, 35F switchover
HVAC Cooling Efficiency|Ducted Heat Pump
HVAC Cooling Partial Space Conditioning|100% Conditioned
```
Note the heating string says `SEER 15.2` but `7.8 HSPF2`. Do not parse it with the MP3/MP4
regex; write a dedicated parser (Phase 2, Task 2.3).

**Apply logic (verbatim structure from the YAML)**
- Package-level: heating fuel is NOT Wood, Other Fuel, or None.
- `HVAC Has Ducts|Yes`.
- NOT a shared system (`HVAC Shared Efficiencies` in {Boiler Baseboards Heating Only
  (Electricity | Fuel), Fan Coil Heating and Cooling (Electricity | Fuel), Fan Coil Cooling Only}).
- Climate split: cold anchor = CZ 5A, 5B, 6A, 6B, 7A, 7B, 7AK, 8AK; warm anchor = 1A, 2A, 2B,
  3A, 3B, 3C, 4A, 4B, 4C.
- **Natural gas hookup** = any of: `Heating Fuel|Natural Gas`, `Water Heater Fuel|Natural Gas`,
  `Clothes Dryer|Gas` or `Gas, Premium`, `Cooking Range|Gas`, `HVAC Secondary Heating
  Fuel|Natural Gas`, `Misc Pool Heater|Natural Gas`, `Misc Hot Tub Spa|Natural Gas`.
- Result: applies to **44.65% of the stock** (Table 3 of the measure doc; the doc also quotes
  44.7% and 50.8% in different places -- trust `applicability` in the data, not the prose).

**What this means for TARE's framing.** TARE today models fossil -> ASHP *full
electrification*: the fossil system is removed, post-retrofit heating is 100% electric, and
CLAUDE.md's Documented Limitation #2 says "Dual-fuel systems are not modeled." Upgrade 05
inverts that: the fossil system is *replaced by a new condensing gas furnace* that stays in
service below 35 F. So for every dual fuel recipient, post-retrofit heating uses **two fuels**,
and for propane/fuel-oil baselines the backup fuel *switches* to natural gas. This single fact
drives most of the refactor (fuel costs, emissions, rebates, capital cost).

### 2.3 Other 2025.1 packages worth loading alongside 05

| Upgrade | Name | Why it matters to TARE |
|---|---|---|
| 04 (`hvac_004`) | Typical Cold Climate Ducted ASHP (variable speed, HSPF2 8.5 / SEER2 17.5, electric resistance backup, optional duct-limited sizing) | The all-electric comparator to dual fuel. Nearest analog to MP4. Its `upgrade.hvac_detailed_performance_data` column is the only one that is correct for it. |
| 03 (`hvac_003`) | Minimum Efficiency Furnaces and Air Conditioners Circa 2025 (like-for-like replacement at federal minimums) | A simulated "replace at wear-out" counterfactual. Could eventually replace TARE's REMDB-based replacement-cost credit sizing. **Deferred.** |
| 01 (`hvac_001`) | Natural Gas Furnace 95% AFUE | Only upgrade for which `upgrade.heating_fuel` / `upgrade.hvac_heating_type_and_fuel` are correct (Section 2.5). |

There is no 2025.1 analog of MP3 (a minimum-efficiency single-stage all-electric ASHP).
`measure_name_crosswalk.csv` in the dataset root is the authoritative cross-release mapping;
download it in Phase 1 and record what it says about the 2022 packages.

### 2.4 Geography: Alaska, Hawaii, and Connecticut

- AK/HI are new. TARE's `eia_fuel_price_data_2025_usd2025.csv`, the AEO2026 fuel-price and
  degree-day factor CSVs, `NON_PARTICIPATING_REBATE_STATES`, and the ACS B19013 AMI file must
  each be checked for AK and HI rows. 2025.1 has AK-specific propane/fuel-oil prices and
  modeled secondary heating for AK homes.
- **CT is an open question.** 2022.1.1 carries the eight pre-2023 CT counties (FIPS
  09001-09015), which is why TARE pins `cb_2021` shapefiles and accepts the state-level AMI
  fallback. Nothing in the 2025.1 README or changelog says whether county FIPS moved to the
  nine planning regions (09110-09190). Phase 1 must read the distinct `in.county` values for
  `in.state == 'CT'` before anything geographic is touched.
- `in.county_metro_status` and metro/micro statistical area fields are new and may be useful
  for reporting, but nothing in TARE depends on them.

### 2.5 Columns TARE reads, and their 2025.1 counterparts

Confirmed from documentation = **C**. Inferred from the naming convention + raw dictionary,
must be verified in Phase 1 = **(verify)**.

| Purpose in TARE | 2022.1.1 (per CLAUDE.md) | 2025.1 | Note |
|---|---|---|---|
| Row identity | `bldg_id`, `upgrade`, `weight` | `bldg_id`, `upgrade`, `weight` -- **C** | `in.sample_weight` also exists in the raw inputs; use the top-level `weight`. |
| Upgrade applied? | (all rows applied) | `applicability` -- **C** | Coerce to bool; filter `== True`. |
| Baseline heating fuel | `in.heating_fuel` | `in.heating_fuel` -- **C** | Values incl. Electricity, Natural Gas, Propane, Fuel Oil, Wood, Other Fuel, None. |
| Baseline heating tech / efficiency | `hvac_heating_efficiency` (parsed) | `in.hvac_heating_type`, `in.hvac_heating_efficiency`, `in.hvac_heating_type_and_fuel` -- **C** | Existing-ASHP exclusion must key off these (`'ASHP'` substring), not off `upgrade.*`. |
| Baseline cooling | cooling type in {Central AC, Room AC} | `in.hvac_cooling_type`, `in.hvac_cooling_efficiency`, `in.hvac_cooling_partial_space_conditioning` -- **C** | Partial conditioning now matters (Decision D4). |
| Ducts / shared system | (not used) | `in.hvac_has_ducts`, `in.hvac_shared_efficiencies` -- **C** | Needed only if TARE re-derives applicability instead of trusting `applicability`. |
| Climate zone (for AFUE tier) | (not used) | `in.ashrae_iecc_climate_zone_2004` -- **C** | 92.5% vs 95% AFUE; also `in.building_america_climate_zone`. |
| Geography | county, state | `in.county`, `in.state`, `in.county_and_puma`, `in.puma` -- **C** | State is the 2-letter key TARE requires. |
| Income | (own ACS join) | `in.area_median_income`, `in.income`, `in.federal_poverty_level`, `in.state_metro_median_income` -- **C** | TARE keeps its own county AMI join; these are for cross-checks only. |
| Total site energy | `out.site_energy.total.energy_consumption.kwh` | `out.site_energy.total.energy_consumption..kwh` **(verify)** | Denominator of `mp{mp}_modeled_savings_frac`. |
| Electric heating | electricity heating (+ hp backup for MP3/MP4) | `out.electricity.heating.energy_consumption..kwh`, `out.electricity.heating_fans_pumps.energy_consumption..kwh`, `out.electricity.heating_hp_bkup.energy_consumption..kwh`, `out.electricity.heating_hp_bkup_fans_pumps.energy_consumption..kwh` **(verify)** | Raw names: `end_use_electricity_heating*_m_btu`. For dual fuel, `electricity.heating_hp_bkup` should be ~0. |
| **Gas furnace backup (NEW)** | n/a | `out.natural_gas.heating_hp_bkup.energy_consumption..kwh` **(verify)** | Raw: `end_use_natural_gas_heating_heat_pump_backup_m_btu`. **This is where the dual fuel furnace's gas lands.** Also `out.natural_gas.heating.energy_consumption..kwh` for the baseline furnace. Propane/fuel-oil equivalents exist for the baseline. |
| Cooling | electricity cooling | `out.electricity.cooling.energy_consumption..kwh`, `..cooling_fans_pumps..` **(verify)** | |
| Equipment sizes (cost multipliers) | `out.params.size_heating_system_primary_k_btu_h`, `..cooling..` | `out.params.size_heating_system_primary..k_btu_h`, `out.params.size_cooling_system_primary..k_btu_h`, **`out.params.size_heat_pump_backup_primary..k_btu_h`** (new relevance), `..size_heating_system_secondary..` (AK) **(verify)** | Raw names: `upgrade_costs.size_*`. The backup size is the **furnace** size for costing the new furnace. |
| Retrofit HP spec | `upgrade_hvac_heating_efficiency` string | `upgrade.hvac_heating_efficiency` (option string above), `upgrade.hvac_cooling_efficiency` = `Ducted Heat Pump` -- **C** | |
| Retrofit heating fuel | (implied electric) | **DO NOT USE** `upgrade.heating_fuel` or `upgrade.hvac_heating_type_and_fuel` -- both read `None` for every upgrade except 01 (known issue) | Derive from the option string + `in.heating_fuel`. |
| Panel (new) | n/a | `in.electric_panel_service_rating`; post-upgrade breaker-space and capacity metrics (`out.electric_panel...`) **(verify)** | Decision D7. |
| Utility bills (new) | n/a | `out.utility_bills.<scenario>...` **(verify)** | Cross-check only. |
| Emissions | (own pipeline) | totals and by-fuel only | Do not use. |
| Design capacities (new) | n/a | `out.hvac_capacity.heating..btu_h`, `..cooling..`, `..heat_pump_backup..` **(verify)** | Alternative to `out.params.size_*` if those are absent in the published file. |

---

## 3. Decisions the researcher must make before Phase 2

Each has a recommended default so Phase 1 can be planned. None is a code question.

| ID | Decision | Options | Recommended default | Why it is load-bearing |
|---|---|---|---|---|
| **D1** | MP numbering for 2025.1 packages | (a) reuse `mp{n}` with `n` = 2025.1 upgrade ID (dual fuel = `mp5`); (b) new tokens (e.g. `df`) | **(a)**, with a hard rule: one ResStock release per run, declared once in `constants.py` (`RESSTOCK_RELEASE = '2025.1'`), and `EQUIPMENT_SPECS` keyed per release. Never mix releases in one frame. | Every column builder (`define_scenario_params`, `create_npv_case_col`, `mp_str = f'mp{mp}'`) already assumes an integer `mp`. (a) reuses all of it. But CLAUDE.md warns "Revisit if MP8-10 are activated" -- 2022's MP5-MP10 tokens must never be confused with 2025.1's. |
| **D2** | Which 2025.1 packages to run | 05 only; 05 + 04; 05 + 04 + 03 | **05 + 04**. 04 is the all-electric comparator the manuscript will want. 03 deferred. | Loader and reference values scale with each package. |
| **D3** | Which homes are in scope | (a) trust `applicability == True` exactly; (b) `applicability` AND TARE's existing-ASHP exclusion; (c) fossil baselines only | **(b)**. ResStock applies 05 to electric-heated homes incl. existing ASHPs; TARE's study framing (Limitation #8) excludes existing-ASHP homes. | Changes the denominator of every adoption rate. |
| **D4** | Cooling for homes with no baseline AC | (a) keep TARE rule: cooling savings = 0 and cooling capital = 0 when `include_cooling = False`; (b) count added cooling energy as a cost and the ASHP's cooling capacity as capital, with no comfort credit | **(b)** for dual fuel, because ResStock forces 100% cooling on every recipient and (a) would hide a real operating and capital cost. Keep (a) for the 2022.1.1 path. Extends the existing "negative cooling savings accepted as real" decision. | Cooling LCC/Savings cases and the `_negative` flag share. |
| **D5** | Post-retrofit fuel mix and prices | n/a -- must model electricity (ASHP) + natural gas (furnace) for every recipient; propane/oil baselines switch backup fuel to NG | Model both fuels from the split ResStock already gives (`electricity.heating` vs `natural_gas.heating_hp_bkup`). Scale each by HDD factor. Document that the 35 F switchover makes the split temperature-dependent, so HDD scaling is an approximation (new limitation). | Fuel-cost stream and NPV. |
| **D6** | Capital cost of the new furnace | (a) ASHP cost only (REMDB, SEER1-converted); (b) ASHP + condensing furnace cost sized on `size_heat_pump_backup_primary` | **(b)**. The measure installs a new 92.5/95% AFUE furnace; ignoring it understates cost. Reuse TARE's furnace cost function from the replacement-cost path if one exists (audit). | NPV. |
| **D7** | Electrical panel upgrade cost | (a) ignore; (b) add a panel-upgrade cost for homes flagged capacity- or space-constrained post-upgrade | **(a)** for the first full run, with the constraint flag carried as a reporting column; (b) as a follow-up sensitivity. | Adds a cost component no MP3/MP4 run has. |
| **D8** | Rebate eligibility for a dual fuel system | (a) treat as an eligible ENERGY STAR ASHP under HEEHR/HOMES; (b) ineligible; (c) HEEHR yes / HOMES by savings fraction as today | Run **(a)** and **(b)** as the `_sub` bounds; the June 2026 HEEHR fossil gate is moot because no fossil system is removed, but eligibility of hybrid systems is exactly Limitation #7's uncertainty. Encode via `REBATE_RULE_CONFIG`, not ad hoc. | `_sub` / `_sub_june2026` cases. |
| **D9** | SEER2/HSPF2 -> SEER1/HSPF for REMDB | fixed factors vs. lookup | Use DOE Appendix M1 crosswalk factors (ducted split: SEER ~= SEER2 / 0.95, HSPF ~= HSPF2 / 0.85 -- **confirm against the M1 document before coding**). SEER2 15.2 -> ~SEER 16.0, which matches the MP3 ENERGY STAR override value already in TARE. | REMDB v4 cost input. |
| **D10** | Geography | include AK/HI or not; CT handling | Include AK/HI if every input table has rows for them; otherwise exclude explicitly and log the weighted count dropped. CT per the Phase 1 audit. | Denominators; map rendering. |

---

## 4. Phased task plan

Every phase = one Claude Code session. Every session: Task 1 is an audit with no edits;
one diff per stop gate; `.ipynb` and `*_EXPORT_*` files are never edited (module changes
plus a backport list instead); no commits. The 2022.1.1 MP3/MP4 pipeline must still
reproduce `2026-09-02_19-04` byte-identically at the end of every phase that touches shared
code.

### Phase 1 -- Acquire and audit (no code edits)

**Goal:** know exactly what the data looks like and exactly what the code assumes.

1. Download to `data/resstock_2025_1/` (AMY2018, national, parquet): baseline
   `metadata_and_annual_results`, upgrade 05, upgrade 04; plus `data_dictionary.tsv`,
   `enumeration_dictionary.tsv`, `upgrades_lookup.json`, `measure_name_crosswalk.csv`,
   `README_resstock_20251.pdf`. Record OEDI paths and file hashes in a `DATA_MANIFEST.md`.
2. Column audit: produce `docs/resstock_2025_1_column_map.csv` with every column TARE reads
   (Section 2.5) -> the 2025.1 name found in `data_dictionary.tsv`, unit, and dtype. Resolve
   every **(verify)** entry. Report any TARE input that has no 2025.1 counterpart.
3. Data facts to measure and write down: distinct `weight` values (expect one, 253.9); row
   count baseline and per upgrade; `applicability` dtype and True count for 05 and 04; distinct
   `upgrade.hvac_heating_efficiency` values for 05 (expect exactly the two option strings) and
   for 04; confirm `upgrade.heating_fuel` is `None` for 05 (known issue); distinct
   `in.county` for CT; `in.state` set (expect AK, HI present); any rows with
   `electricity.heating_hp_bkup > 0` under 05 (should be none or negligible).
4. Code audit (read-only): grep the modules under `cmu_tare_model/` and the current
   `*_EXPORT_*.py` snapshots for: every literal 2022 column string; every `242`; every `'mp3'`,
   `'mp4'`, `menu_mp == 3`; the SEER/HSPF parsing regex in `process_euss_data.df_enduse_compare`;
   how `calculate_lifetime_fuel_costs` chooses the post-retrofit heating fuel; how
   `calculate_lifetime_climate_impacts` treats post-retrofit fossil use; whether a furnace cost
   function exists in `add_remdb_metrics`' replacement path; whether `NON_PARTICIPATING_REBATE_STATES`,
   the fuel-price CSV, and the degree-day CSV cover AK/HI. Report file:line for each.
5. Deliver a findings memo and stop. The memo settles D1-D10 defaults or flags what is blocked.

**Verify:** the column map has zero unresolved rows; the memo lists every hardcoded 2022
assumption with a location. **Stop.**

### Phase 2 -- Release-aware loader (`process_euss_data.py` only)

1. Add `RESSTOCK_RELEASE` and a `RESSTOCK_COLUMN_MAP[release][logical_name] -> physical_name`
   registry (new module, e.g. `cmu_tare_model/utils/resstock_schema.py`). Every loader read
   goes through it. 2022.1.1 mapping must reproduce today's behavior exactly.
2. Baseline load: read `weight` from the frame (never a constant); read `in.*` fields from
   Section 2.5; coerce `applicability` to bool where present.
3. Upgrade load for 05: filter `applicability == True`; parse the dual fuel option string into
   `hp_seer2`, `hp_hspf2`, `backup_fuel`, `backup_afue`, `switchover_f` with a dedicated
   function and type hints; derive `retrofit_heating_fuels = ['electricity', 'natural_gas']`.
   Do not touch the MP3/MP4 parser; gate the ENERGY STAR override (`menu_mp == 3`) additionally
   on `RESSTOCK_RELEASE == '2022.1.1'`.
4. Extend `df_enduse_refactored` to carry, per home: baseline heating energy by fuel;
   retrofit `electricity.heating` (+ fans/pumps) and `natural_gas.heating_hp_bkup`;
   cooling; `base_size_*` copied from the baseline `out.params.size_*` (same pattern as the
   20 Aug 2026 fix); the new `size_heat_pump_backup_primary` for the furnace;
   `in.ashrae_iecc_climate_zone_2004`; `in.electric_panel_service_rating` and the post-upgrade
   constraint flags as pass-through columns.
5. Hand the researcher the notebook cells to backport (the TARE/EUSS load cells).

**Verify:** loading 2022.1.1 through the registry yields a frame equal to the current one
(`DataFrame.equals` on the columns TARE uses); loading 2025.1 upgrade 05 yields the measured
applicable count from Phase 1; `natural_gas.heating_hp_bkup` is non-zero for dual fuel
recipients and zero in baseline. **Stop.**

### Phase 3 -- Measure registry and masking (`constants.py`, masking helpers)

1. `EQUIPMENT_SPECS` gains a release-keyed entry for `mp=5` (dual fuel: SEER2 15.2, HSPF2 7.8,
   backup NG, AFUE by CZ, switchover 35 F) and `mp=4` (2025.1 ccASHP), without disturbing the
   2022 entries. `ALLOWED_TECHNOLOGIES['heating']` still excludes any `'Electricity ASHP'`
   variant (CLAUDE.md rule).
2. `include_heating` for 2025.1 = ResStock `applicability` AND TARE's fuel/tech mask AND
   not-existing-ASHP (D3). `include_cooling` per D4: keep the 2022 rule on the 2022 path; on
   the 2025.1 path add `include_cooling_added` for homes with no baseline AC that receive
   whole-home cooling.
3. `REBATE_ELIGIBLE_HEATING_MPS` becomes release-aware; add 5 (and 4) under 2025.1 per D8.

**Verify:** masked counts match Phase 1 measurements; 2022 masks unchanged (assert equality
against the current frame). **Stop.**

### Phase 4 -- Consumption and degree-day adjustment (`degree_day_consumption_utils.py`)

1. Generalize the retrofit heating adjustment from one column to a list of (fuel, column)
   pairs; scale each by the HDD factor; cooling by CDD as today. Keep the int-cast on year
   columns (CLAUDE.md mandatory read pattern).
2. `mp5_modeled_savings_frac`: numerator = degree-day-adjusted (baseline heating+cooling) -
   (retrofit electricity heating + NG backup + cooling); denominator unchanged
   (`baseline_total_site_consumption`).
3. Record the new limitation: the ASHP/furnace split is set by hours above/below 35 F, which
   HDD scaling does not capture; the sign of savings is fixed by ResStock's base-year split.

**Verify:** for a sample of dual fuel homes, adjusted totals equal ResStock base-year totals
when all factors are 1.0 (ANCHOR_YEAR); MP3/MP4 adjusted columns byte-identical. **Stop.**

### Phase 5 -- Capital cost (`add_remdb_metrics` and the capital-cost path)

1. Insert the SEER2/HSPF2 -> SEER1/HSPF conversion at the REMDB boundary as a named,
   documented function (D9). Never pass SEER2 into the SEER1 regression silently.
2. Upgrade cost = ASHP (REMDB, `size_*` of the retrofit) + condensing furnace (D6, sized on
   `size_heat_pump_backup_primary`). Replacement-cost credit unchanged: old heating and cooling
   systems priced off `base_size_*` and baseline efficiency, exactly as today.
3. Carry panel constraint flags through; no cost attached (D7 default).
4. Extend `validate_capital_costs.py` size-based groupings to the dual fuel package.

**Verify:** the MP3-vs-MP4 replacement-cost identity (0 mismatches) still holds on 2022.1.1;
on 2025.1, replacement cost for a given home is identical across 05 and 04 (same old system).
**Stop.**

### Phase 6 -- Lifetime fuel costs (`calculate_lifetime_fuel_costs.py`, `create_lookup_fuel_prices.py`)

1. Post-retrofit heating fuel cost = electricity stream (ASHP) + natural gas stream (furnace),
   each with its own AEO2026 price factor and the state 2-letter key. Propane/fuel-oil baselines
   price their backup at NG (fuel switch), and their *baseline* at propane/fuel oil.
2. Add AK/HI prices if the audit found them missing (EIA has both; AK propane/oil are
   state-specific in ResStock -- note the source used).
3. Keep the `_cooling_lifetime_savings_fuel_cost` sign convention and the `_negative` flag;
   under D4 the added-cooling homes will be negative by construction -- report their weighted
   share as MP3/MP4 do.

**Verify:** for homes with zero NG backup energy the result equals the all-electric formula;
`ref2025_mp5_*` fuel-cost columns exist for all four discount rates; 2022 streams byte-identical.
**Stop.**

### Phase 7 -- Rebates (`determine_rebate_eligibility_and_amount.py`, `REBATE_RULE_CONFIG`)

1. Add a dual fuel rule set to `REBATE_RULE_CONFIG` per D8: HEEHR eligibility for an
   ENERGY STAR ASHP with integrated gas backup (scenario flag), HOMES via
   `mp5_modeled_savings_frac` as today, June 2026 fuel gate evaluated but recorded as not
   triggered (no fossil removal). Keep the SD participation gate; add AK/HI status from the
   audit.
2. Write the four rebate columns for `mp5` with the existing names.
3. Extend `scripts/verify_june2026_rebate_fossil_gate.py` to pivot program x fuel for 05.

**Verify:** `summarize_rebate_funding` runs for 05; totals are reproducible; 2022 rebate
columns byte-identical. **Stop.**

### Phase 8 -- NPV, adoption, ordering checks (`calculate_lifetime_private_impact.py`, `determine_economic_adoption_potential.py`)

1. Confirm `create_npv_case_col(scenario_prefix, npv_case, method_suffix)` and
   `NPV_CASE_CATEGORIES` need no change; generate the nine `ref2025_mp5_*_private_npv*` and
   nine `*_econ_adopter*` columns in one block per MP (CLAUDE.md anti-pattern: never in a loop).
2. Run the NPV identity and the three CLAUDE.md NPV ordering checks on 05 and 04.

**Verify:** 0 identity violations to the half-cent; 0 ordering violations; adoption denominator
equals the `include_heating` count. **Stop.**

### Phase 9 -- Climate damages (`calculate_lifetime_climate_impacts`)

1. Post-retrofit emissions must include NG furnace combustion (site emissions) plus
   electricity (Cambium 2024 LRMER/SRMER as today). Confirm the module can accept a residual
   fossil stream; if it assumes zero post-retrofit fossil use, generalize it.
2. Produce `ref2025_mp5_heating/cooling_avoided_mt_co2e_{lrmer,srmer}` and damages columns.

**Verify:** avoided heating emissions for dual fuel are lower than for 04 in cold zones (gas
below 35 F) -- a sanity direction, not a value; 2022 climate columns byte-identical. **Stop.**

### Phase 10 -- KPIs, visuals, exports

1. Dotplot and choropleth builders take `mp=5` (labels from `EQUIPMENT_SPECS`, not literals).
2. County map: run the two-direction join-coverage check in `prepare_county_geodataframe`
   with AK/HI present (insets or exclusion per D10); confirm CT joins per the Phase 1 finding.
3. Tepper household/county exports for 05 and 04 with the existing structure.

**Verify:** 0 unmatched data rows in the county join; figures render; export row counts match
the frame. **Stop.**

### Phase 11 -- Full run, reference values, documentation

1. Full end-to-end run (National, 05 + 04, all discount rates); record the run timestamp.
2. `docs/REFERENCE_VALUES.md`: new section "ResStock 2025.1 -- Dual Fuel (05) and ccASHP (04)"
   with CONFIRMED rows (baseline fuel costs, nine NPV cases, nine adoption rates, climate means,
   rebate totals, replacement-cost identity). Never touch the 2022.1.1 rows.
3. Regression: re-run the 2022.1.1 MP3/MP4 path and confirm all 17 output files are
   byte-identical (SHA-256) to `2026-09-02_19-04`.
4. CLAUDE.md updates (one diff each): Project at a Glance (two releases); Terminology (weight
   is release-specific -- read it from the frame; the "~242" rule of thumb becomes "~ the
   release weight"); Data Sources (2025.1 rows); Column Naming (release registry);
   Sensitivity Dimensions (MP axis); Masking (D3/D4); Rebate Policy Scenarios (D8); Documented
   Limitations (#2 resolved for 05; add the 35 F/HDD limitation; #7 unchanged; #8 restated).
5. `docs/SESSION_LOG.md` rows per session; dated `docs/SESSION_CHANGELOG_*.md` per CLAUDE.md.

**Verify:** regression byte-identity holds; reference rows cite the run timestamp. **Stop.**

---

## 5. Things that will silently corrupt results if missed

- Using `upgrade.heating_fuel` or `upgrade.hvac_heating_type_and_fuel` on upgrade 05 (always
  `None`).
- Any surviving `242.13` constant, or a "count under 242 is rdu" heuristic applied to 2025.1.
- Passing SEER2 15.2 into the SEER1 REMDB regression unconverted (a cheaper, less efficient
  unit priced as if it were a worse one).
- Treating dual fuel post-retrofit heating as 100% electric (drops the furnace gas from fuel
  costs and emissions).
- Setting cooling savings and capital to 0 for no-AC homes on the 2025.1 path (D4) -- hides a
  cost ResStock explicitly modeled.
- Ignoring `applicability` and treating copied-from-baseline failure rows as retrofits
  (zero savings, zero cost, spurious NPV = 0 adopters).
- Loading the weighted or intensity columns 2025.1 now includes, then weighting again.
- Mixing 2022 and 2025.1 `mp` tokens in one frame.
- Assuming CT county FIPS are unchanged.

---

## 6. Session prompt seeds

Each phase above maps to a session prompt built with the `vscode-refactoring-session-prompt`
skill: Context = the phase goal and the D-decisions it depends on; Current state = the
Phase 1 memo facts for that module; Tasks = the numbered steps with their Verify line as the
gate. Phase 1's prompt needs only Sections 2.5 and 4.

---

## 7. Deferred (do not start now)

- Grid impact / peak demand for dual fuel (15-minute timeseries; winter peak shifts to gas
  below 35 F). Needs the 2025.1 `timeseries_individual_buildings` files and a decision on how
  `compute_county_scenario_profile` handles two heating fuels.
- Upgrade 03 as a simulated like-for-like replacement counterfactual replacing the REMDB
  replacement-cost sizing.
- Panel-upgrade cost as a sensitivity (D7b).
- Using ResStock's published 2023 utility bills as a check on TARE's EIA/AEO fuel-cost pipeline.
- AMY2012 weather-year sensitivity.
- MP-level cost sensitivity (v4LOW/v4HIGH) -- still blocked by the builder+consumer refactor
  noted in the 12 Jul 2026 session.

---

## Appendix A -- Sources consulted (17 Sept 2026)

- ResStock Data page: https://natlabrockies.github.io/ResStock.github.io/docs/data.html
  (release table: weight 253.9, 28 upgrades, 550,000 samples, AK/HI, field naming convention).
- ResStock 2025 Release 1 README (Nov 2025):
  https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/README_resstock_20251.pdf
  (column-naming changes, AK/HI, utility rates, panels, full upgrade table: dual fuel = Upgrade 05 / `hvac_005`).
- Measure documentation, Dual Fuel Heat Pump, NLR/TP-5500-96414 (May 2026):
  https://docs.nlr.gov/docs/fy26osti/96414.pdf (technology, sizing, switchover, applicability, panel results).
- SDR upgrade definitions at the `2025_Release_1` tag: `project_national/sdr_upgrades_tmy3.yml`
  in https://github.com/NatLabRockies/resstock (option strings and apply-logic anchors quoted in 2.2).
- Raw output/input dictionaries at the same tag: `resources/data/dictionary/outputs.csv`,
  `inputs.csv` (end-use, `upgrade_costs.size_*`, panel, utility-bill, and `in.*` names in 2.5).
- 2025 Release 1 changelog: https://resstock.readthedocs.io/en/latest/changelog/changelog_2025_R1.html
  (Cambium 2024, failed-upgrade handling, `applicability` dtype fix, duct-limited sizing).
- Known issue, 2025.1 column applicability:
  https://natlabrockies.github.io/ResStock.github.io/docs/resources/explanations/Issue_2025_1_Column_Applicability.html
- 2025.1 Technical Reference Guide (not read in full; cited by the above for panel and HVAC
  defaults): https://natlabrockies.github.io/ResStock.github.io/assets/trd/ResStockTechnicalReferenceGuide_2025_1.pdf
- Not obtained: `data_dictionary.tsv`, `enumeration_dictionary.tsv`, `measure_name_crosswalk.csv`,
  `upgrades_lookup.json` (S3 objects; download in Phase 1). All **(verify)** marks trace to this gap.
