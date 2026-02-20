# TARE Model Repository: Commit Review & Deliverable Progress Report

**Report Date:** February 7, 2026

**Repository:** jordan-joseph126/cmu-tare-model

**Claude Commit Review and Progress Tracked:** 66 commits and over 280 updated files

NOTE: Claude was used to review the codebase in it's entirety and all updates throughout the postdoc, map the updates to deliverables and subtasks, and provide a percentage complete estimate. I provided an outline of the deliverables and substasks based upon past meeting agendas/notes and conversations with team members. I re-wrote the report in my own words, corrected any mistakes, and re-organized its contents to provide a coherent summary of the work done and how it advances the Trane Grand Challenge goals and deliverables.

---

## DELIVERABLE 1: RUNS OF MODEL TO ASSESS ADOPTION POTENTIAL

### Completed Tasks

**Discount rate sensitivities (Previously fixed at 7%)**   

- AMI-based linear interpolation for variable discount rate (45% at 0% AMI → 7% at 150%+ AMI)   

- 3 fixed rates (low 2%, base 7%, high 12%) 

- Relevant commits and files: [`7f11e17`, `f113a1f`, `3eb78ae`; `discounting.py`, `constants.py`]   

**Capital cost fix: primary capacity vs. total**    

- Uses primary heating capacity (not total) for capex calculations  

- Relevant commits and files: [`3eb78ae`; `calculate_equipment_installation_costs.py`, `calculate_lifetime_private_impact.py`]  

**NPV equation refactoring**    

- Private NPV with defensive column validation, capital cost calculation separated into helper function 

- Relevant commits and files: [`3eb78ae`, `4fa5c6d`, `bbd8ccb`; `calculate_lifetime_private_impact.py`] 

**Code streamlining: single scenario file** 

- Consolidated from 4 separate scenario notebooks (basic/moderate/advanced/baseline) to unified `tare_scenarios_v2_2.ipynb` 

- Relevant commits and files: [`05e9b1e`, `cd5b9f8`; `model_scenarios/`, `tare_scenarios_v2_2.ipynb`]   

**Memory/performance optimization (~75% target)**   

- Vectorized np.where() operations, no-copy-by-default validation, reference-based DataFrames in df dictionary (~4.9 GB savings)    

- Chunked CSV loading (50K rows) and explicit gc.collect()  

- Relevant commits and files: [`588cd0a`, `bbd8ccb`, `4fa5c6d`; `validation_framework.py`, `load_exported_results_to_df.py`, `calculate_lifetime_public_impact_sensitivity.py`] 

**Sensitivity results restructuring**   

- Saved to 12 manageable DataFrames instead of 3 very large ones (1200+ columns)    

- Relevant commits and files: [`ebf72a5`; scenario notebooks]   

**Climate NPV single-calculation optimization** 

- Elimated duplicate computation: Climate NPV calculated once outside CR function loop (doesn't vary by CR function)    

- Relevant commits and files: [`375a88a`; `calculate_lifetime_public_impact_sensitivity.py`]    

**Export/load function refactoring**    

- Updated to handle discount rate sensitivities 

- Made directory structure more clear for exported results and location for loading 

- Chunked loading with progress tracking    

- Relevant commits and files: [`b180540`, `e70ce8e`, `588cd0a`, `1981e73`; `export_model_run_results.py`, `load_exported_results_to_df.py`] 

**5-step validation framework**

- Data integrity and performance: Does not performing calculations for N/A categories (which previously messed up summary stats)

- Sample representativeness: Improves upon previous approach to dropping the rows with invalid fuel or technology types 

- Unified masking approach across all calculation functions (initialize → create series → valid-only calc → valid-only update → final mask)

- Relevant commits and files: [`4fa5c6d`, `bbd8ccb`; `validation_framework.py`, `calculation_utils.py`]

**IRA rebate logic preserved**

- Policy scenario handling (No IRA vs. AEO2023 Reference) with rebate-adjusted capital costs

- Relevant commits and files: [`calculate_lifetime_private_impact.py`]

**Health impact data update**

- Updated to MSC data with $11.45M VSL USD2023 (previously used former federal guidance on VSL)

- Relevant commits and files: [`3e32d3f`; `create_lookup_health_impact_county.py`]

**Health impact performance improvement**

- Optimized health impact code for national model run: vectorized computation, np.where(), etc.

- Relevant commits and files: [`c263445`; `calculate_lifetime_health_impacts_sensitivity.py`]

**Successful national model runs**

- Multiple confirmed (Nov 2025, Jan 2026)

- Relevant commits and files: [`2b5f40b`, `c263445`, `05e9b1e`]

**Archive legacy code and clean up codebase**

- 74 files (163K lines) moved to archived_files, repo cleaned for Zenodo release

- Relevant commits and files: [`1c93a5b`]

**Cooling/heating category handling**

- Performance improvement: Conditional category processing

- Reduced dataframe size: Cooling metadata columns only for replacement costs, not full consumption. Used for total heating replacement cost scenarios. 

- Fixed to prevent double-counting $8000 rebate since ASHP retrofits apply to both heating and cooling systems

- Relevant commits and files: [`7e07e45`, `4fa5c6d`; `process_euss_data.py`]

**T&D losses data update**

- Previous studies used 6% assumption. Updated from 6% to 5%

- Relevant commits and files: [`611d60a`]

**Functional equivalence testing v2.1 vs v2.2**

- v2.1 scenario files restored for comparison

- Relevant commits and files: [`b1d972e`, `6552fe8`]

### In Progress Tasks

**REMDB v4 capital cost integration**

- v4 code (regression-based, deterministic) has been started in the scenario notebooks, but I need to get it working for state and national model run

- TODO: Capital cost estimates are an order of magnitude different between replacement and retrofit estimates (way too large)

- Relevant commits and files: [Evidence: `68b5ee6`, `ab7f24d` (added then reverted), `7e07e45`; all `remdb_v4_update/` files]

**Non-HVAC end-use cost estimation (v4)**

- Water heating, clothes drying, cooking code exists in `remdb_v4_installed_cost_utils.py` but is commented out (lines 141-146).

- Only heating and cooling are implemented for v4. Main focus of Trane GC is heating and cooling so this is fine. 

- Relevant commits and files: [Commented-out code in util files]

**Fuel type separation (ASHP vs. electric resistance)**

- The `process_euss_data.py` refactoring handles different fuel types.

- TODO: Update visuals to show separation or add "existing equipment" category back to the adoption potential stacked bar charts

- Relevant commits and files: [`process_euss_data.py`]

### Completion Estimate: 75-90%

**Rationale:** The core model infrastructure is solid — NPV calculations, discount rates, validation framework, performance optimizations, and export/load functions. National runs have been completed successfully. However, the REMDB v4 integration (a key deliverable for capital cost accuracy) is blocked by an order-of-magnitude discrepancy in costs. The capital cost issue is a significant concern that must be resolved before adoption runs can be trusted. Need to resolve REMDB v4 concerns, test the refactored codebase, and do a full model run before I can provide the updated paper adoption potential visuals. 

---

## DELIVERABLE 2: PEAK LOAD INCREASES/HOTSPOTS

### Completed Tasks

**Cross-cutting tasks**

- Performance improvements

- Model run config/constants 

- Single scenario file allows for more MPs to be integrated and also ResStock data updates (2025.1 release)

- Reducing verbose printing output

- Updated TARE model for using AWS to obtain the ResStock data through the OEDI. This can be used to query certain homes (useful for home grouping and feeder analysis)

- The above steps also make it possible to integrate the new ResStock data and consequently load savings, panel upgrades, DR, and load shifting, EVs, etc. and additional technologies (dual fuel, cold climate heat pumps, etc.). 

**Heating/cooling load sizing data available**

- `heating_load_kBtuh` used for equipment sizing in capital cost calculations

- Relevant commits and files: [`calculate_equipment_installation_costs.py`, `remdb_v4_installed_cost_utils.py`]

**HDD/CDD consumption adjustment utilities**

- `degree_day_consumption_utils.py` builds on `hdd_consumption_utils.py` by calculating weather-adjusted annual energy consumption for cooling (CDD) and heating (HDD)enduses

- Currently not concerned with cooling consumption and individual NPV, only cooling costs as part of the replacement cost bundle. 

- Relevant commits and files: [`degree_day_consumption_utils.py` and `hdd_consumption_utils.py`]

**Grid emissions data integration**

- eGRID and CAMBIUM electricity system projections used for emissions calculations (GEA Region and REEDS Balancing Area geography)

- Relevant commits and files: [`create_lookup_emissions_electricity_*.py`]

### In Progress Tasks

**Update the codebase for ResStock 2025.1 release data**

- TODO: Create a new set of files for analyzing timeseries data

- TODO: Update column names to match new naming convention

**Load profile analysis**

- Write code to query timeseries data using AWS. (Already done this for annual results and metadata files)

- Continue making performance improvements as part of the enhancement and refactoring process

- TODO: Expand model functionality to handle timeseries data

- TODO: Determine relevant metadata columns for feeder matching, adopting TPIA-MP or GRID-Lab methods

- TODO: Update model to group sets of individual dwelling units and combine their load profiles to determine the maximum peak before and after retrofit

- TODO: Validate estimates with the peak load column and peak load savings columns in the annual/metadata files (this is for each individual home and does not include timestamp)

**Hotspot identification logic**

- Code written for primary heating fuel analysis and "spark gap" serve as indicators for adoption potential and map well to actual data

- Primary heating fuel analysis is at the census tract level and can help further pinpoint hotspot areas

- TODO: Write code to determine magnitude increase in peak load impacts at the county level (total magnitude increase in MW and magnitude increase as a % value)

- TODO: Visualize national chloropleth map with the peak load impacts shown at the county level

- TODO: Perform a case study for Allegheny County 

- TODO: Expand case study to include dual fuel and cold climate systems

- TODO: Validate results against utility (e.g., Duquesne Light Company, West Penn Power/First Energy) and organizational analysis (Pearl Edison and Rewiring Communities analysis)

### Not Yet Started

**Peak load calculation code**

- No hourly or sub-hourly load analysis

### Completion Estimate: 25-50%

**Rationale:** The existing codebase provides foundational data (equipment loads, weather-adjusted consumption, grid region mapping) that could support future peak load analysis, but no actual peak load analysis has been conducted yet. This represents the largest gap among all four deliverables.

---

## DELIVERABLE 3: PAPER DRAFTING

### Completed Tasks

**Paper-quality visualization infrastructure**

- 600 DPI default across all visualization modules; multi-panel subplot grids; adoption tier bar charts; income-stratified boxplots; fuel-type color coding

- Visuals updated as part of the Applied Energy paper revision

- Relevant commits and files: [`visuals_adoption_potential.py` (932 lines), `data_visualization_histograms.py`, `data_visualization_boxplots.py`]

**Results export functionality**

- Structured export by results category (summary, damages, fuel costs) with discount rate organization

- Relevant commits and files: [`export_model_run_results.py`, `load_exported_results_to_df.py`]

**Adoption potential percentage calculations**

- Population-weighted adoption metrics with `print_adoption_decision_percentages()` function

- Relevant commits and files: [`visuals_adoption_potential.py`]

**Multi-index adoption DataFrame creation**

- `create_multiIndex_adoption_df()` for structured analysis output 

- Relevant commits and files: [`visuals_adoption_potential.py`]

**Methods documentation for paper**

- Private capital cost methodology document (653 lines), REMDB v4 architecture docs, validation framework docs

- Relevant commits and files: [`TARE_Private_Capital_Cost_Estimation_Documentation.md`, `REMDB_v4_Refactoring_Documentation.md`]

**Repository cleaned for Zenodo release**

- Codebase archived files have been cleaned

- Linked Zenodo repository and GitHub to simultaneously update DOI and information with each new public release

- Work with Tamar to ensure that the codebase functions well for other users (and different OS)

### In Progress Tasks

**Re-scoping the paper**

- Will be covered in continued meetings with Costa and Valerie after the capital cost estimate discrepancy is resolved.

**Statistical validation analysis**

- Adoption decision percentages can be computed but no formal comparison against external validation data documented in commits.

**Separate the ASHPs from the electric resistive heating technology**

- Updating visuals to include the "Existing Equipment" category in the figure legend and stacked bar chart color code

**Revising Manuscript**

- I marked up the previous version of Arnav's manuscript based off of Trane meetings and Jason's feedback on the paper

- Uploaded a first draft of the updated manuscript

- TODO: Decide upon the approach for the future feeder analysis.

- TODO: Follow up with Arnav about the TPIA-MP and paper collaboration. As it stands he and Amrit would likely be last authors or an acknowledgement because the analysis is being completely redone and the paper re-scoped.

### Completion Estimate: 50-75%

**Rationale:** The visualization code produced publication-ready figures (600 DPI, multi-panel, professional formatting) and methods documentation is thorough. The repository has been explicitly cleaned for paper reference. The analysis outputs needed to populate paper tables/figures depend on resolving the capital cost discrepancy (Deliverable 1).

---

## DELIVERABLE 4: PROJECT DOCUMENTATION

### Completed Tasks

**Comprehensive README**

- Installation (Windows/Mac/Linux), environment setup, repository structure, data download (Zenodo), quick start, troubleshooting 

- Relevant commits and files: [`README.md`; `db5734a`, `6f09946`, `31c0d95`, `0cf33c3`]

**Environment setup guide**

- Step-by-step Conda/Jupyter configuration

- Relevant commits and files: [`ENVIRONMENT_SETUP.md`; `5b9773a`, `45c9728`]

**Private capital cost estimation documentation**

- Technical deep dive on REMDB v3 methodology, cost components, sampling approach 

- Relevant commits and files: [`TARE_Private_Capital_Cost_Estimation_Documentation.md`; `e809e0d`]

**REMDB v4 refactoring documentation**

- Detailed document covering regression methodology, data-driven unit conversions, row ID mapping 

- Relevant commits and files: [`REMDB_v4_Refactoring_Documentation.md`; `7e07e45`, `68b5ee6`]

**REMDB v4 integration plan**

- Prioritized checklist for integrating v4 into production

- Relevant commits and files: [`Integrate_REMDB_v4_Cost_Estimation.md`; `7e07e45`]

**Coding standards checklist for use with Claude**

  REFACTORING GOALS:

  1. Identify and eliminate redundant code patterns

  2. Simplify over-engineered sections by using existing library functionality

  3. Improve code clarity and maintainability

  4. Maintain ALL existing functionality (no behavior changes)

  5. Follow my coding preferences: Google-style docstrings, type hints, strategic comments, 
    DRY principles

  CRITICAL REQUIREMENTS:
  
  - Work through the refactoring STEP BY STEP

  - For EACH proposed change, provide:

    * Before code (5-7 lines of context)

    * After code (with improvements)

    * Clear explanation of WHY the change improves the code

    * Confirmation that functionality is preserved

  - Do NOT provide a complete refactored file all at once

  - After suggesting changes, review them to ensure functional equivalence

**Inline code comments explaining "why"**

- Extensive comments throughout codebase (examples shown below):

  - `validation_framework.py`: "MEMORY OPTIMIZATION: By default, this function no longer creates a copy..."

  - `calculate_lifetime_public_impact_sensitivity.py`: "Key change from v2.2: Climate NPV is calculated ONCE..."

  - `process_euss_data.py`: "Resolves the excessive data columns and double counting with $8000 rebate"

  - `remdb_v4_installed_cost_utils.py`: "NO LONGER USING THE SUM OF HEATING AND COOLING LOADS FOR SYSTEM SIZE"

**Google-style docstrings on public functions**

- Type hints and docstrings across all modules

- Relevant commits and files: [`all .py files`]

**Data source documentation**

- Zenodo URL, ResStock Release 2022.1, eGRID, CAMBIUM, REMDB references

- Relevant commits and files: [`README.md`, various data_processing files]

### In Progress Tasks

**README version update**

- README still says "v2.1" but active development is on v2.2/v2.3. Version information needs updating.

- Relevant commits and files: [`README.md`, various data_processing files]

**UPDATE change log**

- Probably sufficient to keep all documentation within the scripts themselves and provide high level updates with each new release. May be best to delete/archive individual changelog files

- Relevant commits and files: [`CHANGELOG.md`]

**REMDB Data Usage and Capital Cost Estimation Methods Documentation**

- Combine individual documentation from the v3 and v4 code and discussion notes into a unified document

- Update the unified documentation into the plain language model guide

- TODO: Export summary capital cost estimate columns to an excel file (sheet for cooling, heating, column name mapping/explanation)

**Plain language model guide**

- Technical documentation is thorough 

- Sent out initial draft of plain language model guide but have yet to receive feedback

- Will update with all recent changes and fix discrepancies (remove TPIA-MP from the guide and incorrect adoption tier breakdown)

- ResStock is referenced in code, but the plain language guide should also explain how ResStock data flows through the model (which columns, transformations, assumptions, etc.)

**docs/index.md**

- Empty placeholder file; documentation framework started but not populated.

### Completion Estimate: 75%

**Rationale:** Documentation is a clear strength of this project. The README, environment setup, and technical documentation are comprehensive and well-maintained. Inline code comments are exemplary, explaining rationale rather than just mechanics. The coding standards checklist and file organization artifacts show intentional documentation practices. Gaps exist in: version number consistency (v2.1 in README vs v2.2 in code) and need to update the plain language model guide.
