# Adoption KPIs Refactoring Summary

## What was accomplished

- Added a new economic adopter routine in `cmu_tare_model/adoption_potential/determine_adoption_potential_sensitivity.py`.
  - Implemented `economic_adoption_decision(...)` to flag homes with positive incremental private NPV (`moreWTP > 0`).
  - Preserved the validation/masking framework so invalid-baseline homes stay `NaN`.
  - Added handling for the `lessWTP` boundary case needed for exact `Tier 1 + Tier 2` equivalence.

- Diagnosed the equivalence check between the new econ adopter signal and legacy adoption tiers.
  - Confirmed `econ_adopter == True` matches Tier 1 + Tier 2 on valid legacy rows for MP3.
  - Identified one MP4 mismatch caused by a `lessWTP == -0.0` boundary case.
  - Clarified that the larger `NaN` difference was due to comparing econ `NaN` versus legacy tier label counts rather than the same row set.

- Provided notebook-ready code for the new workflow.
  - Added an econ-column generation block to populate `DATAFRAMES_BY_MP[mp]['fixed_base']['inmap']` with the new econ adopter field.
  - Added an adoption-rate block that computes county-level adoption with `compute_adoption_rate(..., adopter_tiers=[True])`.
  - Added an optional plotting block using the new econ adoption rate results.

## Key implementation details

- Econ adopter column name:
  - `iraRef_mp{mp}_heating_econ_adopter_moreWTP_v4MID_fixed_base`

- Correct aggregation method for boolean econ adopter columns:
  - `compute_adoption_rate(..., adopter_tiers=[True])`

- Legacy adoption tier labels remain unchanged; the econ adopter path is a separate boolean decision signal.

## Notes for cleanup

- The notebook currently contains both the original Tier 1/Tier 2 adoption workflow and the new econ adopter workflow.
- The refactor is working as intended: the econ adopter column can be generated and aggregated.
- The results are still equivalent where expected, except for the known `lessWTP == -0.0` edge case.

## Suggested next steps

1. Remove or comment out the temporary diagnostic cells used for equivalence checking.
2. Replace the legacy adoption-rate block with the new econ adopter block where desired.
3. Keep the existing Tier 1/Tier 2 logic if you want side-by-side comparison plots.
4. Optionally add a short notebook note explaining the `adopter_tiers=[True]` requirement for boolean econ adopter aggregation.
