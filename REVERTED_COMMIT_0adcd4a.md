# Reverted Commit Documentation

## Overview

On **March 16, 2026**, commit `0adcd4a` was reverted from `main` due to errors introduced by the changes. The branch was reset to the preceding commit `5548f4b` ("Commit post merge and national model run. Moved legacy files to the archive folder.").

A backup branch `backup-0adcd4a-before-revert` was created before the revert, preserving the full commit for future reference or selective cherry-picking.

---

## Revert Details

| Field | Value |
|---|---|
| **Reverted Commit** | `0adcd4aee9b3d7de0dcc5649a8372b60db96729f` |
| **Parent Commit (reverted to)** | `5548f4bf6523a9542b7f4460b09161fceed3bfe9` |
| **Backup Branch** | `backup-0adcd4a-before-revert` |
| **Revert Method** | `git reset --hard 5548f4b` followed by `git push --force` |
| **Date of Revert** | 2026-03-16 |

---

## Reverted Commit Summary

**Commit message:** "Refactor code structure for improved readability and maintainability. Added number of home counts to plots (using 242 weighting factor)"

**Scope:** 8 files changed — +5,296 insertions / -33,415 deletions

### Files Changed

| File | Change |
|---|---|
| `cmu_tare_model/adoption_potential/data_processing/visuals_adoption_potential.py` | 619 lines modified |
| `cmu_tare_model/tare_model_main_v2_2.ipynb` | +2,446 / -33,260 lines |
| `cmu_tare_model/private_impact/CHANGELOG.md` | +829 lines (new file) |
| `cmu_tare_model/private_impact/Integrate_REMDB_v4_Cost_Estimation.md` | +287 lines (new file) |
| `cmu_tare_model/private_impact/REMDB_v4_Refactoring_Documentation.md` | +617 lines (new file) |
| `cmu_tare_model/private_impact/Private_Capital_Cost_Estimation_Documentation.md` | +653 lines (new file) |
| 2 PNG plot images | Negligible byte-level changes |

---

## Detailed Description of Changes in the Reverted Commit

### 1. Adoption Visualization Overhaul (`visuals_adoption_potential.py`)

- Added a `DWELLING_UNIT_WEIGHT = 242` constant so raw EUSS microdata row counts can be converted to actual home counts.
- Introduced **count-based adoption visualization** with two new functions:
  - `create_multiIndex_adoption_counts_df` — creates a multi-index DataFrame showing adoption home counts (not percentages) by LMI/MUI classification and fuel type.
  - `subplot_grid_adoption_vBar_counts` — renders subplot grids with absolute home counts on the y-axis.
- Made font sizes, bar widths, and legend fonts **scale dynamically** based on the number of subplot panels (`n_panels` parameter), so charts look proportional regardless of how many panels are displayed.
  - Scaling formula: `(3 / n_panels) ** 0.25` — yields ~1.32× at 1 panel, 1.0× at 3, ~0.88× at 5.
  - Bar width scaling: `0.35 * (3 / n_panels) ** 0.12` — subtle width adjustment.
- Added an `n_panels` parameter to `plot_adoption_rate_bar`.
- Legend font size also scales dynamically via the same formula.

### 2. Main Notebook Slimmed Down (`tare_model_main_v2_2.ipynb`)

- Massive net reduction (~30,800 lines removed).
- Likely cleared old output cells and removed duplicated/unused visualization code that was moved into the shared Python module.
- Net change: +2,446 insertions / -33,260 deletions.

### 3. New Documentation (4 new markdown files under `cmu_tare_model/private_impact/`)

- **`CHANGELOG.md`** (+829 lines) — Detailed changelog documenting docstring additions, type hints, error handling improvements, and refactoring history for cost calculation modules including:
  - `calculate_enclosure_upgrade_costs.py`
  - `calculate_equipment_installation_costs.py`
  - `calculate_equipment_replacement_costs.py`
  - `calculate_lifetime_fuel_costs.py`
- **`Integrate_REMDB_v4_Cost_Estimation.md`** (+287 lines) — Integration guide for REMDB v4 cost estimation.
- **`REMDB_v4_Refactoring_Documentation.md`** (+617 lines) — Refactoring documentation for the REMDB v4 transition.
- **`Private_Capital_Cost_Estimation_Documentation.md`** (+653 lines) — Capital cost estimation methodology documentation.

### 4. Minor Image Updates

- Two capital cost sensitivity PNG plots were regenerated with negligible byte-level changes:
  - `*_sensitivity_heating_replacement_mp3_Longmont.png` (152,255 → 152,353 bytes)
  - `*_sensitivity_heating_upgrade_mp3_Longmont.png` (154,802 → 154,709 bytes)

---

## Recovery Instructions

To restore the reverted commit's changes:

```bash
# View the backup branch
git log backup-0adcd4a-before-revert --oneline -1

# Cherry-pick the reverted commit onto current branch
git cherry-pick 0adcd4a

# Or check out the backup branch directly
git checkout backup-0adcd4a-before-revert
```

To selectively recover specific files from the reverted commit:

```bash
# Restore a single file from the reverted commit
git checkout 0adcd4a -- path/to/file

# Example: restore only the visualization changes
git checkout 0adcd4a -- cmu_tare_model/adoption_potential/data_processing/visuals_adoption_potential.py

# Example: restore only the documentation files
git checkout 0adcd4a -- cmu_tare_model/private_impact/CHANGELOG.md
git checkout 0adcd4a -- cmu_tare_model/private_impact/Integrate_REMDB_v4_Cost_Estimation.md
git checkout 0adcd4a -- cmu_tare_model/private_impact/REMDB_v4_Refactoring_Documentation.md
git checkout 0adcd4a -- cmu_tare_model/private_impact/Private_Capital_Cost_Estimation_Documentation.md
```
