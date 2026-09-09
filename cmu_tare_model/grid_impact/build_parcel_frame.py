"""Reconstruct Tamar's per-parcel (row-duplicated) TARE frame from real data.

Joins the current CO measure-package export against the tax-parcel-to-
ResStock match mapping so the no-weight peak-load path
(already_weighted=True in compute_peak_load_summary) can be exercised on a
real per-parcel frame instead of synthetic or stale data. See CLAUDE.md,
"TARE Model -- Tamar's Feeder Peak Run" session notes, for why the two
weighting regimes (sampled frame vs. row-duplicated frame) must not be mixed.
"""

from typing import List

import pandas as pd

# Every required column is ELECTRICITY power (peak_electricity_*_kw), ELECTRICITY
# energy (total_electricity_consumption, kWh), or fuel-agnostic thermal load
# (peak_load_*_kbtu_hr). None is the all-fuel site-energy total: the peak-load
# path is electricity only, so no site-energy column belongs in this list.
REQUIRED_EXPORT_PEAK_COLUMN_TEMPLATES: List[str] = [
    "base_peak_electricity_heating_kw",
    "base_peak_electricity_cooling_kw",
    "base_peak_load_heating_kbtu_hr",
    "base_peak_load_cooling_kbtu_hr",
    "base_total_electricity_consumption",
    "mp{mp}_peak_electricity_heating_kw",
    "mp{mp}_peak_electricity_cooling_kw",
    "mp{mp}_peak_load_heating_kbtu_hr",
    "mp{mp}_peak_load_cooling_kbtu_hr",
    "mp{mp}_total_electricity_consumption",
]


def build_parcel_frame(
    df_export: pd.DataFrame,
    df_mapping: pd.DataFrame,
    mp: int,
) -> pd.DataFrame:
    """Build the per-parcel, row-duplicated frame for Tamar's feeder run.

    Joins the current CO household export (one row per modeled ResStock
    building) against the tax-parcel match mapping (one row per real tax
    parcel) so each matched parcel gets its own row, carrying that parcel's
    representative building's TARE values unchanged. This is the frame meant
    to be run with already_weighted=True in compute_peak_load_summary -- the
    row duplication IS the weighting, so the EUSS sample weight must not also
    be applied on top of it.

    Note on the FIPS gotcha: county_fips loses its leading zero on a CSV
    round-trip (Colorado's "08xxx" reads back as the int 8xxx). This function
    keeps the frame in memory and does not write it to disk, so the gotcha
    does not apply here. If a caller writes the returned frame to CSV, reload
    it with dtype={"county_fips": str} and left-pad to 5 characters.

    Args:
        df_export: Current per-measure-package CO household export (e.g.
            tepper_household_mp{mp}_CO_*.csv), with one row per modeled
            building, a bldg_id column, and the base/mp{mp} peak and
            total-electricity-consumption columns.
        df_mapping: The tax-parcel-to-representative-building match mapping
            (PSM_output_buildYear07_09_2026.csv), with tax_parcel_ID and
            representative_ID columns. representative_ID is NaN for
            unmatched parcels.
        mp: Measure-package number (3 or 4) -- selects which mp{mp}_ columns
            are validated.

    Returns:
        DataFrame with one row per matched parcel whose representative
        building is present in df_export: every df_export column carried
        through unchanged, plus tax_parcel_ID and match_count (the number of
        parcels sharing that row's representative building). The weight
        column is preserved but must not be applied downstream --
        already_weighted=True is the intended run mode for this frame.

    Raises:
        TypeError: If df_export or df_mapping is not a DataFrame.
        ValueError: If mp is not a positive integer.
        KeyError: If df_export is missing a required peak or consumption
            column (a stale export or one built on the wrong branch) or
            df_mapping is missing tax_parcel_ID / representative_ID.
    """
    # Step 1 -- validate inputs before any computation.
    if not isinstance(df_export, pd.DataFrame):
        raise TypeError(f"df_export must be a DataFrame, got {type(df_export)!r}")
    if not isinstance(df_mapping, pd.DataFrame):
        raise TypeError(f"df_mapping must be a DataFrame, got {type(df_mapping)!r}")
    if not isinstance(mp, int) or mp <= 0:
        raise ValueError(f"mp must be a positive integer, got {mp!r}")

    required_export_cols = [
        template.format(mp=mp) for template in REQUIRED_EXPORT_PEAK_COLUMN_TEMPLATES
    ] + ["bldg_id", "weight"]
    missing_export = [c for c in required_export_cols if c not in df_export.columns]
    if missing_export:
        raise KeyError(
            f"df_export is missing required column(s): {missing_export}. "
            f"This export is likely stale or built on a branch without the "
            f"current peak-load columns -- rebuild it on the "
            f"joseph-2026-nature-comms-submission branch before "
            f"reconstructing the parcel frame."
        )

    required_mapping_cols = ["tax_parcel_ID", "representative_ID"]
    missing_mapping = [
        c for c in required_mapping_cols if c not in df_mapping.columns
    ]
    if missing_mapping:
        raise KeyError(f"df_mapping is missing required column(s): {missing_mapping}")

    # Step 2 -- drop the unmatched parcels (no representative_ID).
    df_matched = df_mapping.dropna(subset=["representative_ID"]).copy()

    # representative_ID reads from CSV as float64 (the NaN rows upcast the
    # whole column); bldg_id in the export is int64 -- cast to match, or the
    # join below finds nothing.
    df_matched["representative_ID"] = df_matched["representative_ID"].astype(
        "int64"
    )

    # Step 3 -- match_count: parcels sharing one representative building.
    df_matched["match_count"] = df_matched.groupby("representative_ID")[
        "tax_parcel_ID"
    ].transform("size")

    # Step 4 -- flag representative buildings absent from this export before
    # joining. TARE's own masking rules (existing-ASHP exclusion, invalid
    # fuel/tech, no heating fuel) drop some baseline buildings from the
    # modeled output entirely, upstream of Tamar's match -- so some
    # representative_IDs are expected to be missing here. An inner join
    # would drop those parcels silently; warn with the exact count instead.
    export_bldg_ids = set(df_export["bldg_id"])
    matched_rep_ids = set(df_matched["representative_ID"].unique())
    missing_rep_ids = matched_rep_ids - export_bldg_ids
    if missing_rep_ids:
        n_parcels_dropped = int(
            df_matched["representative_ID"].isin(missing_rep_ids).sum()
        )
        print(
            f"  WARNING: {len(missing_rep_ids):,d} of {len(matched_rep_ids):,d} "
            f"representative buildings are not in the MP{mp} export "
            f"(likely excluded upstream by TARE's masking rules) -- "
            f"{n_parcels_dropped:,d} of {len(df_matched):,d} matched parcels "
            f"will be dropped by the inner join below."
        )

    # Step 5 -- inner join: one row per matched parcel whose representative
    # building survives in the export. Weight is preserved but NOT applied --
    # run this frame with already_weighted=True.
    df_parcel = df_matched.merge(
        df_export,
        left_on="representative_ID",
        right_on="bldg_id",
        how="inner",
    )

    return df_parcel
