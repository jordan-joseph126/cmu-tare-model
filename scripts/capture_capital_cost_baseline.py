"""Capture a byte-for-byte baseline of the capital-cost pipeline outputs.

This script is the functional-equivalence oracle for the capital-cost refactor
(Session B, 07 July 2026). Later refactor tasks re-run it and diff the produced
parquet files against the committed baseline to prove that a change did not move
any value (Tasks 2 and 3) or to quantify exactly which values moved (Task 4).

Why a synthetic input instead of the real EUSS stock
----------------------------------------------------
The ~331k-home EUSS ResStock extract lives on Zenodo and is git-ignored
(see cmu_tare_model/data/), so it is not available in this environment. The
REMDB v4 cost table IS available, so the oracle pairs the REAL cost regression
with a DETERMINISTIC synthetic home set that is hand-built to exercise every
branch the refactor touches:

  - every replacement row_id (gas/propane/oil furnace, electric baseboard,
    ducted and non-ducted ASHP) and every cooling row_id (room AC, central AC,
    ducted and non-ducted heat pump);
  - efficiencies below, at, and above each efficiency floor;
  - capacities below the lower bound within tolerance, below beyond tolerance,
    above the upper bound within tolerance, above beyond tolerance, and well
    inside the bounds;
  - NaN capacity and NaN efficiency rows;
  - rows that resolve to row_id 'unknown'.

For a pure refactor, running the SAME input through the old and the new code and
getting byte-identical output is a sound equivalence proof for that input. The
synthetic set is built to cover the code paths, so equivalence on it is strong
evidence of equivalence on the real stock. The `_original`, pm1, and pm2 columns
are captured so the efficiency-floor and capacity-clamping semantics are pinned.

Outputs (under baseline_capture/, row index preserved):
  - homes_input.parquet            the synthetic input frame
  - <combo>_main.parquet           add_remdb_metrics df_main per (end_use, type)
  - <combo>_detailed.parquet       add_remdb_metrics df_detailed per combo
  - v4_regression_costs.parquet    pure v4 installed cost per combo x percentile
  - manifest.json                  row count, column lists, and file hashes

Run from the project root:
    python scripts/capture_capital_cost_baseline.py
"""

import hashlib
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import PROJECT_ROOT
from cmu_tare_model.constants import (
    CAPACITY_BOUND_CLAMPING_TOLERANCE,
    EFFICIENCY_FLOORS_PM2,
)
from cmu_tare_model.utils.remdb_v4_installed_cost_utils import (
    add_remdb_metrics,
    load_remdb_v4_data,
)

# Directory that holds the committed baseline parquet files.
BASELINE_DIR = os.path.join(PROJECT_ROOT, "baseline_capture")

# The (end_use, metric_type) combinations the capital-cost pipeline runs.
# Cooling has no upgrade metric -- a heat pump upgrade covers both loads, so the
# cooling upgrade is never priced separately (see _assign_upgrade_row_id).
PIPELINE_COMBOS: List[Tuple[str, str]] = [
    ("heating", "replacement"),
    ("heating", "upgrade"),
    ("cooling", "replacement"),
]

# REMDB percentiles that map to the v4LOW / v4MID / v4HIGH cost scenarios.
PERCENTILES: List[str] = ["low", "mid", "high"]


def build_synthetic_homes() -> pd.DataFrame:
    """Build a deterministic home set that exercises every refactor code path.

    The rows are constructed by hand (not sampled) so the baseline is stable
    across machines and Python versions. Capacities are chosen relative to the
    REMDB training bounds so that the capacity-clamping branch (Task 4) sees
    homes on both sides of both bounds, within and beyond tolerance.

    Returns:
        DataFrame with the columns add_remdb_metrics() reads for heating and
        cooling, plus an ``euss_home_id`` label carried through for traceability.
    """
    rows: List[Dict[str, object]] = []

    # Step 1 -- gas furnace homes across the AFUE floor (floor = 0.80).
    # 60/68/76% are below the floor; 80% is at it; 92.5% is above it.
    for afue in ["60% AFUE", "68% AFUE", "76% AFUE", "80% AFUE", "92.5% AFUE"]:
        rows.append(
            _heating_row(
                base_fuel="Natural Gas",
                heating_type="Natural Gas Fuel Furnace",
                has_ducts="Yes",
                cooling_type="Central AC",
                heating_eff=afue,
                cooling_eff="SEER 13, 11.7 EER",
                # 80,000 BTU/h furnace: BTU/hr metric, no capacity bounds mapped.
                heating_kbtu=80.0,
                cooling_kbtu=36.0,
            )
        )

    # Step 2 -- propane and fuel-oil furnaces (both proxy to gas furnace row_id).
    for fuel, htype in [("Propane", "Propane Fuel Furnace"),
                        ("Fuel Oil", "Fuel Oil Fuel Furnace")]:
        rows.append(
            _heating_row(
                base_fuel=fuel,
                heating_type=htype,
                has_ducts="Yes",
                cooling_type="Central AC",
                heating_eff="80% AFUE",
                cooling_eff="SEER 14, 12 EER",
                heating_kbtu=72.0,
                cooling_kbtu=30.0,
            )
        )

    # Step 3 -- electric baseboard (electric_baseboard_default row_id).
    # This row_id has no pm2_metric, so pm2 resolves to 0.0 (no floor applies).
    rows.append(
        _heating_row(
            base_fuel="Electricity",
            heating_type="Electricity Electric Baseboard",
            has_ducts="No",
            cooling_type="Room AC",
            heating_eff="100% AFUE",
            cooling_eff="SEER 11, 9.8 EER",
            heating_kbtu=30.0,
            cooling_kbtu=12.0,
        )
    )

    # Step 4 -- ASHP homes across the SEER floor (floor = 15.0), ducted and not.
    # SEER 8/10/13 are below the floor; SEER 15 is at it. Capacities are placed
    # relative to the ducted ASHP bounds (1.5 - 5.0 tons -> 18 - 60 kBtu/h).
    ashp_specs = [
        ("SEER 8, 6.8 HSPF", "Yes", 36.0),     # below floor, inside bounds
        ("SEER 10, 7.7 HSPF", "Yes", 16.2),    # 1.35 tons: on the 10% tol edge
        ("SEER 13, 8.2 HSPF", "No", 66.0),     # 5.5 tons: 10% above 5.0 (clamp down)
        ("SEER 15, 8.8 HSPF", "Yes", 12.0),    # 1.0 ton: 33% below 1.5 (beyond)
        ("SEER 21, 10 HSPF", "No", 84.0),      # 7.0 tons: 40% above 5.0 (beyond)
        ("SEER 16, 9 HSPF", "Yes", 17.1),      # 1.425 tons: 5% below 1.5 (clamp up)
        ("SEER 16, 9 HSPF", "Yes", 63.0),      # 5.25 tons: 5% above 5.0 (clamp down)
    ]
    for seer, ducts, kbtu in ashp_specs:
        rows.append(
            _heating_row(
                base_fuel="Electricity",
                heating_type="Electricity ASHP",
                has_ducts=ducts,
                cooling_type="Heat Pump",
                heating_eff=seer,
                cooling_eff=seer,
                heating_kbtu=kbtu,
                cooling_kbtu=kbtu,
            )
        )

    # Step 5 -- NaN capacity and NaN efficiency rows (NaN must propagate).
    rows.append(
        _heating_row(
            base_fuel="Natural Gas",
            heating_type="Natural Gas Fuel Furnace",
            has_ducts="Yes",
            cooling_type="Central AC",
            heating_eff=np.nan,
            cooling_eff="SEER 13, 11.7 EER",
            heating_kbtu=np.nan,
            cooling_kbtu=np.nan,
        )
    )

    # Step 6 -- a row that resolves to row_id 'unknown' (no matching condition).
    rows.append(
        _heating_row(
            base_fuel="Wood",
            heating_type="Other Fuel Furnace",
            has_ducts="Unknown",
            cooling_type="Evaporative Cooler",
            heating_eff="Other",
            cooling_eff="Other",
            heating_kbtu=50.0,
            cooling_kbtu=20.0,
        )
    )

    df = pd.DataFrame(rows)
    df["euss_home_id"] = range(len(df))

    # Upgrade efficiency: set by the measure package, not the housing stock.
    # A fixed value keeps the upgrade path deterministic (real MP3/MP4 differ,
    # but the refactor is efficiency-independent for the row_id mapping).
    df["upgrade_hvac_heating_efficiency"] = "SEER 18, 10 HSPF"
    return df


def _heating_row(
    base_fuel: str,
    heating_type: str,
    has_ducts: str,
    cooling_type: str,
    heating_eff: object,
    cooling_eff: object,
    heating_kbtu: object,
    cooling_kbtu: object,
) -> Dict[str, object]:
    """Assemble one home record with the columns the pipeline reads.

    Args:
        base_fuel: Baseline heating fuel (drives replacement row_id).
        heating_type: Heating technology label.
        has_ducts: 'Yes' or 'No' (drives ducted vs. non-ducted row_id).
        cooling_type: Cooling technology label (drives cooling row_id).
        heating_eff: Heating efficiency string or NaN.
        cooling_eff: Cooling efficiency string or NaN.
        heating_kbtu: Heating capacity in kBtu/h, or NaN.
        cooling_kbtu: Cooling capacity in kBtu/h, or NaN.

    Returns:
        A dict representing one row of the synthetic input frame.
    """
    return {
        "base_heating_fuel": base_fuel,
        "heating_type": heating_type,
        "hvac_has_ducts": has_ducts,
        "hvac_cooling_type": cooling_type,
        "hvac_heating_efficiency": heating_eff,
        "hvac_cooling_efficiency": cooling_eff,
        "size_heating_system_primary_k_btu_h": heating_kbtu,
        "size_cooling_system_primary_k_btu_h": cooling_kbtu,
    }


def compute_v4_regression_cost(
    df_detailed: pd.DataFrame, end_use: str, metric_type: str, percentile: str
) -> pd.Series:
    """Reproduce the REMDB v4 installed-cost regression on captured parameters.

    Mirrors _calculate_v4_replacement / _calculate_v4_upgrade exactly:
        material_price = pm1 * pm1_coef + pm2 * pm2_coef + intercept
        installed_cost = material_price * multiplier_retrofit + adder_retrofit

    Capturing this here (rather than calling the cost module) keeps the oracle
    independent of the validation-framework masking, which needs income data
    that is not available offline. The regression itself is what Tasks 2 and 4
    can move, so it is what the baseline pins.

    Args:
        df_detailed: Detailed frame from add_remdb_metrics for this combo.
        end_use: 'heating' or 'cooling'.
        metric_type: 'replacement' or 'upgrade'.
        percentile: 'low', 'mid', or 'high'.

    Returns:
        Series of installed costs (NaN where any input parameter is NaN).
    """
    prefix = f"{end_use}_{metric_type}_"
    pm1 = df_detailed[f"{prefix}pm1_euss"]
    pm2 = df_detailed[f"{prefix}pm2_euss"]
    material_price = (
        pm1 * df_detailed[f"{prefix}pm1_coef_{percentile}"]
        + pm2 * df_detailed[f"{prefix}pm2_coef_{percentile}"]
        + df_detailed[f"{prefix}intercept_{percentile}"]
    )
    installed_cost = (
        material_price * df_detailed[f"{prefix}multiplier_retrofit"]
        + df_detailed[f"{prefix}adder_retrofit"]
    )
    return installed_cost


def _hash_frame(df: pd.DataFrame) -> str:
    """Return a stable content hash of a DataFrame including NaN positions.

    Args:
        df: DataFrame to fingerprint.

    Returns:
        Hex SHA-256 digest of the frame's values and column order.
    """
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, df.columns)).encode())
    hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return hasher.hexdigest()


def main() -> None:
    """Build the synthetic input, run the pipeline, and write the baseline."""
    os.makedirs(BASELINE_DIR, exist_ok=True)

    # Step 1 -- load the real REMDB cost table and the synthetic homes.
    remdb_costs = load_remdb_v4_data()
    df_homes = build_synthetic_homes()

    manifest: Dict[str, object] = {
        "n_homes": int(len(df_homes)),
        "efficiency_floors": EFFICIENCY_FLOORS_PM2,
        "capacity_clamp_tolerance": CAPACITY_BOUND_CLAMPING_TOLERANCE,
        "files": {},
    }

    # Step 2 -- persist the input frame so the oracle is reproducible.
    homes_path = os.path.join(BASELINE_DIR, "homes_input.parquet")
    df_homes.to_parquet(homes_path)
    manifest["files"]["homes_input.parquet"] = _hash_frame(df_homes)

    # Step 3 -- run add_remdb_metrics for each combo and capture both frames.
    v4_costs = pd.DataFrame(index=df_homes.index)
    v4_costs["euss_home_id"] = df_homes["euss_home_id"]

    for end_use, metric_type in PIPELINE_COMBOS:
        df_main, df_detailed = add_remdb_metrics(
            df=df_homes,
            remdb_v4_costs=remdb_costs,
            end_use=end_use,
            metric_type=metric_type,
            percentile="mid",
            verbose=False,
        )
        combo = f"{end_use}_{metric_type}"

        main_path = os.path.join(BASELINE_DIR, f"{combo}_main.parquet")
        detailed_path = os.path.join(BASELINE_DIR, f"{combo}_detailed.parquet")
        df_main.to_parquet(main_path)
        df_detailed.to_parquet(detailed_path)
        manifest["files"][f"{combo}_main.parquet"] = _hash_frame(df_main)
        manifest["files"][f"{combo}_detailed.parquet"] = _hash_frame(df_detailed)

        # Step 4 -- capture the v4 regression cost at every percentile. The
        # 'mid' detailed frame only carries mid coefficients, so re-run
        # add_remdb_metrics per percentile to get low/high coefficients too.
        for pct in PERCENTILES:
            _, df_detailed_pct = add_remdb_metrics(
                df=df_homes,
                remdb_v4_costs=remdb_costs,
                end_use=end_use,
                metric_type=metric_type,
                percentile=pct,
                verbose=False,
            )
            cost = compute_v4_regression_cost(
                df_detailed_pct, end_use, metric_type, pct
            )
            v4_costs[f"{combo}_installed_cost_{pct}"] = cost

    costs_path = os.path.join(BASELINE_DIR, "v4_regression_costs.parquet")
    v4_costs.to_parquet(costs_path)
    manifest["files"]["v4_regression_costs.parquet"] = _hash_frame(v4_costs)

    # Step 5 -- write the manifest last so its presence signals a complete run.
    manifest_path = os.path.join(BASELINE_DIR, "manifest.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"""\
[OK] Capital-cost baseline captured
     Homes: {manifest['n_homes']} | Combos: {len(PIPELINE_COMBOS)}
     Files: {len(manifest['files'])} written to {BASELINE_DIR}""")
    for name, digest in manifest["files"].items():
        print(f"     {name}: {digest[:16]}...")


if __name__ == "__main__":
    main()
