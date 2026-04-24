"""
State-level thermal COP and break-even COP computation from EUSS data.

Provides two public functions:
- ``compute_thermal_cop``: aggregates EUSS heating data to state-level
  heat-pump thermal COP and baseline furnace AFUE.
- ``compute_breakeven_cop``: computes the COP threshold at which a heat
  pump matches a gas furnace's annual fuel cost at a given AFUE.

Location: cmu_tare_model/adoption_kpis/thermal_cop.py
"""

from typing import Optional

import numpy as np
import pandas as pd

from cmu_tare_model.adoption_kpis.data_loading import (
    CLIMATE_ZONE_COL,
    COUNTY_COL,
    HEATING_LOAD_COL,
    HP_BACKUP_ELEC_COL,
    HP_FANS_PUMPS_COL,
    KBTU_PER_KWH,
)


# ============================================================================
# CLIMATE ZONE BENCHMARK RANGES
# ============================================================================

COP_BENCHMARK_RANGES: dict[str, dict] = {
    "1-3": {"mp3": (2.4, 3.2), "mp4": (3.0, 4.2), "label": "Warm (CZ 1-3)"},
    "4-5": {"mp3": (2.0, 2.8), "mp4": (2.5, 3.5), "label": "Mixed (CZ 4-5)"},
    "6-7": {"mp3": (1.6, 2.4), "mp4": (2.0, 3.0), "label": "Cold (CZ 6-7)"},
}
"""Literature-derived COP benchmark ranges by IECC climate zone group.

mp3 = ASHP (SEER 15 / HSPF 9); mp4 = ASHP (SEER 24 / HSPF 13).
"""

# COP plausibility bounds used in verbose diagnostics
_COP_SUSPECT_LOW: float = 1.5
_COP_SUSPECT_HIGH: float = 5.0

# AFUE plausibility bounds
_AFUE_SUSPECT_LOW: float = 0.50
_AFUE_SUSPECT_HIGH: float = 1.00


# ============================================================================
# HELPERS
# ============================================================================

def iecc_to_cz_group(iecc_zone: str) -> str:
    """Map an IECC climate zone string to the three-group benchmark label.

    Args:
        iecc_zone: IECC 2004 zone string (e.g., ``'4A'``, ``'5B'``, ``'7A'``).
            Accepts ``None`` or ``float('nan')`` and returns ``'unknown'``.

    Returns:
        One of ``'1-3'``, ``'4-5'``, ``'6-7'``, or ``'unknown'``.

    Raises:
        ValueError: If the numeric prefix is outside 1–7.
    """
    if iecc_zone is None or (isinstance(iecc_zone, float) and np.isnan(iecc_zone)):
        return "unknown"
    try:
        zone_num = int(str(iecc_zone)[0])
    except (ValueError, IndexError):
        return "unknown"
    if 1 <= zone_num <= 3:
        return "1-3"
    elif 4 <= zone_num <= 5:
        return "4-5"
    elif 6 <= zone_num <= 7:
        return "6-7"
    else:
        raise ValueError(
            f"IECC zone numeric prefix {zone_num} outside 1–7: '{iecc_zone}'"
        )


# ============================================================================
# PUBLIC FUNCTIONS
# ============================================================================

def compute_thermal_cop(
    df_baseline: pd.DataFrame,
    df_upgrade: pd.DataFrame,
    group_cols: list[str] = None,
    fuel_filter: str = "Natural Gas",
    require_baseline_heating: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Compute state-level thermal COP and baseline furnace AFUE from EUSS data.

    COP = Σ(Q_delivered) / Σ(E_hp + E_backup + E_fans_pumps) per group.
    AFUE = Σ(Q_delivered) / Σ(F_gas) per group.

    The fan/pump electricity column
    (``out.electricity.heating_fans_pumps.energy_consumption.kwh``) is
    always included in the denominator because the numerator
    (EUSS heating load) includes the heat deposited by fan motors.
    Omitting it inflates COP to 6.0+.

    Args:
        df_baseline: EUSS baseline DataFrame indexed by ``bldg_id``.
        df_upgrade: EUSS upgrade DataFrame indexed by ``bldg_id``,
            already filtered to ``applicability == True``.
        group_cols: Columns to group by.  Accepts any subset of
            ``['state', 'cz_group']``.  Defaults to ``['state']``.
        fuel_filter: Value of ``in.heating_fuel`` to restrict to.
            Pass ``None`` to include all fuel types.
        require_baseline_heating: If ``True`` (default), homes with
            zero baseline heating load are excluded before aggregation.
            Set to ``False`` to reproduce the pre-fix behavior.
        verbose: Print diagnostic summary after aggregation.

    Returns:
        DataFrame with ``group_cols`` as leading columns followed by:

        - ``thermal_cop``: heat delivered per unit electricity consumed
        - ``baseline_afue``: furnace efficiency derived from EUSS data
        - ``Q_upgrade_total_kbtu``: total HP heat delivered (kBtu)
        - ``hp_total_elec_kbtu``: total HP electricity incl. fans (kBtu)
        - ``Q_baseline_total_kbtu``: total baseline heating load (kBtu)
        - ``gas_consumed_total_kbtu``: total baseline gas consumed (kBtu)
        - ``fans_pumps_pct``: fan/pump share of total HP electricity (%)
        - ``home_count``: number of homes in the group

    Raises:
        KeyError: If a required EUSS column is absent from either DataFrame.

    Notes:
        AFUE is derived from EUSS data (Q_delivered / F_gas), never from
        an assumed constant such as 0.80, which would cause double-counting.

    Verification:
        - Std ASHP COP (national, MP3) ≈ 2.02 (±0.02)
        - High-eff COP (national, MP4) ≈ 3.34 (±0.02)
        - Baseline AFUE (national) ≈ 0.76 (±0.02)
    """
    if group_cols is None:
        group_cols = ["state"]

    elec_col = "out.electricity.heating.energy_consumption.kwh"
    bkup_col = HP_BACKUP_ELEC_COL
    fans_col = HP_FANS_PUMPS_COL
    load_col = HEATING_LOAD_COL
    gas_col = "out.natural_gas.heating.energy_consumption.kwh"

    # Validate required columns
    for col in [load_col, gas_col]:
        if col not in df_baseline.columns:
            raise KeyError(f"Missing column '{col}' in baseline DataFrame")
    for col in [load_col, elec_col, bkup_col, fans_col]:
        if col not in df_upgrade.columns:
            raise KeyError(f"Missing column '{col}' in upgrade DataFrame")

    # Build merged frame including all potential grouping columns
    merge_dict: dict[str, pd.Series] = {
        "state": df_baseline["in.state"],
        "heating_fuel": df_baseline["in.heating_fuel"],
        "Q_baseline_kbtu": df_baseline[load_col].fillna(0),
        "gas_consumed_kwh": df_baseline[gas_col].fillna(0),
    }
    if CLIMATE_ZONE_COL in df_baseline.columns:
        merge_dict["cz_group"] = df_baseline[CLIMATE_ZONE_COL].map(iecc_to_cz_group)
    if COUNTY_COL in df_baseline.columns:
        merge_dict["county"] = df_baseline[COUNTY_COL]

    df_merged = pd.DataFrame(merge_dict).join(
        pd.DataFrame({
            "Q_upgrade_kbtu": df_upgrade[load_col].fillna(0),
            "hp_elec_kwh": df_upgrade[elec_col].fillna(0),
            "hp_bkup_elec_kwh": df_upgrade[bkup_col].fillna(0),
            "hp_fans_pumps_kwh": df_upgrade[fans_col].fillna(0),
        }),
        how="inner",
    )

    if verbose:
        print(f"Matched homes (baseline ∩ upgrade): {len(df_merged):,}")

    if fuel_filter is not None:
        n_before = len(df_merged)
        df_merged = df_merged[df_merged["heating_fuel"] == fuel_filter]
        if verbose:
            pct = 100 * len(df_merged) / n_before
            print(f"Filtered to '{fuel_filter}': {len(df_merged):,} / {n_before:,} "
                  f"({pct:.1f}%)")

    if require_baseline_heating:
        n_before = len(df_merged)
        df_merged = df_merged[df_merged["Q_baseline_kbtu"] > 0]
        n_excluded = n_before - len(df_merged)
        if verbose and n_excluded > 0:
            print(f"Excluded {n_excluded:,} homes with zero baseline heating "
                  f"({len(df_merged):,} remaining)")

    # Total HP electricity (incl. backup resistance + fan/pump)
    df_merged["hp_total_elec_kwh"] = (
        df_merged["hp_elec_kwh"]
        + df_merged["hp_bkup_elec_kwh"]
        + df_merged["hp_fans_pumps_kwh"]
    )

    # Convert to kBtu for unit consistency with heating load
    df_merged["hp_total_elec_kbtu"] = df_merged["hp_total_elec_kwh"] * KBTU_PER_KWH
    df_merged["hp_fans_pumps_kbtu"] = df_merged["hp_fans_pumps_kwh"] * KBTU_PER_KWH
    df_merged["gas_consumed_kbtu"] = df_merged["gas_consumed_kwh"] * KBTU_PER_KWH

    grouped = df_merged.groupby(group_cols).agg(
        Q_upgrade_total_kbtu=("Q_upgrade_kbtu", "sum"),
        hp_total_elec_kbtu=("hp_total_elec_kbtu", "sum"),
        hp_fans_pumps_total_kbtu=("hp_fans_pumps_kbtu", "sum"),
        Q_baseline_total_kbtu=("Q_baseline_kbtu", "sum"),
        gas_consumed_total_kbtu=("gas_consumed_kbtu", "sum"),
        home_count=("Q_baseline_kbtu", "size"),
    ).reset_index()

    grouped["thermal_cop"] = np.where(
        grouped["hp_total_elec_kbtu"] > 0,
        grouped["Q_upgrade_total_kbtu"] / grouped["hp_total_elec_kbtu"],
        np.nan,
    )

    grouped["baseline_afue"] = np.where(
        grouped["gas_consumed_total_kbtu"] > 0,
        grouped["Q_baseline_total_kbtu"] / grouped["gas_consumed_total_kbtu"],
        np.nan,
    )

    grouped["fans_pumps_pct"] = np.where(
        grouped["hp_total_elec_kbtu"] > 0,
        grouped["hp_fans_pumps_total_kbtu"] / grouped["hp_total_elec_kbtu"] * 100,
        0,
    )

    if verbose:
        fuel_label = fuel_filter if fuel_filter else "all fuels"
        group_label = " × ".join(group_cols)
        print(f"\n--- Thermal COP Summary ({fuel_label}, by {group_label}) ---")
        print(f"Groups: {len(grouped)}")
        print(f"Mean:   {grouped['thermal_cop'].mean():.2f}")
        print(f"Median: {grouped['thermal_cop'].median():.2f}")
        print(f"Range:  {grouped['thermal_cop'].min():.2f}"
              f" - {grouped['thermal_cop'].max():.2f}")

        suspects = grouped[
            (grouped["thermal_cop"] < _COP_SUSPECT_LOW)
            | (grouped["thermal_cop"] > _COP_SUSPECT_HIGH)
        ]
        if len(suspects) > 0:
            print(f"[WARN] {len(suspects)} group(s) with COP outside "
                  f"[{_COP_SUSPECT_LOW}, {_COP_SUSPECT_HIGH}]:")
            for _, r in suspects.iterrows():
                label = ", ".join(str(r[c]) for c in group_cols)
                print(f"    {label}: {r['thermal_cop']:.2f}")
        else:
            print(f"[OK] All groups within COP range "
                  f"[{_COP_SUSPECT_LOW}, {_COP_SUSPECT_HIGH}]")

        print(f"\n--- Baseline AFUE Summary ({fuel_label}) ---")
        print(f"Mean:   {grouped['baseline_afue'].mean():.2f}")
        print(f"Median: {grouped['baseline_afue'].median():.2f}")
        print(f"Range:  {grouped['baseline_afue'].min():.2f}"
              f" - {grouped['baseline_afue'].max():.2f}")

        afue_suspects = grouped[
            (grouped["baseline_afue"] < _AFUE_SUSPECT_LOW)
            | (grouped["baseline_afue"] > _AFUE_SUSPECT_HIGH)
        ]
        if len(afue_suspects) > 0:
            print(f"[WARN] {len(afue_suspects)} group(s) with AFUE outside "
                  f"[{_AFUE_SUSPECT_LOW}, {_AFUE_SUSPECT_HIGH}]")

        print(f"\n--- Fan/Pump Energy as % of Total HP Electricity ---")
        print(f"Mean:   {grouped['fans_pumps_pct'].mean():.1f}%")
        print(f"Range:  {grouped['fans_pumps_pct'].min():.1f}%"
              f" - {grouped['fans_pumps_pct'].max():.1f}%")

    return grouped.sort_values(group_cols[0]).reset_index(drop=True)


def compute_breakeven_cop(
    df_prices: pd.DataFrame,
    df_cop: pd.DataFrame,
) -> pd.DataFrame:
    """Compute state-level break-even COP for standard AFUE scenarios.

    The break-even COP is the heat pump efficiency at which annual
    operating costs equal those of a gas furnace at a given AFUE.
    Derived by setting bill impact ratio = 1 and solving for COP:

        COP_breakeven = spark_gap × AFUE

    where spark_gap = P_elec / P_gas on a $/MMBTU basis.

    Args:
        df_prices: Output of :func:`~spark_gap.calculate_spark_gap`.
            Must contain columns ``state``, ``spark_gap``,
            and optionally ``state_name``.
        df_cop: Output of :func:`compute_thermal_cop`.
            Must contain ``state`` and ``baseline_afue``.

    Returns:
        DataFrame with one row per state (inner join on ``state``),
        with columns:

        - ``state``: 2-letter state abbreviation
        - ``state_name``: full state name (if present in df_prices)
        - ``spark_gap``: electricity-to-gas price ratio
        - ``baseline_afue``: data-derived furnace efficiency
        - ``breakeven_cop_80``: break-even COP at 80% AFUE
        - ``breakeven_cop_90``: break-even COP at 90% AFUE
        - ``breakeven_cop_95``: break-even COP at 95% AFUE
        - ``breakeven_cop_100``: break-even COP at 100% AFUE (ideal)

    Notes:
        ASSUMPTION: State-level EIA residential prices used as proxy.
        County-level fuel prices are not available in this dataset.
    """
    afue_scenarios = [0.80, 0.90, 0.95, 1.00]

    base_cols = ["state", "spark_gap"]
    if "state_name" in df_prices.columns:
        base_cols = ["state", "state_name", "spark_gap"]
    result = df_prices[base_cols].copy()

    # Merge in data-derived AFUE
    result = result.merge(
        df_cop[["state", "baseline_afue"]],
        on="state",
        how="inner",
    )

    for afue in afue_scenarios:
        col = f"breakeven_cop_{int(afue * 100)}"
        result[col] = (result["spark_gap"] * afue).round(2)

    return result.reset_index(drop=True)
