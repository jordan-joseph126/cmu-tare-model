"""Small helper functions shared by the county-level KPI visualizations.

``pct_change`` and ``make_symmetric_norm`` support the county choropleths in
``visualize_geospatial_data.py``; ``print_column_summary`` prints the plain-text
county-level summary the main notebook shows above each of those maps. All
three started as inline notebook helpers with no module home, flagged by their
own comment for a move once they saw wider use ("these three helpers have no
module home yet ... move them into an adoption_kpis viz-helper module"). Moved
here 2 Sep 2026 during the notebook/codebase cleanup session.
"""

from typing import Dict, List

import pandas as pd
from matplotlib.colors import Normalize


def pct_change(new: pd.Series, old: pd.Series) -> pd.Series:
    """Per-element percent change (new - old) / old * 100.

    Returns NaN wherever old <= 0 (invalid baseline) or either input is NaN,
    so homes with zero or negative baseline cost are excluded rather than
    producing infinite or misleading values.

    Args:
        new: Post-retrofit values.
        old: Baseline values (the denominator).

    Returns:
        Per-element percent change, aligned to the input index.
    """
    old_safe = old.where(old > 0, other=float("nan"))
    return (new - old_safe) / old_safe * 100


def make_symmetric_norm(
    values: pd.Series,
    center: float = 0.0,
    low_q: float = 0.02,
    high_q: float = 0.98,
) -> Normalize:
    """Symmetric Normalize centered at ``center``.

    Clips to the [low_q, high_q] percentiles before computing the symmetric
    deviation, so a single extreme county cannot compress the colormap.

    Args:
        values: Values the colormap will cover (may include NaN).
        center: The value the color scale is centered on. Default 0.0.
        low_q: Lower quantile used to bound the deviation. Default 0.02.
        high_q: Upper quantile used to bound the deviation. Default 0.98.

    Returns:
        A matplotlib Normalize instance spanning [center - dev, center + dev].
    """
    v = values.dropna()
    q_low = v.quantile(low_q)
    q_high = v.quantile(high_q)
    dev = max(abs(q_low - center), abs(q_high - center))
    return Normalize(vmin=center - dev, vmax=center + dev)


def print_column_summary(
    results: Dict[int, pd.DataFrame],
    column: str,
    label: str,
    selected_mps: List[int],
    mp_subtitles: Dict[int, str],
    positive_direction: str = "increase",
) -> None:
    """Print per-MP min/median/mean/max summary for a county-level column.

    Args:
        results: ``{mp: DataFrame}`` with one row per county, carrying
            ``column``.
        column: Column name to summarize.
        label: Not used in the printed output; kept so call sites that already
            pass a display label (unused since this helper's original inline
            notebook version) do not need to change.
        selected_mps: Measure-package numbers to summarize, in order.
        mp_subtitles: Not used in the printed output; kept for the same reason
            as ``label``.
        positive_direction: Which sign counts as "positive" in the printed
            share -- ``'increase'`` counts values > 0, anything else counts
            values < 0. Also printed verbatim in the summary line (e.g.
            ``'HP saves money (< 0)'``).
    """
    print(f"\n--- Summary: {column} ---")
    for mp in selected_mps:
        _v = results[mp][column].dropna()
        if positive_direction == "increase":
            _pct = (_v > 0).mean() * 100
        else:
            _pct = (_v < 0).mean() * 100
        print(f"  MP{mp}: n={len(_v):,} counties | "
              f"min={_v.min():.1f} | med={_v.median():.1f} | "
              f"mean={_v.mean():.1f} | max={_v.max():.1f} | "
              f"{_pct:.1f}% of counties {positive_direction}")
