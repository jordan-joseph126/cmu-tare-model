from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from cmu_tare_model.constants import LOCAL_REBATE_AMOUNT
from cmu_tare_model.utils.column_names import create_adoption_col, create_npv_case_col
from cmu_tare_model.utils.modeling_params import define_scenario_params


# Rebate scenario: (label, color, marker), in legend order.
# The color is used when dots are not split by heating type.
REBATE_STYLES = {
    'unsub': ('Unsubsidized', '#2a78d6', 'o'),
    'sub': ('2024 Rebate', '#eb6834', '^'),
    'sub_june2026': ('2026 Rebate w/o fuel switching', '#1baf7a', 's'),
    'sub_local': (f'Local rebates (${LOCAL_REBATE_AMOUNT:,})', '#eda100', 'D'),
}

# Label for the dots that cover every sample home
ALL_HEATING_TYPES = 'All heating types'

# One fixed color per heating type, the same as the NPV bar charts
HEATING_COLORS = {
    ALL_HEATING_TYPES: '#000000',
    'Natural Gas Fuel Furnace': '#2a78d6',
    'Natural Gas Fuel Boiler': '#eb6834',
    'Propane Fuel Furnace': '#1baf7a',
    'Electricity Electric Furnace': '#eda100',
    'Electricity Baseboard': '#e87ba4',
}

# Replacement cost offset scope: figure title
OFFSET_SCOPES = {
    'heatingLCC_coolingSavings': 'Heating Replacement Cost Offset',
    'heatingSavings_coolingLCC': 'Cooling Replacement Cost Offset',
    'heatingLCC_coolingLCC': 'Heating + Cooling Replacement Cost Offset',
}

# Panel title for each measure package
PANEL_TITLES = {
    3: 'MP3: standard heat pump',
    4: 'MP4: high-efficiency heat pump',
    5: 'MP5: dual fuel heat pump',
}

# Font sizes for every label in the figure
LABEL_SIZE = 20
TITLE_SIZE = 26


def plot_adoption_vs_npv_panels(
    dataframes_by_mp: Dict[int, pd.DataFrame],
    scope: str,
    discount_rate: str,
    policy: str,
    weight_dict: Optional[Dict[int, float]] = None,
    by_heating_type: bool = False,
) -> pd.DataFrame:
    """Plot adoption against average private NPV, one panel per measure package.

    Args:
        dataframes_by_mp: The TARE table for each measure package, e.g.
            {3: df_mp3, 4: df_mp4}.
        scope: Replacement cost offset scope, one of OFFSET_SCOPES.
        discount_rate: Discount rate key, e.g. 'fixed_base'.
        policy: Policy scenario passed to define_scenario_params.
        weight_dict: Tax parcels per building id. None weights every sample
            home the same.
        by_heating_type: False draws one dot per rebate scenario, colored by
            rebate. True draws one dot per heating type and rebate scenario:
            color is the heating type, shape is the rebate.

    Returns:
        The average NPV and adoption rate for each dot.

    Raises:
        ValueError: If scope is not one of OFFSET_SCOPES.
    """
    if scope not in OFFSET_SCOPES:
        raise ValueError(f"scope must be one of {list(OFFSET_SCOPES)}, got {scope!r}")

    # Panels share both axes, so dots line up across packages
    fig, axes = plt.subplots(
        1, len(dataframes_by_mp), figsize=(18, 8), sharex=True, sharey=True
    )
    rows = []

    for ax, (mp, df_tare) in zip(axes, dataframes_by_mp.items()):
        # Step 1 -- sample homes and their weights
        df_sample = df_tare.loc[df_tare['include_sample']].copy()
        if weight_dict is None:
            df_sample['weight'] = 1.0
        else:
            # Keep only sample homes matched to tax parcels
            df_sample['weight'] = df_sample.index.map(weight_dict)
            df_sample = df_sample.dropna(subset=['weight'])

        # Step 2 -- the groups of homes that get their own dot
        groups = {ALL_HEATING_TYPES: df_sample}
        if by_heating_type:
            for heating_type in HEATING_COLORS:
                if heating_type != ALL_HEATING_TYPES:
                    groups[heating_type] = df_sample[
                        df_sample['heating_type'] == heating_type
                    ]
            # Heating types with no color get no dots, but count in the black ones
            other_types = set(df_sample['heating_type'].dropna()) - set(groups)
            if other_types:
                print(f"MP{mp}, not shown as their own dots: {sorted(other_types)}")

        # Step 3 -- one dot per group and rebate scenario
        scenario_prefix = define_scenario_params(mp, policy)[0]
        for rebate, (label, rebate_color, marker) in REBATE_STYLES.items():
            npv_case = f'{scope}_{rebate}'
            npv_col = create_npv_case_col(
                scenario_prefix, npv_case, f'_{discount_rate}'
            )
            adopter_col = create_adoption_col(
                scenario_prefix, npv_case, f'_{discount_rate}'
            )

            for heating_type, df_group in groups.items():
                df_valid = df_group.dropna(subset=[npv_col])
                if df_valid.empty:
                    continue
                mean_npv = np.average(df_valid[npv_col], weights=df_valid['weight'])
                adoption_pct = 100 * np.average(
                    df_valid[adopter_col], weights=df_valid['weight']
                )

                if by_heating_type:
                    color = HEATING_COLORS[heating_type]
                    dot_marker = marker
                else:
                    color = rebate_color
                    dot_marker = 'o'
                ax.scatter(mean_npv, adoption_pct, s=110, color=color,
                           marker=dot_marker, edgecolor='white', linewidth=1.5,
                           zorder=3)
                rows.append({
                    'mp': mp,
                    'heating_type': heating_type,
                    'rebate_scenario': label,
                    'mean_npv': mean_npv,
                    'adoption_pct': adoption_pct,
                    'weighted_count': df_valid['weight'].sum(),
                })

        # Step 4 -- panel axes
        # Font sizes are set here, so earlier notebook cells can't make them too big
        # Homes to the right of this line gain money on average
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(PANEL_TITLES[mp], fontsize=TITLE_SIZE)
        ax.set_xlabel('Average private NPV (2025 USD)', fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=LABEL_SIZE)
        ax.grid(color='lightgray', linewidth=0.5)
        ax.set_axisbelow(True)

    # Step 5 -- shared axes and labels
    # A little past 0 and 100, so dots on the edge are not cut in half
    axes[0].set_ylim(-3, 103)
    axes[0].set_yticks(range(0, 101, 20))
    axes[0].yaxis.set_major_formatter('{x:.0f}%')
    axes[0].margins(x=0.1)
    # At most about 5 dollar labels, so they don't run into each other
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=5))
    # Write -$4,000, not $-4,000
    axes[0].xaxis.set_major_formatter(
        lambda x, _: f'-${-x:,.0f}' if x < 0 else f'${x:,.0f}'
    )
    
    # Two lines, so the label is never taller than the panels
    if weight_dict is None:
        axes[0].set_ylabel('Economic adoption\n(% of homes)', fontsize=LABEL_SIZE)
    else:
        axes[0].set_ylabel(
            'Economic adoption\n(% of tax parcels)', fontsize=LABEL_SIZE
        )

    fig.suptitle(
        f'{OFFSET_SCOPES[scope]}\n'
        'Average Private NPV ($) and Heat Pump Adoption (%)',
        fontsize=TITLE_SIZE,
    )

       # Step 6 -- legends below the panels
       # Step 6 -- legends below the panels
    legend_style = {
        'fontsize': LABEL_SIZE,
        'title_fontsize': LABEL_SIZE,
        'ncol': 2,
    }
    # Legend markers grow with the text
    marker_size = 0.6 * LABEL_SIZE
    if by_heating_type:
        heating_handles = [
            Line2D([], [], marker='o', linestyle='None', markersize=marker_size,
                   color=color, label=heating_type)
            for heating_type, color in HEATING_COLORS.items()
        ]
        rebate_handles = [
            Line2D([], [], marker=marker, linestyle='None',
                   markersize=marker_size, color='gray', label=label)
            for label, _color, marker in REBATE_STYLES.values()
        ]
        # Side by side; placed once their widths are known (below)
        legends = [
            fig.legend(handles=heating_handles, title='Baseline heating',
                       loc='lower left', **legend_style),
            fig.legend(handles=rebate_handles, title='Rebate scenario',
                       loc='lower left', **legend_style),
        ]
    else:
        rebate_handles = [
            Line2D([], [], marker='o', linestyle='None', markersize=marker_size,
                   color=color, label=label)
            for label, color, _marker in REBATE_STYLES.values()
        ]
        legends = [
            fig.legend(handles=rebate_handles, title='Rebate scenario',
                       loc='lower center', **legend_style),
        ]

    # Measure the legends, so the layout fits them at any font size
    renderer = fig.canvas.get_renderer()
    if by_heating_type:
        # Center the two legends as a pair, with a small gap between them
        widths = [
            legend.get_window_extent(renderer).width / fig.bbox.width
            for legend in legends
        ]
        gap = 0.02
        left = (1 - sum(widths) - gap) / 2
        legends[0].set_bbox_to_anchor((left, 0), transform=fig.transFigure)
        legends[1].set_bbox_to_anchor(
            (left + widths[0] + gap, 0), transform=fig.transFigure
        )
    # Leave just enough room at the bottom for the tallest legend
    legend_height = max(
        legend.get_window_extent(renderer).height for legend in legends
    ) / fig.bbox.height
    fig.tight_layout(rect=(0, legend_height + 0.02, 1, 1))
    plt.show()


    return pd.DataFrame(rows)
