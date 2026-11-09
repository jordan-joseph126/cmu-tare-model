import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

def plot_NPV_bars(df_tare, npv_col, case_study_weight_dict):


    # Light to dark blue, for values that have an order
    BLUE_RAMP = LinearSegmentedColormap.from_list('blue_ramp', ['#86b6ef', '#0d366b'])
    VINTAGE_RAMP = LinearSegmentedColormap.from_list(
        'vintage_ramp', plt.cm.viridis(np.linspace(0, 0.8, 256))
    )


    # One fixed color per group, the same in every figure
    HEATING_COLORS = {
        'Natural Gas Fuel Furnace': '#2a78d6',
        'Natural Gas Fuel Boiler': '#eb6834',
        'Propane Fuel Furnace': '#1baf7a',
        'Electricity Electric Furnace': '#eda100',
        'Electricity Baseboard': '#e87ba4',
    }
    COOLING_COLORS = {
        'Central AC': '#2a78d6',
        'Room AC': '#eb6834',
        'No AC': '#898781',
    }

    # Oldest to newest, light to dark
    VINTAGES = [
        '<1940', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s', '2000s', '2010s',
    ]


    # One entry per figure. The key is the column that sets the bar colors.
    # 'colors' makes a group legend; 'colormap' makes a color gradient.
    # 'blank_label' names blank values (ResStock's "None" loads as blank).
    LEGEND_OPTIONS = {
        # 'single_color' draws every bar in one color, with no legend.
            'no_legend': {
            'single_color': "#5c5069",
        },
        'heating_type': {
            'title': 'Baseline heating',
            'colors': HEATING_COLORS,
        },
        'cooling_type': {
            'title': 'Baseline cooling',
            'colors': COOLING_COLORS,
            'blank_label': 'No AC',
        },
        'percent_AMI': {
            'title': 'Household income (% of area median)',
            'colormap': BLUE_RAMP,
            'vmin': 0,
            'vmax': 300,
        },
        'vintage': {
            'title': 'Vintage',
            'colormap': VINTAGE_RAMP,
            'vmin': 0,
            'vmax': len(VINTAGES) - 1,
            'order': VINTAGES,
        },

    }

    for color_col, legend in LEGEND_OPTIONS.items():
        # Keep sample homes matched to tax parcels, sorted by NPV
        plot_cols = [npv_col] if 'single_color' in legend else [npv_col, color_col]
        df_plot = df_tare.loc[df_tare['include_sample'], plot_cols].copy()
        df_plot['custom_weight'] = df_plot.index.map(case_study_weight_dict)
        df_plot = df_plot.dropna(subset=['custom_weight']).sort_values(npv_col)
        if 'blank_label' in legend:
            df_plot[color_col] = df_plot[color_col].fillna(legend['blank_label'])

        # Each bar is as wide as the number of tax parcels it represents
        bar_edges = np.concatenate(([0], df_plot['custom_weight'].cumsum()))

        # Where the NPV crosses zero
        parcels_below_zero = df_plot.loc[df_plot[npv_col] < 0, 'custom_weight'].sum()
        pct_at_or_above_zero = 100 * (1 - parcels_below_zero / bar_edges[-1])

        # Turn each home's value into a bar color
        if 'single_color' in legend:
            bar_colors = legend['single_color']
        elif 'colors' in legend:
            bar_colors = df_plot[color_col].map(legend['colors'])
        else:
            values = df_plot[color_col]
            if 'order' in legend:
                # Text groups become their position in the order
                values = values.map(
                    {label: i for i, label in enumerate(legend['order'])}
                )
            norm = plt.Normalize(legend['vmin'], legend['vmax'])
            bar_colors = legend['colormap'](norm(values))


        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(
            bar_edges[:-1],
            df_plot[npv_col],
            width=df_plot['custom_weight'],
            align='edge',
            color=bar_colors,
            linewidth=0,
            antialiased=False,
        )

        # Group legend or color gradient
        if 'colors' in legend:
            legend_items = [
                Patch(color=color, label=label)
                for label, color in legend['colors'].items()
            ]
            ax.legend(
                handles=legend_items,
                title=legend['title'],
                loc='upper left',
                bbox_to_anchor=(1.01, 1),
                fontsize=12,
                title_fontsize=12,
            )
        elif 'colormap' in legend:
            color_scale = plt.cm.ScalarMappable(norm=norm, cmap=legend['colormap'])
            color_bar = fig.colorbar(color_scale, ax=ax, pad=0.02)
            color_bar.set_label(legend['title'], fontsize=12)
            if 'order' in legend:
                color_bar.set_ticks(range(len(legend['order'])), labels=legend['order'])

            color_bar.ax.tick_params(labelsize=12)

        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(parcels_below_zero, color='gray', linestyle='--', linewidth=1)
        ax.annotate(
            f'{pct_at_or_above_zero:.1f}% of modeled parcels have NPV >= $0',
            xy=(parcels_below_zero, 0),
            xytext=(-8, 8),
            textcoords='offset points',
            ha='right',
        )

        ax.set_xlim(0, bar_edges[-1])
        ax.set_xlabel('Tax parcels count')
        ax.set_ylabel('Private NPV (2025 USD)')
        ax.set_title(npv_col, fontsize=9)
        ax.xaxis.set_major_formatter('{x:,.0f}')
        ax.yaxis.set_major_formatter('${x:,.0f}')

        # Matched parcels left off the figure because they have no NPV
        df_no_npv = df_tare.loc[~df_tare['include_sample']].copy()
        df_no_npv['custom_weight'] = df_no_npv.index.map(case_study_weight_dict)
        df_no_npv = df_no_npv.dropna(subset=['custom_weight'])
        parcels_no_npv = df_no_npv['custom_weight'].sum()
        parcels_heat_pump = df_no_npv.loc[
            df_no_npv['cooling_type'] == 'Heat Pump', 'custom_weight'
        ].sum()
        parcels_all = parcels_no_npv + bar_edges[-1]

        fig.text(
            0.01,
            0.01,
            f'Not shown: {parcels_no_npv:,.0f} of {parcels_all:,.0f} matched parcels '
            f'({100 * parcels_no_npv / parcels_all:.1f}%) have no NPV:\n'
            f'{parcels_heat_pump:,.0f} already have a heat pump '
            f'and {parcels_no_npv - parcels_heat_pump:,.0f} have a '
            f'wall or floor furnace.',
            ha='left',
            va='bottom',
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 1))

        plt.show()