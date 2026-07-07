import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Dict, Any, Union

from cmu_tare_model.utils.column_names import create_adoption_col

# =========================================================================
# FUNCTIONS: VISUALIZATION USING DATAFRAMES AND SUBPLOTS
# =========================================================================

# Build the economic-adopter column name for a given MP and NPV case.
def build_adoption_scenario_names(
    mp: int,
    npv_case: str,
    cost_scenario: str,
    discount_rate: str,
) -> List[str]:
    """Build the economic-adopter column name for a given MP and NPV case.

    Returns a single-item list so callers that iterate over the result
    (column-existence checks, scenario loops) work without modification.

    Args:
        mp: Measure package number (e.g., 3 or 4).
        npv_case: One of NPV_CASE_CATEGORIES: 'heatingSavings_coolingLCC',
            'heatingLCC_coolingSavings', or 'heatingLCC_coolingLCC'.
        cost_scenario: Retained for caller compatibility; not embedded in the
            output column name (cost-scenario token removed July 2026).
        discount_rate: Short discount rate key (e.g., 'fixed_base').

    Returns:
        Single-item list with the economic-adopter column name.
        Pattern: ref2025_mp{mp}_{npv_case}_econ_adopter_{discount_rate}
    """
    return [
        create_adoption_col(
            scenario_prefix=f'ref2025_mp{mp}_',
            npv_case=npv_case,
            method_suffix=f'_{discount_rate}',
        )
    ]


def create_multiIndex_adoption_df(
        df: pd.DataFrame,
        menu_mp: int,
        npv_case: str,
        cost_scenario: str,
        discount_rate: str,
) -> pd.DataFrame:
    """
    Create a multi-index DataFrame with economic adoption percentages by fuel type
    and income classification.

    For each (fuel, income) group, computes the percentage of applicable homes
    (non-NaN adopter values) that are economic adopters (econ_adopter == 1.0).
    The result is compatible with subplot_grid_adoption_vBar and prepare_plot_data.

    Args:
        df: DataFrame with the economic-adopter column and the columns
            'base_heating_fuel' and 'lmi_or_mui'.
        menu_mp: Measure package identifier.
        npv_case: One of NPV_CASE_CATEGORIES: 'heatingSavings_coolingLCC',
            'heatingLCC_coolingSavings', or 'heatingLCC_coolingLCC'.
        cost_scenario: Retained for caller compatibility; not embedded in the
            output column name (cost-scenario token removed July 2026).
        discount_rate: Short discount rate key (e.g., 'fixed_base').

    Returns:
        DataFrame with MultiIndex columns (adopter_col, 'Economic Adopter') and
        (adopter_col, 'Total Adoption Potential'). Row index is a MultiIndex of
        (base_heating_fuel, lmi_or_mui). Values are percentages 0-100.

    Raises:
        ValueError: If 'lmi_or_mui' is missing or the adopter column is not found.
    """
    if 'lmi_or_mui' not in df.columns:
        raise ValueError(
            "Required column 'lmi_or_mui' not found. "
            "Ensure the DataFrame has been processed with calculate_percent_AMI."
        )

    adoption_col = create_adoption_col(
        scenario_prefix=f'ref2025_mp{menu_mp}_',
        npv_case=npv_case,
        method_suffix=f'_{discount_rate}',
    )

    if adoption_col not in df.columns:
        raise ValueError(
            f"Adopter column not found: '{adoption_col}'. "
            "Ensure economic_adoption_decision has been run for this combination."
        )

    lmi_mui_categories = ['LMI', 'MUI']
    df = df.copy()
    df['lmi_or_mui'] = pd.Categorical(
        df['lmi_or_mui'], categories=lmi_mui_categories, ordered=True
    )

    # Compute adoption rate per (fuel, income) group as a percentage.
    # Only valid homes (non-NaN adopter values) enter the denominator.
    def _pct_adopters(series: pd.Series) -> float:
        valid = series.dropna()
        if len(valid) == 0:
            return 0.0
        return round((valid == 1.0).sum() / len(valid) * 100, 0)

    pct_series = df.groupby(
        ['base_heating_fuel', 'lmi_or_mui'], observed=False
    )[adoption_col].apply(_pct_adopters)

    # Wrap into a DataFrame with multi-index columns.
    percentages_df = pct_series.to_frame()
    percentages_df.columns = pd.MultiIndex.from_tuples(
        [(adoption_col, 'Economic Adopter')]
    )

    # 'Total Adoption Potential' mirrors 'Economic Adopter' in the single-tier
    # architecture. It exists so dotplot and print helpers that look up
    # 'Total Adoption Potential' by name continue to work.
    percentages_df[(adoption_col, 'Total Adoption Potential')] = (
        percentages_df[(adoption_col, 'Economic Adopter')]
    )

    # Filter to allowed incumbent fuels.
    allowed_fuels = ['Electricity', 'Fuel Oil', 'Natural Gas', 'Propane']
    fuel_level = 'base_heating_fuel'
    percentages_df = percentages_df[
        percentages_df.index.get_level_values(fuel_level).isin(allowed_fuels)
    ]
    percentages_df.sort_index(level=[fuel_level, 'lmi_or_mui'], inplace=True)

    return percentages_df


def plot_adoption_rate_bar(
    df: pd.DataFrame,
    scenarios: List[str],
    title: str,
    x_label: str,
    y_label: str,
    ax: plt.Axes,
    x_tick_format: str = "lmi_only"  # Updated default to reflect LMI/MUI usage
) -> None:
    """
    Plots stacked bar chart for adoption tiers on the given axes.
    
    Args:
        df: DataFrame with multi-index structure containing adoption data
        scenarios: List of scenario column names (without tier part)
        title: Title for the plot
        x_label: Label for x-axis
        y_label: Label for y-axis
        ax: Matplotlib axes to plot on
        x_tick_format: Format for x-tick labels. Options:
                      "lmi_only" - Show only LMI/MUI classification
                      "fuel_only" - Show only fuel type
                      "combined" - Show "Fuel Type, LMI/MUI"
                      "all" - Show all index levels separated by commas
        
    Returns:
        None. The plot is created on the provided axes.
        
    Raises:
        ValueError: If required columns are not found in the DataFrame
    """
    # Define the color mapping for the economic-adopter bar
    color_mapping = {
        'Economic Adopter': 'steelblue',
    }
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex for columns")
    
    # ========== SORT BY ECONOMIC ADOPTER RATE (DESCENDING) ==========
    try:
        sort_cols = [
            (col, 'Economic Adopter')
            for col in df.columns.get_level_values(0).unique()
            if (col, 'Economic Adopter') in df.columns
        ]
        if sort_cols:
            df = df.sort_values(sort_cols[0], ascending=False)
    except Exception:
        pass
    # ==================== END OF SORT =================================

    # Filter the DataFrame to only include the Economic Adopter column
    tier_columns = ['Economic Adopter']
    available_columns = df.columns.get_level_values(1).unique()
    
    if not any(tier in available_columns for tier in tier_columns):
        raise ValueError(f"No 'Economic Adopter' column found. Available columns: {available_columns}")
    
    adoption_data = df.loc[:, df.columns.get_level_values(1).isin(tier_columns)]
    
    # Remove unused levels to clean up the columns
    adoption_data.columns = adoption_data.columns.remove_unused_levels()
    
    # Plotting setup
    n = len(adoption_data.index)
    bar_width = 0.35  # Width of bars
    index = list(range(n))  # Base index for bars
    
    for i, scenario in enumerate(scenarios):
        try:
            # Find the Economic Adopter column for this scenario
            econ_col = None
            for col in adoption_data.columns:
                if scenario in col[0] and col[1] == 'Economic Adopter':
                    econ_col = col

            if econ_col is not None:
                econ_values = adoption_data[econ_col].values
                scenario_index = np.array(index) + i * bar_width
                ax.bar(
                    scenario_index,
                    econ_values,
                    bar_width,
                    color=color_mapping['Economic Adopter'],
                    edgecolor='white',
                    label='Economic Adopter' if i == 0 else ""
                )
            else:
                raise ValueError(f"No 'Economic Adopter' column found for scenario {scenario}. "
                                 f"Available: {adoption_data.columns.tolist()}")
                
        except Exception as e:
            print(f"Error plotting scenario {scenario}: {str(e)}")
    
    # Set axis labels and title
    ax.set_xlabel(x_label, fontweight='bold', fontsize=24)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=24)
    ax.set_title(title, fontweight='bold', fontsize=24)

    # Set x-ticks and labels
    if n > 0:
        ax.set_xticks([i + bar_width / 2 for i in range(n)])
        
        # Format x-tick labels based on index structure and format choice
        if isinstance(adoption_data.index, pd.MultiIndex):
            # Format tick labels based on the selected format
            if x_tick_format == "lmi_only" and adoption_data.index.nlevels > 1:
                # Use only the second level (LMI/MUI classification)
                ax.set_xticklabels([name[1] for name in adoption_data.index.tolist()], 
                                   rotation=90, ha='right')
            
            elif x_tick_format == "fuel_only" and adoption_data.index.nlevels > 0:
                # Use only the first level (fuel type)
                ax.set_xticklabels([name[0] for name in adoption_data.index.tolist()], 
                                   rotation=90, ha='right')
            
            elif x_tick_format == "combined" and adoption_data.index.nlevels > 1:
                # Combine first two levels with comma separator
                tick_labels = [f"{name[0]}, {name[1]}" for name in adoption_data.index.tolist()]
                ax.set_xticklabels(tick_labels, rotation=90, ha='right')
            
            elif x_tick_format == "all":
                # Combine all available levels with comma separators
                tick_labels = []
                for idx in adoption_data.index.tolist():
                    if isinstance(idx, tuple):
                        tick_labels.append(", ".join(str(x) for x in idx))
                    else:
                        tick_labels.append(str(idx))
                ax.set_xticklabels(tick_labels, rotation=90, ha='right')
            
            else:
                # Default: use the full index as is
                ax.set_xticklabels(adoption_data.index.tolist(), rotation=90, ha='right')
        else:
            # For non-MultiIndex, just use the index values
            ax.set_xticklabels(adoption_data.index.tolist(), rotation=90, ha='right')
    
    # Set font size for tick labels
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    
    # Set y-ticks from 0 to 100 in steps of 10%
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylim(0, 100)


def subplot_grid_adoption_vBar(
    dataframes: List[pd.DataFrame],
    scenarios_list: List[List[str]],
    subplot_positions: List[Tuple[int, int]],
    filter_fuel: Optional[List[str]] = None,
    x_labels: Optional[List[str]] = None,
    plot_titles: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    figure_size: Tuple[int, int] = (12, 10),
    sharex: bool = False,
    sharey: bool = False,
    x_tick_format: str = "lmi_only"  # New parameter to control x-tick formatting across all subplots
) -> plt.Figure:
    """
    Creates a grid of subplots to visualize adoption rates across different scenarios using LMI/MUI classification.
    
    Args:
        dataframes: List of DataFrames, each formatted for use in plot_adoption_rate_bar
        scenarios_list: List of scenario identifiers for each DataFrame
        subplot_positions: Positions of subplots in grid as (row, col) tuples
        filter_fuel: Optional list of fuel types to filter by
        x_labels: Optional labels for x-axis of each subplot
        plot_titles: Optional titles for each subplot
        y_labels: Optional labels for y-axis of each subplot
        suptitle: Optional central title for entire figure
        figure_size: Size of entire figure (width, height) in inches
        sharex: Whether subplots should share same x-axis
        sharey: Whether subplots should share same y-axis
        x_tick_format: Format for x-tick labels across all subplots. Options:
                      "lmi_only", "fuel_only", "combined", "all"
        
    Returns:
        Matplotlib Figure object containing the visualization
        
    Raises:
        ValueError: If inputs are incompatible or improperly formatted
    """
    # Define the color mapping for the economic-adopter bar
    color_mapping = {
        'Economic Adopter': 'steelblue',
    }

    # Validate input lengths
    if not (len(dataframes) == len(scenarios_list) == len(subplot_positions)):
        raise ValueError("Length mismatch: dataframes, scenarios_list, and subplot_positions must have the same length")
    
    # Determine grid dimensions from subplot positions
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    # Create figure and axes
    # fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)    
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=figure_size,
        sharex=sharex,
        sharey=sharey,
        dpi=600  # High resolution for better quality!
    )

    # Ensure axes is always 2D for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = np.array(axes)

    for idx, (df, scenarios) in enumerate(zip(dataframes, scenarios_list)):
        try:
            # Get the subplot position
            pos = subplot_positions[idx]
            ax = axes[pos[0], pos[1]]
            
            # Apply additional fuel filtering if requested
            # (create_multiIndex_adoption_df already filters, but this allows further filtering)
            filtered_df = df.copy()
            if filter_fuel:
                # Check if fuel is in index and filter
                fuel_level_names = [name for name in df.index.names if 'fuel' in name.lower()]
                if fuel_level_names:
                    fuel_level = fuel_level_names[0]
                    filtered_df = filtered_df[filtered_df.index.get_level_values(fuel_level).isin(filter_fuel)]
            
            # Set labels and title if provided
            x_label = x_labels[idx] if x_labels and idx < len(x_labels) else ""
            y_label = y_labels[idx] if y_labels and idx < len(y_labels) else ""
            title = plot_titles[idx] if plot_titles and idx < len(plot_titles) else ""
            
            # Plot the data with consistent x-tick formatting
            plot_adoption_rate_bar(filtered_df, scenarios, title, x_label, y_label, ax, x_tick_format)
            
        except Exception as e:
            print(f"Error plotting subplot at position {pos}: {str(e)}")
            # Create an empty plot with error message
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Add a title to the entire figure if provided
    if suptitle:
        fig.suptitle(suptitle, fontweight='bold', fontsize=26)

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_mapping.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in legend_labels]
            
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='lower center', 
        ncol=len(legend_labels), 
        prop={'size': 22}, 
        labelspacing=0.5, 
        bbox_to_anchor=(0.5, -0.10)
    )

    # First apply tight_layout with reasonable rect parameters
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    
    # Add appropriate bottom padding for x-tick labels
    fig.subplots_adjust(bottom=0.25)
    
    # Loop through all axes to add more padding between tick labels and axis label
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i,j].xaxis.labelpad = 20  # Increase space between ticks and label

    return fig


# =========================================================================
# UPDATED PRINTING FUNCTIONS FOR ADOPTION DECISION PERCENTAGES
# =========================================================================
"""
Print adoption decision percentages with REQUIRED population-weighted overall calculation.

This version always calculates "Overall" using population weights (number of homes
in each fuel×income group) to match the methodology used in the climate impact analysis.

Required inputs:
- source_dataframes: Original DataFrames for population weighting
- category: Equipment category to find fuel column (e.g., 'heating' -> 'base_heating_fuel')
"""

def print_adoption_decision_percentages(
    dataframes: List[pd.DataFrame],
    scenario_names: List[str],
    source_dataframes: List[pd.DataFrame],
    category: str,
    title: str = None,
    subtitle: Optional[str] = None,
    print_header_key: bool = True,
    filter_fuel: Optional[List[str]] = None,
) -> None:
    """
    Print adoption decision percentages with population-weighted overall calculation.
    
    Calculates "Overall" using population weights to match the climate analysis 
    methodology where each home is weighted equally.
    
    Args:
        dataframes: List of multi-index DataFrames from create_multiIndex_adoption_df.
        scenario_names: List of scenario names (e.g., ['ref2025_mp3_...']).
        source_dataframes: List of original DataFrames for population weighting.
            Must have same length as scenario_names.
        category: Equipment category (e.g., 'heating', 'waterHeating', 'cooking').
            Used to find fuel column: f'base_{category}_fuel'
        title: Optional section title.
        subtitle: Optional subtitle for the section.
        print_header_key: Whether to print the header key explaining the output format.
        filter_fuel: Optional list of fuels to include. 
            Default: ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']
    
    Returns:
        None. Prints formatted output to stdout.
    
    Raises:
        ValueError: If source_dataframes length doesn't match scenario_names.
        KeyError: If required columns (fuel, income) are missing from source DataFrames.
    
    Example Output:
        ('Overall [Pop-Weighted]'): T1 23% + T2 8% = TAD 31%, TAD + T3 46% = TADS 77%
    """
    # =================================================================
    # INPUT VALIDATION (Fail Fast)
    # =================================================================
    
    # Validate source_dataframes is provided and has correct length
    if source_dataframes is None:
        raise ValueError(
            "source_dataframes is required for population-weighted calculation. "
            "Pass the original DataFrames used to create the multi-index DataFrames."
        )
    
    if len(source_dataframes) != len(scenario_names):
        raise ValueError(
            f"source_dataframes length ({len(source_dataframes)}) must match "
            f"scenario_names length ({len(scenario_names)})."
        )
    
    if len(dataframes) != len(scenario_names):
        raise ValueError(
            f"dataframes length ({len(dataframes)}) must match "
            f"scenario_names length ({len(scenario_names)})."
        )
    
    fuel_col = f'base_{category}_fuel'
    income_col = 'lmi_or_mui'

    for idx, source_df in enumerate(source_dataframes):
        missing_cols = [c for c in [fuel_col, income_col] if c not in source_df.columns]
        if missing_cols:
            raise KeyError(
                f"Source DataFrame at index {idx} is missing required columns: {missing_cols}."
            )

    # Single metric: Economic Adopter (NPV >= 0).
    header_key = """
(Base Fuel, Income Level):
    Economic Adopter (%): Homes where the incremental heat-pump cost is
        recovered from energy-bill savings (private NPV >= 0).
    Overall is population-weighted (each home counted equally).
"""
    
    allowed_fuels = filter_fuel if filter_fuel else ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']

    if title is not None:
        print("-" * 80)
        print(f"{title.upper()}")
        print("-" * 80)

    if print_header_key:
        print(header_key)

    if subtitle is not None:
        print(f"\n{subtitle.upper()}\n")

    print(f"Scenarios: {' | '.join(scenario_names)}")
    print("-" * 80)

    # =================================================================
    # PROCESS EACH SCENARIO
    # =================================================================
    all_results = {}

    for idx, (df, scenario_name, source_df) in enumerate(
        zip(dataframes, scenario_names, source_dataframes)
    ):
        valid_source = source_df[
            source_df[fuel_col].isin(allowed_fuels)
            & source_df[income_col].isin(['LMI', 'MUI'])
        ]
        group_counts = valid_source.groupby(
            [fuel_col, income_col], observed=True
        ).size()
        total_homes = group_counts.sum()

        # Apply fuel filter to multi-index df
        if filter_fuel is not None:
            fuel_level_names = [n for n in df.index.names if 'fuel' in n.lower()]
            if fuel_level_names:
                df = df[df.index.get_level_values(fuel_level_names[0]).isin(filter_fuel)]

        # Find the scenario column (Economic Adopter) in the MultiIndex
        scenario_columns = [
            col for col in df.columns.get_level_values(0).unique()
            if scenario_name.lower().replace('-', '').replace('_', '') in
               col.lower().replace('-', '').replace('_', '')
        ]
        if not scenario_columns:
            raise ValueError(f"No columns found for scenario '{scenario_name}'")
        scenario_col = scenario_columns[0]
        tier_key = (scenario_col, 'Economic Adopter')

        # Population-weighted overall
        weighted_econ = 0.0
        for group_idx in df.index:
            if not isinstance(group_idx, tuple):
                continue
            fuel, income = group_idx
            w = (
                group_counts.get((fuel, income), 0) / total_homes
                if total_homes else 0.0
            )
            econ_val = df.loc[group_idx].get(tier_key, 0)
            weighted_econ += econ_val * w

        overall_key = "('Overall [Pop-Weighted]')"
        if overall_key not in all_results:
            all_results[overall_key] = []
        all_results[overall_key].append(f"Economic Adopter {weighted_econ:.0f}%")

        # Per-group percentages
        for group_idx in df.index:
            if isinstance(group_idx, tuple):
                fuel, income = group_idx
                group_key = f"('{fuel}', '{income}')"
            else:
                group_key = f"('{group_idx}')"

            econ_val = df.loc[group_idx].get(tier_key, 0)

            if group_key not in all_results:
                all_results[group_key] = []
            all_results[group_key].append(f"Economic Adopter {econ_val:.0f}%")

    # =================================================================
    # PRINT RESULTS
    # =================================================================
    for group_key, scenario_results in all_results.items():
        print(f"{group_key}: {' | '.join(scenario_results)}")

    print()  # blank line after section


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
"""
discount_rate = 'fixed_base'
cost_scenario = 'v4MID'
npv_case = 'heatingLCC_coolingLCC'
menu_mp = 3

print_adoption_decision_percentages(
    dataframes=[
        ALL_HEATING_ADOPTION_MI[menu_mp][npv_case][cost_scenario][discount_rate],
    ],
    scenario_names=[
        build_adoption_scenario_names(menu_mp, npv_case, cost_scenario, discount_rate)[0],
    ],
    source_dataframes=[
        DATAFRAMES_BY_MP[menu_mp][discount_rate],
    ],
    category='heating',
    title="SPACE HEATING ADOPTION POTENTIAL: ECONOMIC ADOPTERS (NPV >= 0)",
    subtitle=f"MP{menu_mp}: {npv_case}",
    print_header_key=True,
)
"""


