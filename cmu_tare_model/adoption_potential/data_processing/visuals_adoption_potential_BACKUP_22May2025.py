# Updated V2 Working as of May 22, 2024
# Wanted to update the function to add flexibility for lmi or mui instead of income level and provide x-tick label flexibility
# =========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Dict, Any, Union

# =========================================================================
# FUNCTIONS: VISUALIZATION USING DATAFRAMES AND SUBPLOTS
# =========================================================================

def create_multiIndex_adoption_df(
        df: pd.DataFrame,
        menu_mp: int,
        category: str,
        scc: str,
        rcm_model: str,
        cr_function: str
) -> pd.DataFrame:
    """
    Creates a multi-index DataFrame showing adoption percentages by income level and fuel type.
    
    Args:
        df: DataFrame with adoption data
        menu_mp: Measure package identifier
        category: Equipment category
        scc: Social cost of carbon assumption
        rcm_model: RCM model
        cr_function: Concentration-response function
        
    Returns:
        Multi-index DataFrame with adoption percentages
    """
    # Define income categories for sorting
    income_categories = ['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income']
    
    # Convert to categorical for proper sorting
    if 'income_level' in df.columns:
        df['income_level'] = pd.Categorical(
            df['income_level'],
            categories=income_categories,
            ordered=True
        )
    
    # Define column names with sensitivity dimensions
    adoption_cols = [
        f'preIRA_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}',
        f'iraRef_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    ]
    
    # Try backward compatibility if needed
    if not all(col in df.columns for col in adoption_cols):
        old_cols = [
            f'preIRA_mp{menu_mp}_{category}_adoption_lrmer',
            f'iraRef_mp{menu_mp}_{category}_adoption_lrmer'
        ]
        
        if all(col in df.columns for col in old_cols):
            adoption_cols = old_cols
            print(f"Using backward-compatible column names for {category}")
        else:
            print(f"Error: Required adoption columns not found for {category}")
            return pd.DataFrame()  # Return empty DataFrame if columns not found
    
    try:
        # Group by fuel and income, calculate normalized counts
        percentages_df = df.groupby(
            [f'base_{category}_fuel', 'income_level'],
            observed=False
        )[adoption_cols].apply(
            lambda x: x.apply(lambda y: y.value_counts(normalize=True))
        ).unstack().fillna(0) * 100
        
        percentages_df = percentages_df.round(0)
    except Exception as e:
        print(f"Error calculating percentages: {str(e)}")
        return pd.DataFrame()
    
    # Ensure all tiers exist, add if missing
    tiers = [
        'Tier 1: Feasible', 
        'Tier 2: Feasible vs. Alternative', 
        'Tier 3: Subsidy-Dependent Feasibility'
    ]
    
    for column in adoption_cols:
        # Add missing tiers
        for tier in tiers:
            if (column, tier) not in percentages_df.columns:
                percentages_df[(column, tier)] = 0
        
        # Calculate totals
        percentages_df[(column, 'Total Adoption Potential')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')]
        )
        
        percentages_df[(column, 'Total Adoption Potential (Additional Subsidy)')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')] + 
            percentages_df[(column, 'Tier 3: Subsidy-Dependent Feasibility')]
        )
    
    # Rebuild the MultiIndex and filter columns
    percentages_df.columns = pd.MultiIndex.from_tuples(percentages_df.columns)
    
    # Keep only columns with tiers
    keep_tiers = tiers + ['Total Adoption Potential', 'Total Adoption Potential (Additional Subsidy)']
    keep_cols = [(col, tier) for col in adoption_cols for tier in keep_tiers 
                if (col, tier) in percentages_df.columns]
    
    if keep_cols:
        filtered_df = percentages_df.loc[:, keep_cols]
    else:
        print(f"Warning: No tier columns found for {category}")
        return percentages_df  # Return unfiltered if no tiers found
    
    # # Sort by index
    # filtered_df.sort_index(level=[f'base_{category}_fuel', 'income_level'], inplace=True)
    
    # return filtered_df

    # Sort by index
    filtered_df.sort_index(level=[f'base_{category}_fuel', 'income_level'], inplace=True)

    # ADDED: Filter to include only specific fuel types
    allowed_fuels = ['Electricity', 'Fuel Oil', 'Natural Gas', 'Propane']
    fuel_level = f'base_{category}_fuel'
    if isinstance(filtered_df.index, pd.MultiIndex) and fuel_level in filtered_df.index.names:
        filtered_df = filtered_df[filtered_df.index.get_level_values(fuel_level).isin(allowed_fuels)]

    return filtered_df


def plot_adoption_rate_bar(
    df: pd.DataFrame,
    scenarios: List[str],
    title: str,
    x_label: str,
    y_label: str,
    ax: plt.Axes,
    x_tick_format: str = "income_only"  # New parameter with default
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
                      "income_only" - Show only income level
                      "fuel_only" - Show only fuel type
                      "combined" - Show "Fuel Type, Income Level"
                      "all" - Show all index levels separated by commas
        
    Returns:
        None. The plot is created on the provided axes.
        
    Raises:
        ValueError: If required columns are not found in the DataFrame
    """
    # Define the color mapping for adoption tiers
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }
    
    # Ensure the DataFrame is properly formatted
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex for columns")
    
    # Filter the DataFrame to only include the tier columns
    tier_columns = ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility']
    available_columns = df.columns.get_level_values(1).unique()
    
    if not any(tier in available_columns for tier in tier_columns):
        raise ValueError(f"No tier columns found in DataFrame. Available columns: {available_columns}")
    
    adoption_data = df.loc[:, df.columns.get_level_values(1).isin(tier_columns)]
    
    # Remove unused levels to clean up the columns
    adoption_data.columns = adoption_data.columns.remove_unused_levels()
    
    # Plotting setup
    n = len(adoption_data.index)
    bar_width = 0.35  # Width of bars
    index = list(range(n))  # Base index for bars
    
    for i, scenario in enumerate(scenarios):
        try:
            # Find tier columns for this scenario
            # First, check if first-level columns contain the exact scenario string
            tier1_col = None
            tier2_col = None
            tier3_col = None
            
            # Look through all available columns to find matches for this scenario
            for col in adoption_data.columns:
                # Check if the scenario is in the column name (for flexible matching)
                if scenario in col[0]:
                    if col[1] == 'Tier 1: Feasible':
                        tier1_col = col
                    elif col[1] == 'Tier 2: Feasible vs. Alternative':
                        tier2_col = col
                    elif col[1] == 'Tier 3: Subsidy-Dependent Feasibility':
                        tier3_col = col
            
            # Verify we found all needed columns
            if tier1_col and tier2_col and tier3_col:
                tier1 = adoption_data[tier1_col].values
                tier2 = adoption_data[tier2_col].values
                tier3 = adoption_data[tier3_col].values
                
                # Adjust the index for this scenario
                scenario_index = np.array(index) + i * bar_width
                
                # Plot the bars for the scenario
                ax.bar(
                    scenario_index, 
                    tier1, 
                    bar_width, 
                    color=color_mapping['Tier 1: Feasible'], 
                    edgecolor='white'
                )
                ax.bar(
                    scenario_index, 
                    tier2, 
                    bar_width, 
                    bottom=tier1, 
                    color=color_mapping['Tier 2: Feasible vs. Alternative'], 
                    edgecolor='white'
                )
                ax.bar(
                    scenario_index, 
                    tier3, 
                    bar_width, 
                    bottom=(tier1+tier2), 
                    color=color_mapping['Tier 3: Subsidy-Dependent Feasibility'], 
                    edgecolor='white'
                )
            else:
                print(f"Warning: Missing tier columns for scenario {scenario}")
                print(f"Available columns: {adoption_data.columns.tolist()}")
                
        except Exception as e:
            print(f"Error plotting scenario {scenario}: {str(e)}")
    
    # Set axis labels and title
    ax.set_xlabel(x_label, fontweight='bold', fontsize=20)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
    ax.set_title(title, fontweight='bold', fontsize=20)
       
    # Set x-ticks and labels
    if n > 0:
        ax.set_xticks([i + bar_width / 2 for i in range(n)])
        
        # Format x-tick labels based on index structure and format choice
        if isinstance(adoption_data.index, pd.MultiIndex):
            # Get the index names for reference (they might be None)
            index_names = adoption_data.index.names
            
            # Format tick labels based on the selected format
            if x_tick_format == "income_only" and adoption_data.index.nlevels > 1:
                # Use only the second level (typically income designation)
                ax.set_xticklabels([name[1] for name in adoption_data.index.tolist()], 
                                   rotation=90, ha='right')
            
            elif x_tick_format == "fuel_only" and adoption_data.index.nlevels > 0:
                # Use only the first level (typically fuel type)
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
    
    # [Rest of the function remains the same]

    # Set font size for tick labels
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    
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
    sharey: bool = False
) -> plt.Figure:
    """
    Creates a grid of subplots to visualize adoption rates across different scenarios.
    
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
        
    Returns:
        Matplotlib Figure object containing the visualization
        
    Raises:
        ValueError: If inputs are incompatible or improperly formatted
    """
    # Define the color mapping for adoption tiers
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    # Validate input lengths
    if not (len(dataframes) == len(scenarios_list) == len(subplot_positions)):
        raise ValueError("Length mismatch: dataframes, scenarios_list, and subplot_positions must have the same length")
    
    # Determine grid dimensions from subplot positions
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    # Create figure and axes
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
    
    # Ensure axes is always 2D
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
            # (new create_multiIndex_adoption_df already filters, but this allows further filtering)
            if filter_fuel:
                # Check if fuel is in index and filter
                fuel_level_names = [name for name in df.index.names if 'fuel' in name.lower()]
                if fuel_level_names:
                    fuel_level = fuel_level_names[0]
                    df = df[df.index.get_level_values(fuel_level).isin(filter_fuel)]
            
            # Set labels and title if provided
            x_label = x_labels[idx] if x_labels and idx < len(x_labels) else ""
            y_label = y_labels[idx] if y_labels and idx < len(y_labels) else ""
            title = plot_titles[idx] if plot_titles and idx < len(plot_titles) else ""
            
            # Plot the data
            plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)
        except Exception as e:
            print(f"Error plotting subplot at position {pos}: {str(e)}")
            # Create an empty plot with error message
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    # Add a title to the entire figure if provided
    if suptitle:
        fig.suptitle(suptitle, fontweight='bold', fontsize=22)

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_mapping.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in legend_labels]
            
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='lower center', 
        ncol=len(legend_labels), 
        prop={'size': 20}, 
        labelspacing=0.5, 
        bbox_to_anchor=(0.5, -0.05)
    )

    # Adjust the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to leave space for the suptitle
    
    return fig
