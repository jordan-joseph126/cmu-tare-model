import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Dict, Any, Union

from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_potential_utils import (
    verify_columns_exist,
    detect_adoption_columns,
    filter_columns,
    filter_by_fuel,
    plot_adoption_rate_bar
)

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
    if 'lowModerateIncome_designation' in df.columns:
        df['lowModerateIncome_designation'] = pd.Categorical(
            df['lowModerateIncome_designation'],
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
            [f'base_{category}_fuel', 'lowModerateIncome_designation'],
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
    
    # Sort by index
    filtered_df.sort_index(level=[f'base_{category}_fuel', 'lowModerateIncome_designation'], inplace=True)
    
    return filtered_df


def subplot_grid_adoption_vBar(
        dataframes: List[pd.DataFrame],
        subplot_positions: List[Tuple[int, int]],
        categories: List[str],
        menu_mp: int,
        scc: str,
        rcm_model: str,
        cr_function: str,
        filter_fuel: Optional[List[str]] = None,
        x_labels: Optional[List[str]] = None,
        plot_titles: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        suptitle: Optional[str] = None,
        figure_size: Tuple[int, int] = (12, 10),
        sharex: bool = False,
        sharey: bool = False
) -> None:
    """
    Creates a grid of subplots to visualize adoption rates across different categories.
    
    Args:
        dataframes: List of DataFrames with adoption data
        subplot_positions: Positions of subplots as (row, col) tuples
        categories: List of equipment categories (e.g., ['heating', 'waterHeating'])
        menu_mp: Measure package identifier
        scc: Social cost of carbon assumption
        rcm_model: RCM model
        cr_function: Concentration-response function
        filter_fuel: List of fuel types to filter by
        x_labels: Labels for the x-axis of each subplot
        plot_titles: Titles for each subplot
        y_labels: Labels for the y-axis of each subplot
        suptitle: Central title for the entire figure
        figure_size: Size of the figure as (width, height)
        sharex: Whether subplots should share the same x-axis
        sharey: Whether subplots should share the same y-axis
    """
    # Define consistent color mapping for legend
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }
    
    # Basic validation
    if len(dataframes) != len(subplot_positions) or len(dataframes) != len(categories):
        print("Error: Number of dataframes, positions, and categories must match")
        return
    
    # Build scenarios list for each DataFrame
    scenarios_list = []
    for category in categories:
        # Try new column naming pattern first
        new_cols = [
            f'preIRA_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}',
            f'iraRef_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
        ]
        
        # Fall back to old pattern if needed (without checking DataFrame yet)
        old_cols = [
            f'preIRA_mp{menu_mp}_{category}_adoption_lrmer',
            f'iraRef_mp{menu_mp}_{category}_adoption_lrmer'
        ]
        
        # We'll add these to the scenarios list and check actual existence later
        scenarios_list.append(new_cols)
    
    # Determine grid dimensions
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1
    
    # Create figure and axes
    fig, axes = plt.subplots(
        nrows=num_rows, 
        ncols=num_cols, 
        figsize=figure_size, 
        sharex=sharex, 
        sharey=sharey
    )
    
    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Process each subplot
    for idx, (df, scenarios, category) in enumerate(zip(dataframes, scenarios_list, categories)):
        # Skip if DataFrame is empty
        if df.empty:
            print(f"Warning: Empty DataFrame for {category}")
            continue
        
        # Check if the new columns actually exist
        if not any(col[0] == scenarios[0] for col in df.columns if isinstance(col, tuple)):
            # Try backward compatibility
            old_cols = [
                f'preIRA_mp{menu_mp}_{category}_adoption_lrmer',
                f'iraRef_mp{menu_mp}_{category}_adoption_lrmer'
            ]
            
            if any(col[0] == old_cols[0] for col in df.columns if isinstance(col, tuple)):
                scenarios = old_cols
                print(f"Using backward-compatible column names for {category}")
            else:
                print(f"Warning: No adoption columns found for {category}")
                continue
        
        # Apply fuel filter if needed
        if filter_fuel and isinstance(df.index, pd.MultiIndex):
            # Try to find the fuel level in the index
            fuel_level_idx = None
            for i, name in enumerate(df.index.names):
                if name and 'fuel' in str(name).lower():
                    fuel_level_idx = i
                    break
            
            if fuel_level_idx is not None:
                df = df.loc[df.index.get_level_values(fuel_level_idx).isin(filter_fuel)]
            else:
                print(f"Warning: Could not find fuel level in index for {category}")
        
        # Skip if DataFrame is empty after filtering
        if df.empty:
            print(f"Warning: No data after fuel filtering for {category}")
            continue
        
        # Set up plot parameters
        pos = subplot_positions[idx]
        ax = axes[pos[0], pos[1]]
        
        # Set title and labels
        title = plot_titles[idx] if plot_titles and idx < len(plot_titles) else category.capitalize()
        x_label = x_labels[idx] if x_labels and idx < len(x_labels) else ""
        y_label = y_labels[idx] if y_labels and idx < len(y_labels) else ""
        
        # Plot the data
        plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)
    
    # Add title
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold', fontsize=20)
    
    # Create legend
    legend_labels = list(color_mapping.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) 
                     for label in legend_labels]
    
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='lower center', 
        ncol=len(legend_labels), 
        prop={'size': 20}, 
        labelspacing=0.5, 
        bbox_to_anchor=(0.5, -0.05)
    )
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
