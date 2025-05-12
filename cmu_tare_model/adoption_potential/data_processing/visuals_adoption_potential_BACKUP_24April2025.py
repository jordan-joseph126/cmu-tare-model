import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple, Dict, Any, Union

from cmu_tare_model.adoption_potential.data_processing.visuals_adoption_potential_utils import (
    generate_column_groups,
    verify_columns_exist,
    detect_adoption_columns,
    filter_columns,
    filter_by_fuel,
    plot_adoption_rate_bar
)

# =========================================================================
# FUNCTIONS: VISUALIZATION USING DATAFRAMES AND SUBPLOTS
# =========================================================================

def create_df_adoption(
        df: pd.DataFrame, 
        menu_mp: int, 
        category: str, 
        scc: str = 'upper', 
        rcm_model: str = 'AP2', 
        cr_function: str = 'acs', 
        mer_type: str = 'lrmer'
) -> pd.DataFrame:
    """
    Generates a new DataFrame with specific adoption columns.
    
    Args:
        df: Original DataFrame
        menu_mp: Measure package identifier
        category: Equipment category (e.g., 'heating', 'waterHeating')
        scc: Social cost of carbon assumption ('lower', 'central', or 'upper')
        rcm_model: RCM model ('AP2', 'EASIUR', or 'InMAP')
        cr_function: Concentration-response function ('acs' or 'h6c')
        mer_type: Marginal emission rate type ('lrmer' or 'srmer')

    Returns:
        DataFrame with selected columns
    """
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Define required columns in the desired order
    required_cols = [
        f'base_{category}_fuel',
        f'include_{category}',
        'percent_AMI',
        'lowModerateIncome_designation',
        f'preIRA_mp{menu_mp}_{category}_public_npv_{scc}_{rcm_model}_{cr_function}',    # PRE IRA REFERENCE CASE SCENARIO
        f'preIRA_mp{menu_mp}_{category}_total_capitalCost', 
        f'preIRA_mp{menu_mp}_{category}_private_npv_lessWTP',
        f'preIRA_mp{menu_mp}_{category}_total_npv_lessWTP_{scc}_{rcm_model}_{cr_function}',
        f'preIRA_mp{menu_mp}_{category}_net_capitalCost',
        f'preIRA_mp{menu_mp}_{category}_private_npv_moreWTP',
        f'preIRA_mp{menu_mp}_{category}_total_npv_moreWTP_{scc}_{rcm_model}_{cr_function}',
        f'preIRA_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}',
        f'mp{menu_mp}_{category}_rebate_amount',    # IRA REFERENCE CASE SCENARIO
        f'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_{rcm_model}_{cr_function}',
        f'iraRef_mp{menu_mp}_{category}_total_capitalCost', 
        f'iraRef_mp{menu_mp}_{category}_private_npv_lessWTP',
        f'iraRef_mp{menu_mp}_{category}_total_npv_lessWTP_{scc}_{rcm_model}_{cr_function}',
        f'iraRef_mp{menu_mp}_{category}_net_capitalCost',
        f'iraRef_mp{menu_mp}_{category}_private_npv_moreWTP',
        f'iraRef_mp{menu_mp}_{category}_total_npv_moreWTP_{scc}_{rcm_model}_{cr_function}',
        f'iraRef_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}',
    ]

    # Define optional columns (sensitivity analysis)
    optional_cols = [
        f'preIRA_mp{menu_mp}_{category}_avoided_mt_co2e_{mer_type}',    
        f'iraRef_mp{menu_mp}_{category}_avoided_mt_co2e_{mer_type}',
        f'iraRef_mp{menu_mp}_{category}_usd2023_per_mtCO2e_{mer_type}',
        f'iraRef_mp{menu_mp}_{category}_benefit_{scc}_{rcm_model}_{cr_function}',
    ]

    try:
        # Use verify_columns_exist for consistent column checking
        existing_cols = verify_columns_exist(
            df=df_copy,
            required_cols=required_cols,
            optional_cols=optional_cols
            )
            
        # Select the relevant columns in the original order
        return df_copy[existing_cols]
    
    except Exception as e:
        print(f"Error creating adoption DataFrame: {str(e)}")
        return df.copy()


def create_multiIndex_adoption_df(
        df: pd.DataFrame, 
        menu_mp: int, 
        category: str, 
        scc: str = 'upper', 
        rcm_model: str = 'AP2', 
        cr_function: str = 'acs',
        mer_type: str = 'lrmer'  # Kept for backward compatibility
) -> pd.DataFrame:
    """
    Creates a multi-index DataFrame showing adoption percentages by tier, fuel type, and income level.
    
    Args:
        df: DataFrame containing adoption data
        menu_mp: Measure package identifier
        category: Equipment category (e.g., 'heating', 'waterHeating')
        scc: Social cost of carbon assumption
        rcm_model: RCM model 
        cr_function: Concentration-response function
        mer_type: Marginal emission rate type (kept for backward compatibility)
        
    Returns:
        A filtered multi-index DataFrame with adoption percentages
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    try:
        # Define required columns and verify they exist
        fuel_col = f'base_{category}_fuel'
        verify_columns_exist(df_processed, [fuel_col, 'lowModerateIncome_designation'])
        
        # Get adoption columns (will raise ValueError if not found)
        adoption_cols = detect_adoption_columns(
            df_processed, menu_mp, category, scc, rcm_model, cr_function
        )
        
        # Set 'lowModerateIncome_designation' as categorical with order
        income_categories = ['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income']            
        df_processed['lowModerateIncome_designation'] = pd.Categorical(
            df_processed['lowModerateIncome_designation'], 
            categories=income_categories, 
            ordered=True
        )
        
        # Print value counts of adoption columns for debugging
        print(f"\nAdoption column value counts for {category}:")
        for col in adoption_cols:
            print(f"\n{col}:")
            print(df_processed[col].value_counts())
        
        # Group by fuel and income, calculate normalized counts
        percentages_df = df_processed.groupby(
            [fuel_col, 'lowModerateIncome_designation'], 
            observed=False
        )[adoption_cols].apply(
            lambda x: x.apply(lambda y: y.value_counts(normalize=True))
        ).unstack().fillna(0) * 100
        
        # Round for readability
        percentages_df = percentages_df.round(0)
        
        # Ensure tier columns exist and calculate totals
        for column in adoption_cols:
            for tier in ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 
                        'Tier 3: Subsidy-Dependent Feasibility']:
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
            
        # Rebuild MultiIndex and filter
        percentages_df.columns = pd.MultiIndex.from_tuples(percentages_df.columns)
        filtered_df = filter_columns(percentages_df)
        
        # Sort the DataFrame appropriately  
        if not filtered_df.empty:
            # Sort by index levels if they exist
            index_levels = [level for level in filtered_df.index.names if level is not None]
            if index_levels:
                try:
                    filtered_df = filtered_df.sort_index(level=index_levels)
                except Exception as e:
                    print(f"Warning: Could not sort DataFrame: {str(e)}")
                
        return filtered_df
        
    except Exception as e:
        print(f"Error processing adoption data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

# ==============================================================================
# NEEDS TO BE UPDATED FOR SIMPLIFIED APPROACH ABOVE!!!
# ==============================================================================
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
) -> None:
    """
    Creates a grid of subplots to visualize adoption rates across different scenarios.

    Args:
        dataframes: List of DataFrames with adoption data
        scenarios_list: List of column names for each DataFrame 
                      (e.g., ['preIRA_mp8_heating_adoption_upper_AP2_acs'])
        subplot_positions: Positions of subplots as (row, col) tuples
        filter_fuel: List of fuel types to filter by
        x_labels: Labels for the x-axis of each subplot
        plot_titles: Titles for each subplot
        y_labels: Labels for the y-axis of each subplot
        suptitle: Central title for the entire figure
        figure_size: Size of the figure as (width, height)
        sharex: Whether subplots should share the same x-axis
        sharey: Whether subplots should share the same y-axis
    
    Notes:
        When using the new column naming pattern, provide the full column names:
        - Old format: 'preIRA_mp8_heating_adoption_lrmer'
        - New format: 'preIRA_mp8_heating_adoption_upper_AP2_acs'
    """
    # Define color mapping
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    # Validate inputs
    if not dataframes or len(dataframes) != len(scenarios_list) or len(dataframes) != len(subplot_positions):
        print("Warning: Invalid input parameters. Check dataframes, scenarios, and positions.")
        return
    
    # Determine grid dimensions
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    # Create figure and axes
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
    
    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])
        
    # Process each subplot
    for idx, (df, scenarios) in enumerate(zip(dataframes, scenarios_list)):
        # Apply fuel filter if provided
        if filter_fuel and not df.empty:
            df = filter_by_fuel(df, filter_fuel)
        
        # Get subplot position and configure
        pos = subplot_positions[idx]
        ax = axes[pos[0], pos[1]]
        
        # Set labels if provided
        x_label = x_labels[idx] if x_labels and idx < len(x_labels) else ""
        y_label = y_labels[idx] if y_labels and idx < len(y_labels) else ""
        title = plot_titles[idx] if plot_titles and idx < len(plot_titles) else ""

        # Plot the data
        try:
            plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)
        except Exception as e:
            print(f"Error plotting at position {pos}: {str(e)}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', 
                  transform=ax.transAxes, color='red')

    # Add title and legend
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')

    # Create legend
    legend_labels = list(color_mapping.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) 
                    for label in legend_labels]
            
    fig.legend(legend_handles, legend_labels, loc='lower center', 
             ncol=len(legend_labels), prop={'size': 20}, 
             labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))

    # Adjust layout and display
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
