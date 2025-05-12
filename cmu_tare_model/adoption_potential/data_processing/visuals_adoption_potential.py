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

def create_df_adoption(
        df: pd.DataFrame, 
        menu_mp: int, 
        category: str,
        scc: str = 'upper', 
        rcm_model: str = 'AP2', 
        cr_function: str = 'acs'
) -> pd.DataFrame:
    """
    Generates a new DataFrame with specific adoption columns.
    
    Supports both new column naming pattern with sensitivity dimensions and
    falls back to the original pattern for backward compatibility.
    
    Args:
        df: Original DataFrame
        menu_mp: Measure package identifier
        category: Equipment category (e.g., 'heating', 'waterHeating')
        scc: Social cost of carbon assumption ('lower', 'central', or 'upper')
        rcm_model: RCM model ('AP2', 'EASIUR', or 'InMAP')
        cr_function: Concentration-response function ('acs' or 'h6c')

    Returns:
        DataFrame with selected columns related to adoption
    """
    # Create a copy of the dataframe
    df_copy = df.copy()
    
    print(f"\nCreating adoption DataFrame for {category}:")
    
    # Begin with basic columns if available
    base_cols = ['state', 'city', 'county', 'puma', 'percent_AMI', 'lowModerateIncome_designation']
    base_cols = [col for col in base_cols if col in df_copy.columns]
    
    # Try to find columns matching both new and old patterns
    adoption_col_new_pre = f'preIRA_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    adoption_col_new_ira = f'iraRef_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    
    adoption_col_old_pre = f'preIRA_mp{menu_mp}_{category}_adoption_lrmer'
    adoption_col_old_ira = f'iraRef_mp{menu_mp}_{category}_adoption_lrmer'
    
    # Check if new pattern columns exist
    new_pattern_exists = (adoption_col_new_pre in df_copy.columns and 
                          adoption_col_new_ira in df_copy.columns)
    
    if new_pattern_exists:
        print(f"- Using new column naming pattern with sensitivity dimensions")
        cols_to_add = [
            f'base_{category}_fuel',
            f'include_{category}',
            f'preIRA_mp{menu_mp}_{category}_public_npv_{scc}_{rcm_model}_{cr_function}',
            f'preIRA_mp{menu_mp}_{category}_total_capitalCost', 
            f'preIRA_mp{menu_mp}_{category}_private_npv_lessWTP',
            adoption_col_new_pre,
            f'mp{menu_mp}_{category}_rebate_amount',
            f'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_{rcm_model}_{cr_function}',
            f'iraRef_mp{menu_mp}_{category}_total_capitalCost', 
            f'iraRef_mp{menu_mp}_{category}_private_npv_lessWTP',
            adoption_col_new_ira
        ]
    else:
        # Fall back to old pattern for backward compatibility
        print(f"- Using original column naming pattern (backward compatibility)")
        cols_to_add = [
            f'base_{category}_fuel',
            f'include_{category}',
            f'preIRA_mp{menu_mp}_{category}_public_npv_lrmer',
            f'preIRA_mp{menu_mp}_{category}_total_capitalCost', 
            f'preIRA_mp{menu_mp}_{category}_private_npv_lessWTP',
            adoption_col_old_pre,
            f'mp{menu_mp}_{category}_rebate_amount',
            f'iraRef_mp{menu_mp}_{category}_public_npv_lrmer',
            f'iraRef_mp{menu_mp}_{category}_total_capitalCost', 
            f'iraRef_mp{menu_mp}_{category}_private_npv_lessWTP',
            adoption_col_old_ira
        ]
    
    # Filter to only include columns that exist in the DataFrame
    cols_to_add = [col for col in cols_to_add if col in df_copy.columns]
    
    # Check if we have at least one adoption column
    adoption_cols = []
    if new_pattern_exists:
        adoption_cols = [adoption_col_new_pre, adoption_col_new_ira]
    else:
        adoption_cols = [adoption_col_old_pre, adoption_col_old_ira]
    
    found_adoption_cols = [col for col in adoption_cols if col in df_copy.columns]
    if not found_adoption_cols:
        raise ValueError(f"No adoption columns found for {category} with menu_mp={menu_mp}")
    
    # Combine base columns with filtered columns
    all_cols = base_cols + cols_to_add
    
    # Select columns that exist in the DataFrame
    existing_cols = [col for col in all_cols if col in df_copy.columns]
    print(f"- Selected {len(existing_cols)} columns for {category}")
    
    return df_copy[existing_cols]


def create_multiIndex_adoption_df(
        df: pd.DataFrame,
        menu_mp: int,
        category: str,
        scc: str = 'upper',
        rcm_model: str = 'AP2',
        cr_function: str = 'acs',
        mer_type: str = 'lrmer'  # For backward compatibility
) -> pd.DataFrame:
    """
    Creates a multi-index DataFrame for adoption visualization.
    
    Supports both new and old column naming patterns for backward compatibility.
    
    Args:
        df: Input DataFrame with adoption columns
        menu_mp: Measure package identifier
        category: Equipment category (e.g., 'heating', 'waterHeating')
        scc: Social cost of carbon assumption ('lower', 'central', or 'upper')
        rcm_model: RCM model ('AP2', 'EASIUR', or 'InMAP')
        cr_function: Concentration-response function ('acs' or 'h6c')
        mer_type: Marginal emission rate type (for backward compatibility)
        
    Returns:
        Multi-index DataFrame with adoption percentages
    """
    # Set up income categories as ordinal
    income_categories = ['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income']
    
    if 'lowModerateIncome_designation' in df.columns:
        df['lowModerateIncome_designation'] = pd.Categorical(
            df['lowModerateIncome_designation'], 
            categories=income_categories, 
            ordered=True
        )
    
    # Try both new and old column naming patterns
    # First try new pattern with sensitivity dimensions
    new_pre_col = f'preIRA_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    new_ira_col = f'iraRef_mp{menu_mp}_{category}_adoption_{scc}_{rcm_model}_{cr_function}'
    
    # Check if new pattern columns exist
    new_pattern_exists = (new_pre_col in df.columns and new_ira_col in df.columns)
    
    # Define the columns for adoption data
    if new_pattern_exists:
        print(f"Using new column pattern with sensitivity dimensions")
        adoption_cols = [new_pre_col, new_ira_col]
    else:
        print(f"Using original column pattern (backward compatibility)")
        old_pre_col = f'preIRA_mp{menu_mp}_{category}_adoption_{mer_type}'
        old_ira_col = f'iraRef_mp{menu_mp}_{category}_adoption_{mer_type}'
        adoption_cols = [old_pre_col, old_ira_col]
    
    # Check if adoption columns exist
    missing_cols = [col for col in adoption_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing adoption columns: {missing_cols}")
    
    # Determine the fuel column name
    fuel_col = f'base_{category}_fuel'
    if fuel_col not in df.columns:
        raise ValueError(f"Required column '{fuel_col}' not found")
    
    # Group by fuel type and income designation, calculate normalized counts
    percentages_df = df.groupby([fuel_col, 'lowModerateIncome_designation'], observed=False)[adoption_cols].apply(
        lambda x: x.apply(lambda y: y.value_counts(normalize=True))).unstack().fillna(0) * 100
    percentages_df = percentages_df.round(0)

    # Ensure all tier columns exist
    tiers = ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility']
    for column in adoption_cols:
        for tier in tiers:
            if (column, tier) not in percentages_df.columns:
                percentages_df[(column, tier)] = 0

        # Calculate total adoption potential
        percentages_df[(column, 'Total Adoption Potential')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')]
        )

        # Calculate total with additional subsidy
        percentages_df[(column, 'Total Adoption Potential (Additional Subsidy)')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')] + 
            percentages_df[(column, 'Tier 3: Subsidy-Dependent Feasibility')]
        )

    # Rebuild the column MultiIndex
    percentages_df.columns = pd.MultiIndex.from_tuples(percentages_df.columns)
    
    # Filter and reorder columns
    filtered_df = filter_columns(percentages_df)
    
    # Sort by index
    sorted_df = filtered_df.sort_index(level=[fuel_col, 'lowModerateIncome_designation'])
    
    return sorted_df


def subplot_grid_adoption_vBar(
        dataframes: List[pd.DataFrame],
        subplot_positions: List[Tuple[int, int]],
        categories: Optional[List[str]] = None,
        menu_mp: Optional[int] = None,
        scenarios_list: Optional[List[List[str]]] = None,
        scc: str = 'upper',
        rcm_model: str = 'AP2',
        cr_function: str = 'acs',
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
                   Required if scenarios_list is not provided.
        menu_mp: Measure package identifier
                 Required if scenarios_list is not provided.
        scenarios_list: List of lists of column names for each DataFrame 
                      If provided, overrides categories and menu_mp.
        scc: Social cost of carbon assumption ('lower', 'central', or 'upper')
        rcm_model: RCM model ('AP2', 'EASIUR', or 'InMAP')
        cr_function: Concentration-response function ('acs' or 'h6c')
        filter_fuel: List of fuel types to filter by
        x_labels: Labels for the x-axis of each subplot
        plot_titles: Titles for each subplot
        y_labels: Labels for the y-axis of each subplot
        suptitle: Central title for the entire figure
        figure_size: Size of the figure as (width, height)
        sharex: Whether subplots should share the same x-axis
        sharey: Whether subplots should share the same y-axis
    """
    print(f"\nCreating subplot grid for adoption visualization:")
    print(f"- Number of dataframes: {len(dataframes)}")
    print(f"- Number of subplot positions: {len(subplot_positions)}")
    if categories:
        print(f"- Categories: {categories}")
    print(f"- menu_mp: {menu_mp}")
    print(f"- scc: {scc}, rcm_model: {rcm_model}, cr_function: {cr_function}")
    if filter_fuel:
        print(f"- Filtering for fuel types: {filter_fuel}")
    
    # Define color mapping for legend
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    # Validate inputs
    if not dataframes or len(dataframes) != len(subplot_positions):
        print("! Error: Invalid input parameters. Check dataframes and positions.")
        return
        
    # Derive scenarios from categories if scenarios_list not provided
    derived_scenarios_list = []
    if scenarios_list is None:
        if categories is None or menu_mp is None:
            print("! Error: Must provide either scenarios_list or both categories and menu_mp.")
            return
            
        if len(categories) != len(dataframes):
            print("! Warning: Length of categories doesn't match length of dataframes.")
            if len(categories) < len(dataframes):
                print(f"  Truncating to {len(categories)} dataframes")
                dataframes = dataframes[:len(categories)]
                subplot_positions = subplot_positions[:len(categories)]
            else:
                print(f"  Using only {len(dataframes)} categories")
                categories = categories[:len(dataframes)]
            
        # Derive scenarios for each DataFrame
        for df, category in zip(dataframes, categories):
            try:
                adoption_cols = detect_adoption_columns(
                    df, menu_mp, category, scc, rcm_model, cr_function
                )
                derived_scenarios_list.append(adoption_cols)
                print(f"- Detected adoption columns for {category}: {adoption_cols}")
            except Exception as e:
                print(f"! Error detecting adoption columns for {category}: {str(e)}")
                derived_scenarios_list.append([])
        
        # Use derived scenarios
        scenarios_list = derived_scenarios_list
    
    # Ensure scenarios_list matches dataframes length
    if len(scenarios_list) != len(dataframes):
        print("! Warning: Length of scenarios_list doesn't match length of dataframes.")
        min_len = min(len(scenarios_list), len(dataframes))
        dataframes = dataframes[:min_len]
        subplot_positions = subplot_positions[:min_len]
        scenarios_list = scenarios_list[:min_len]
        if categories:
            categories = categories[:min_len]
    
    # Determine grid dimensions
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1
    
    print(f"- Creating {num_rows}x{num_cols} subplot grid")

    # Create figure and axes
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, 
                           sharex=sharex, sharey=sharey)
    
    # Handle single subplot case
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])
        
    # Process each subplot
    for idx, (df, scenarios) in enumerate(zip(dataframes, scenarios_list)):
        # Skip if no scenarios found
        if not scenarios:
            print(f"! Warning: No scenarios for DataFrame {idx}. Skipping.")
            continue
            
        # Apply fuel filter if provided
        if filter_fuel and not df.empty:
            df = filter_by_fuel(df, filter_fuel)
            
        # Skip if filtered DataFrame is empty
        if df.empty:
            print(f"! Warning: No data after fuel filtering for DataFrame {idx}. Skipping.")
            continue
            
        try:
            # Get subplot position and configure
            pos = subplot_positions[idx]
            ax = axes[pos[0], pos[1]]
            
            # Set labels if provided
            x_label = x_labels[idx] if x_labels and idx < len(x_labels) else ""
            y_label = y_labels[idx] if y_labels and idx < len(y_labels) else ""
            
            # Set title - use category if available, otherwise use provided title
            if plot_titles and idx < len(plot_titles):
                title = plot_titles[idx]
            elif categories and idx < len(categories):
                title = categories[idx].capitalize()
            else:
                title = f"Plot {idx+1}"

            # Plot the data
            plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)
            
        except Exception as e:
            print(f"! Error plotting at position {pos}: {str(e)}")
            # Get subplot position
            pos = subplot_positions[idx]
            ax = axes[pos[0], pos[1]]
            # Display error message on plot
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', 
                  transform=ax.transAxes, color='red')

    # Add title and legend
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold', fontsize=20)

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

    
# ========================================================================
# EXAMPLE USAGE
# ========================================================================
# # Create adoption DataFrames for various categories
# df_heating = create_df_adoption(df_euss_am_mp8_home_easiur, menu_mp=8, category='heating', 
#                                rcm_model='EASIUR', cr_function='acs')
# df_water = create_df_adoption(df_euss_am_mp8_home_easiur, menu_mp=8, category='waterHeating', 
#                              rcm_model='EASIUR', cr_function='acs')

# # Convert to multi-index format for visualization
# mi_heating = create_multiIndex_adoption_df(df_heating, menu_mp=8, category='heating',
#                                          rcm_model='EASIUR', cr_function='acs')
# mi_water = create_multiIndex_adoption_df(df_water, menu_mp=8, category='waterHeating',
#                                        rcm_model='EASIUR', cr_function='acs')

# # Create visualization
# subplot_grid_adoption_vBar(
#     dataframes=[mi_heating, mi_water],
#     subplot_positions=[(0, 0), (0, 1)],
#     categories=['heating', 'waterHeating'],
#     menu_mp=8,
#     rcm_model='EASIUR',  # Now correctly passed to detect_adoption_columns
#     suptitle='Adoption Potential by Tier',
#     filter_fuel=['Electricity', 'Natural Gas']
# )