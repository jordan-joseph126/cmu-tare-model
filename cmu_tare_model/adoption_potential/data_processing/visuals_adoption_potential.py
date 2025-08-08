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
    Creates a multi-index DataFrame showing adoption percentages by LMI/MUI classification and fuel type.
    
    Args:
        df: DataFrame with adoption data
        menu_mp: Measure package identifier
        category: Equipment category
        scc: Social cost of carbon assumption
        rcm_model: RCM model
        cr_function: Concentration-response function
        
    Returns:
        Multi-index DataFrame with adoption percentages
        
    Raises:
        ValueError: If required columns are not found in the DataFrame
    """
    # Define LMI/MUI categories for sorting
    lmi_mui_categories = ['LMI', 'MUI']
    
    # Validate that the required column exists
    if 'lmi_or_mui' not in df.columns:
        raise ValueError("Required column 'lmi_or_mui' not found in DataFrame."
                        "Please ensure the DataFrame has been processed with the updated calculate_percent_AMI function.")
    
    # Convert to categorical for proper sorting
    df['lmi_or_mui'] = pd.Categorical(
        df['lmi_or_mui'],
        categories=lmi_mui_categories,
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
            available_cols = [col for col in df.columns if 'adoption' in col]
            raise ValueError(f"Required adoption columns not found for {category}. "
                           f"Expected: {adoption_cols}. Available adoption columns: {available_cols}")
    
    try:
        # Group by fuel and LMI/MUI classification, calculate normalized counts
        percentages_df = df.groupby(
            [f'base_{category}_fuel', 'lmi_or_mui'],
            observed=False
        )[adoption_cols].apply(
            lambda x: x.apply(lambda y: y.value_counts(normalize=True))
        ).unstack().fillna(0) * 100
        
        percentages_df = percentages_df.round(0)
    except Exception as e:
        raise ValueError(f"Error calculating percentages for {category}: {str(e)}. "
                        f"Check that required columns exist and contain expected values.")
    
    # Ensure all tiers exist, add if missing
    tiers = [
        'Tier 1: Feasible', 
        'Tier 2: Feasible vs. Alternative', 
        'Tier 3: Subsidy-Dependent Feasibility'
    ]
    
    for column in adoption_cols:
        # Add missing tiers with zero values
        for tier in tiers:
            if (column, tier) not in percentages_df.columns:
                percentages_df[(column, tier)] = 0
        
        # Calculate adoption potential totals
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
    
    # Keep only columns with tiers and totals
    keep_tiers = tiers + ['Total Adoption Potential', 'Total Adoption Potential (Additional Subsidy)']
    keep_cols = [(col, tier) for col in adoption_cols for tier in keep_tiers 
                if (col, tier) in percentages_df.columns]
    
    if keep_cols:
        filtered_df = percentages_df.loc[:, keep_cols]
    else:
        print(f"Warning: No tier columns found for {category}")
        return percentages_df  # Return unfiltered if no tiers found

    # Sort by index (fuel type and LMI/MUI classification)
    filtered_df.sort_index(level=[f'base_{category}_fuel', 'lmi_or_mui'], inplace=True)

    # Filter to include only specific fuel types
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
    # Define the color mapping for adoption tiers
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }
    
    # Ensure the DataFrame is properly formatted
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex for columns")
    
    # ========== ADD THIS BLOCK FOR DESTENIE'S SORTING SUGGESTION ==========
    # Sort by IRA-Reference Total Adoption Potential
    try:
        ira_scenarios = [col for col in df.columns.get_level_values(0).unique() 
                        if 'iraref' in col.lower() or 'ira_ref' in col.lower() or 'ira-ref' in col.lower()]
        
        if ira_scenarios:
            ira_scenario = ira_scenarios[0]
            sort_column = (ira_scenario, 'Total Adoption Potential')
            
            if sort_column in df.columns:
                df = df.sort_values(sort_column, ascending=False)
    except Exception:
        pass
    # ========== END OF ADDITION ==========

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
                
                # Plot the stacked bars for the scenario
                ax.bar(
                    scenario_index, 
                    tier1, 
                    bar_width, 
                    color=color_mapping['Tier 1: Feasible'], 
                    edgecolor='white',
                    label='Tier 1: Feasible' if i == 0 else ""  # Only label once for legend
                )
                ax.bar(
                    scenario_index, 
                    tier2, 
                    bar_width, 
                    bottom=tier1, 
                    color=color_mapping['Tier 2: Feasible vs. Alternative'], 
                    edgecolor='white',
                    label='Tier 2: Feasible vs. Alternative' if i == 0 else ""
                )
                ax.bar(
                    scenario_index, 
                    tier3, 
                    bar_width, 
                    bottom=(tier1+tier2), 
                    color=color_mapping['Tier 3: Subsidy-Dependent Feasibility'], 
                    edgecolor='white',
                    label='Tier 3: Subsidy-Dependent Feasibility' if i == 0 else ""
                )
            else:
                print(f"Warning: Missing tier columns for scenario {scenario}")
                print(f"Available columns: {adoption_data.columns.tolist()}")
                
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


def print_adoption_decision_percentages(
    dataframes: List[pd.DataFrame],
    scenario_names: List[str],
    title: str = None,
    subtitle: Optional[str] = None,
    print_header_key: bool = True,
    filter_fuel: Optional[List[str]] = None,
) -> None:
    """
    Print adoption decision percentages using the same logic as create_multiIndex_adoption_df.
    
    Args:
        dataframes: List of DataFrames with multi-index structure from create_multiIndex_adoption_df
        scenario_names: List of scenario names (e.g., ['Pre-IRA', 'IRA-Reference'])
        title: Section title
        subtitle: Optional subtitle for the section
        print_header_key: Whether to print the header key for the output
        filter_fuel: Optional list of fuels to include (uses same filtering as existing module)
    """
    
    # Define the mapping based on the existing module's tier structure
    tier_mapping = {
        'Tier 1: Feasible': 'AD',
        'Total Adoption Potential': 'TAD', 
        'Total Adoption Potential (Additional Subsidy)': 'TADS',
    }
    
    header_key = """(Base Fuel, Income Level): 
    AD (%):   --> Tier 1 (%): Adopters that recover the total capital cost of retrofit
    TAD (%):  --> Tier 1+2 (%): Adopters that recover either the total or net capital cost of retrofit
    TADS (%): --> Tier 1+2+3 (%): Both less and more WTP Adopters plus those that require subsidies to adopt (positive total NPV)
    """

    # Print header (matching user's desired format)
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
    
    # Process each dataframe and collect results
    all_results = {}
    
    for df, scenario_name in zip(dataframes, scenario_names):
        # Apply fuel filtering using the same logic as the existing module
        if filter_fuel is not None:
            # Use the same fuel filtering logic as subplot_grid_adoption_vBar
            fuel_level_names = [name for name in df.index.names if 'fuel' in name.lower()]
            if fuel_level_names:
                fuel_level = fuel_level_names[0]
                df = df[df.index.get_level_values(fuel_level).isin(filter_fuel)]
        
        # Find the scenario columns in the MultiIndex columns
        scenario_columns = [col for col in df.columns.get_level_values(0).unique() 
                          if scenario_name.lower().replace('-', '').replace('_', '') in 
                             col.lower().replace('-', '').replace('_', '')]
        
        if not scenario_columns:
            print(f"Warning: No columns found for scenario '{scenario_name}' in DataFrame")
            continue
            
        scenario_col = scenario_columns[0]  # Take the first matching column
        
        # Calculate overall percentages using the same approach as the module
        overall_percentages = []
        for tier, abbr in tier_mapping.items():
            if (scenario_col, tier) in df.columns:
                # Calculate mean across all groups (same as overall calculation)
                overall_pct = df[(scenario_col, tier)].mean()
                overall_percentages.append(f"{abbr} {overall_pct:.0f}%")
        
        overall_key = "('Overall')"
        if overall_key not in all_results:
            all_results[overall_key] = []
        all_results[overall_key].append(", ".join(overall_percentages))
        
        # Calculate percentages for each group using the existing index structure
        for group_idx in df.index:
            # Handle both single-level and multi-level indices
            if isinstance(group_idx, tuple):
                fuel, income = group_idx
                group_key = f"('{fuel}', '{income}')"
            else:
                group_key = f"('{group_idx}')"
            
            group_percentages = []
            for tier, abbr in tier_mapping.items():
                if (scenario_col, tier) in df.columns:
                    value = df.loc[group_idx, (scenario_col, tier)]
                    group_percentages.append(f"{abbr} {value:.0f}%")
            
            if group_key not in all_results:
                all_results[group_key] = []
            all_results[group_key].append(", ".join(group_percentages))
    
    # Print results in the same order as the existing module (Overall first, then sorted groups)
    for group_key, scenario_results in all_results.items():
        combined_results = " | ".join(scenario_results)
        print(f"{group_key}: {combined_results}")
    
    print()  # Add blank line after section

# scc = 'central'
# rcm_model = 'inmap'
# cr_function = 'acs'

# print_adoption_decision_percentages(
#         dataframes=[
#             df_mi_basic_heating_adoption_inmap_acs, df_mi_basic_heating_adoption_inmap_acs,
#             ],
#         scenario_names=[
#             f'preIRA_mp8_heating_adoption_{scc}_{rcm_model}_{cr_function}',
#             f'iraRef_mp8_heating_adoption_{scc}_{rcm_model}_{cr_function}',
#             ],
#         title="Space Heating Air-Source Heat Pump (ASHP) Retrofit Scenario Comparison",
#         subtitle="Basic Retrofit (MP8): Central SCC|InMAP|ACS",
#         print_header_key=True,
#     )

# print_adoption_decision_percentages(
#         dataframes=[
#             df_mi_moderate_heating_adoption_inmap_acs, df_mi_moderate_heating_adoption_inmap_acs,
#             ],
#         scenario_names=[
#             f'preIRA_mp9_heating_adoption_{scc}_{rcm_model}_{cr_function}',
#             f'iraRef_mp9_heating_adoption_{scc}_{rcm_model}_{cr_function}',
#             ],
#         title=None,
#         subtitle="Moderate Retrofit (MP9): Central SCC|InMAP|ACS",
#         print_header_key=False,
#     )

# print_adoption_decision_percentages(
#         dataframes=[
#             df_mi_advanced_heating_adoption_inmap_acs, df_mi_advanced_heating_adoption_inmap_acs
#             ],
#         scenario_names=[
#             f'preIRA_mp10_heating_adoption_{scc}_{rcm_model}_{cr_function}',
#             f'iraRef_mp10_heating_adoption_{scc}_{rcm_model}_{cr_function}'
#             ],
#         title=None,
#         subtitle="Advanced Retrofit (MP10): Central SCC|InMAP|ACS",
#         print_header_key=False,
#     )
