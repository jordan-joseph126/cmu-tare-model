import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS: VISUALIZATION USING DATAFRAMES AND SUBPLOTS
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

# LAST UPDATED DECEMBER 26, 2024 @ 9:10 PM
def create_df_adoption(df, menu_mp, category):
    """
    Generates a new DataFrame with specific adoption columns based on provided parameters.
    
    Args:
    df (pd.DataFrame): Original DataFrame.
    menu_mp (int): Measure package identifier.

    Returns:
    pd.DataFrame: A DataFrame with the selected columns.

    UPDATES:
    - Removed reference to EPA_SCC_USD2023_PER_MT as it is a constant and doesnt need to be a dataframe column.
    - Removed code that calculates the cost effectiveness of CO2e abatement for each equipment category. (now done in adoption_decision)

    """    
    # Create a copy of the dataframe
    df_copy = df.copy()

    # Begin df with these cols
    summary_cols = ['state', 'city', 'county', 'puma', 'percent_AMI', 'lowModerateIncome_designation']

    cols_to_add = [f'base_{category}_fuel',
                   f'preIRA_mp{menu_mp}_{category}_private_npv_lessWTP', # PRE-IRA PRIVATE
                   f'preIRA_mp{menu_mp}_{category}_total_capitalCost', 
                   f'preIRA_mp{menu_mp}_{category}_private_npv_moreWTP', 
                   f'preIRA_mp{menu_mp}_{category}_net_capitalCost',
                   f'preIRA_mp{menu_mp}_{category}_avoided_mt_co2e_lrmer', # LRMER
                   f'preIRA_mp{menu_mp}_{category}_public_npv_lrmer',
                   f'preIRA_mp{menu_mp}_{category}_adoption_lrmer',
                   f'preIRA_mp{menu_mp}_{category}_avoided_mt_co2e_srmer', # SRMER
                   f'preIRA_mp{menu_mp}_{category}_public_npv_srmer',
                   f'preIRA_mp{menu_mp}_{category}_adoption_srmer',
                   f'mp{menu_mp}_{category}_rebate_amount', # IRA-REFERENCE PRIVATE
                   f'iraRef_mp{menu_mp}_{category}_private_npv_lessWTP', 
                   f'iraRef_mp{menu_mp}_{category}_total_capitalCost', 
                   f'iraRef_mp{menu_mp}_{category}_private_npv_moreWTP', 
                   f'iraRef_mp{menu_mp}_{category}_net_capitalCost',
                   f'iraRef_mp{menu_mp}_{category}_avoided_mt_co2e_lrmer', # LRMER
                   f'iraRef_mp{menu_mp}_{category}_usd2023_per_mtCO2e_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_public_npv_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_additional_public_benefit_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_adoption_lrmer',
                   f'iraRef_mp{menu_mp}_{category}_avoided_mt_co2e_srmer', # SRMER
                   f'iraRef_mp{menu_mp}_{category}_usd2023_per_mtCO2e_srmer',
                   f'iraRef_mp{menu_mp}_{category}_public_npv_srmer',
                   f'iraRef_mp{menu_mp}_{category}_additional_public_benefit_srmer',
                   f'iraRef_mp{menu_mp}_{category}_adoption_srmer',
                   ]
            
    # Use extend instead of append to add each element of cols_to_add to summary_cols
    summary_cols.extend(cols_to_add)

    # Select the relevant columns
    df_copy = df_copy[summary_cols]

    return df_copy

# UPDATED SEPTEMBER 20, 2024 @ 1:30 AM
import pandas as pd

def filter_columns(df):
    keep_columns = [col for col in df.columns if 'Tier 1: Feasible' in col[1] or 
                    'Tier 2: Feasible vs. Alternative' in col[1] or 
                    'Tier 3: Subsidy-Dependent Feasibility' in col[1] or 
                    'Total Adoption Potential' in col[1] or 
                    'Total Adoption Potential (Additional Subsidy)' in col[1]]    
    
    return df.loc[:, keep_columns]

def create_multiIndex_adoption_df(df, menu_mp, category, mer_type):
    # Explicitly set 'lowModerateIncome_designation' as a categorical type with order
    income_categories = ['Low-Income', 'Moderate-Income', 'Middle-to-Upper-Income']

    df['lowModerateIncome_designation'] = pd.Categorical(df['lowModerateIncome_designation'], categories=income_categories, ordered=True)
    
    # Define the columns for adoption data
    adoption_cols = [f'preIRA_mp{menu_mp}_{category}_adoption_{mer_type}', 
                     f'iraRef_mp{menu_mp}_{category}_adoption_{mer_type}']

    # Group by f'base_{category}_fuel' and 'lowModerateIncome_designation', calculate normalized counts
    percentages_df = df.groupby([f'base_{category}_fuel', 'lowModerateIncome_designation'], observed=False)[adoption_cols].apply(
        lambda x: x.apply(lambda y: y.value_counts(normalize=True))).unstack().fillna(0) * 100
    percentages_df = percentages_df.round(0)

    # Ensure 'Tier 1: Feasible' columns exist, set to 0 if they don't
    for column in adoption_cols:
        if (column, 'Tier 1: Feasible') not in percentages_df.columns:
            percentages_df[(column, 'Tier 1: Feasible')] = 0
        if (column, 'Tier 2: Feasible vs. Alternative') not in percentages_df.columns:
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')] = 0
        if (column, 'Tier 3: Subsidy-Dependent Feasibility') not in percentages_df.columns:
            percentages_df[(column, 'Tier 3: Subsidy-Dependent Feasibility')] = 0

        percentages_df[(column, 'Total Adoption Potential')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')]
        )

        percentages_df[(column, 'Total Adoption Potential (Additional Subsidy)')] = (
            percentages_df[(column, 'Tier 1: Feasible')] + 
            percentages_df[(column, 'Tier 2: Feasible vs. Alternative')] + 
            percentages_df[(column, 'Tier 3: Subsidy-Dependent Feasibility')]
        )

    # Rebuild the column MultiIndex
    percentages_df.columns = pd.MultiIndex.from_tuples(percentages_df.columns)
    
    # Filter DataFrame to keep relevant columns only
    filtered_df = filter_columns(percentages_df)

    new_order = []
    for prefix in ['preIRA_mp', 'iraRef_mp']:
        for suffix in ['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility', 'Total Adoption Potential', 'Total Adoption Potential (Additional Subsidy)']:
            col = (f'{prefix}{menu_mp}_{category}_adoption_{mer_type}', suffix)
            if col in filtered_df.columns:
                new_order.append(col)

    # Check if new_order is empty before reordering columns
    if new_order:
        # Reorder columns based on new_order
        filtered_df = filtered_df.loc[:, pd.MultiIndex.from_tuples(new_order)]
                    
        # Sort DataFrame by the entire index
        filtered_df.sort_index(level=[f'base_{category}_fuel', 'lowModerateIncome_designation'], inplace=True)
    else:
        print("Warning: No matching columns found for reordering")

    return filtered_df

# Usage example (assuming df_basic_adoption_heating is properly formatted and loaded):
# df_multiIndex_heating_adoption = create_multiIndex_adoption_df(df_basic_adoption_heating, 8, 'heating')
# df_multiIndex_heating_adoption

# UPDATED SEPTEMBER 14, 2024 @ 12:46 AM
def subplot_grid_adoption_vBar(dataframes, scenarios_list, subplot_positions, filter_fuel=None, x_labels=None, plot_titles=None, y_labels=None, suptitle=None, figure_size=(12, 10), sharex=False, sharey=False):
    """
    Creates a grid of subplots to visualize adoption rates across different scenarios, with an option to plot specific data related to adoption.

    Parameters:
    - dataframes (list of pd.DataFrame): List of pandas DataFrames, each DataFrame is assumed to be formatted for use in plot_adoption_rate_bar.
    - scenarios_list (list of list): List of scenarios corresponding to each DataFrame.
    - subplot_positions (list of tuples): Positions of subplots in the grid, specified as (row, col) tuples.
    - filter_fuel (list of str, optional): List of fuel types to filter the DataFrames by 'base_fuel' column in a multi-index.
    - x_labels (list of str, optional): Labels for the x-axis of each subplot.
    - plot_titles (list of str, optional): Titles for each subplot.
    - y_labels (list of str, optional): Labels for the y-axis of each subplot.
    - suptitle (str, optional): A central title for the entire figure.
    - figure_size (tuple, optional): Size of the entire figure (width, height) in inches.
    - sharex (bool, optional): Whether subplots should share the same x-axis.
    - sharey (bool, optional): Whether subplots should share the same y-axis.

    Returns:
    None. Displays the figure based on the provided parameters.
    """
    # Define the color mapping as specified
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
    axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure axes is always 2D

    for idx, (df, scenarios) in enumerate(zip(dataframes, scenarios_list)):
        # Apply the filter_fuel if provided
        if filter_fuel:
            df = df.loc[(df.index.get_level_values('base_fuel').isin(filter_fuel)), :]
        
        pos = subplot_positions[idx]
        ax = axes[pos[0], pos[1]]
        x_label = x_labels[idx] if x_labels else ""
        y_label = y_labels[idx] if y_labels else ""
        title = plot_titles[idx] if plot_titles else ""

        plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax)

    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_mapping.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[label]) for label in legend_labels]
            
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), prop={'size': 20}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))

    # Adjust the layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to leave space for the suptitle
    plt.show()

def plot_adoption_rate_bar(df, scenarios, title, x_label, y_label, ax):
    # Assume the DataFrame 'df' has a suitable structure, similar to earlier examples
    adoption_data = df.loc[:, df.columns.get_level_values(1).isin(['Tier 1: Feasible', 'Tier 2: Feasible vs. Alternative', 'Tier 3: Subsidy-Dependent Feasibility'])]
    adoption_data.columns = adoption_data.columns.remove_unused_levels()

    # Define the color mapping as specified
    global color_mapping
    color_mapping = {
        'Tier 1: Feasible': 'steelblue',
        'Tier 2: Feasible vs. Alternative': 'lightblue',
        'Tier 3: Subsidy-Dependent Feasibility': 'lightsalmon'
    }

    # Plotting logic
    n = len(adoption_data.index)
    bar_width = 0.35  # Width of bars
    index = list(range(n))  # Base index for bars

    for i, scenario in enumerate(scenarios):
        if (scenario, 'Tier 1: Feasible') in adoption_data.columns and (scenario, 'Tier 2: Feasible vs. Alternative') in adoption_data.columns and (scenario, 'Tier 3: Subsidy-Dependent Feasibility') in adoption_data.columns:
            tier1 = adoption_data[scenario, 'Tier 1: Feasible'].values
            tier2 = adoption_data[scenario, 'Tier 2: Feasible vs. Alternative'].values
            tier3 = adoption_data[scenario, 'Tier 3: Subsidy-Dependent Feasibility'].values

            # Adjust the index for this scenario
            scenario_index = np.array(index) + i * bar_width
            
            # Plot the bars for the scenario
            ax.bar(scenario_index, tier1, bar_width, color=color_mapping['Tier 1: Feasible'], edgecolor='white')
            ax.bar(scenario_index, tier2, bar_width, bottom=tier1, color=color_mapping['Tier 2: Feasible vs. Alternative'], edgecolor='white')
            ax.bar(scenario_index, tier3, bar_width, bottom=(tier1+tier2), color=color_mapping['Tier 3: Subsidy-Dependent Feasibility'], edgecolor='white')


    ax.set_xlabel(x_label, fontweight='bold', fontsize=20)
    ax.set_ylabel(y_label, fontweight='bold', fontsize=20)
    ax.set_title(title, fontweight='bold', fontsize=20)
    
    ax.set_xticks([i + bar_width / 2 for i in range(n)])
    ax.set_xticklabels([f'{name[1]}' for name in adoption_data.index.tolist()], rotation=90, ha='right')

    # Set font size for tick labels on the x-axis
    ax.tick_params(axis='x', labelsize=20)

    # Set font size for tick labels on the y-axis
    ax.tick_params(axis='y', labelsize=20)

    # Set y-ticks from 0 to 100 in steps of 10%
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_ylim(0, 100)
