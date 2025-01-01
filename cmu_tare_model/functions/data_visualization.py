import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from config import PROJECT_ROOT

"""
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
FUNCTIONS FOR DATA VISUALIZATION
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""
# ======================================================================================================================
# FORMAT DATA USING .DESCRIBE() METHODS
# ======================================================================================================================

def summarize_stats_table(df, data_columns, column_name_mapping, number_formatting, include_zero=True):
    """
    Generate a formatted summary statistics table for specified columns in a DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame from which to compute statistics.
    - data_columns (list of str): The columns to include in the summary statistics.
    - column_name_mapping (dict): A dictionary to rename the columns in the summary statistics output.
    - number_formatting (str): The format string to use for numeric values in the output.
    - include_zero (bool, optional): Whether to include zero values in the statistics. Defaults to True.
      If False, zeros are replaced with NaN, which are then ignored in the computations.

    Returns:
    - DataFrame: A DataFrame containing the summary statistics, with formatted numeric values
      and renamed columns according to the input specifications.
    """

    # Create a copy of the DataFrame to avoid modifying the original data
    df_copy = df.copy()

    # Replace 0 values with NaN in the selected columns if include_zero is set to False
    if not include_zero:
        df_copy[data_columns] = df_copy[data_columns].replace(0, np.nan)

    # Compute summary statistics for the selected columns
    # The 'describe' function returns summary statistics including count, mean, std, min, 25%, 50%, 75%, max
    # Apply formatting to each number in these statistics according to the given format
    summary_stats = df_copy[data_columns].describe().apply(lambda col: col.map(lambda x: f"{x:{number_formatting}}"))

    # Rename the columns in the summary statistics DataFrame according to the provided mapping
    summary_stats.rename(columns=column_name_mapping, inplace=True)

    return summary_stats

# ======================================================================================================================
# HISTOGRAMS
# ======================================================================================================================

# Added base fuel color-coded legend
# Possibly update colors to make more color blind accessible
color_map_fuel = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'firebrick',
}

# Define a function to plot the histogram and percentile subplot
def create_subplot_histogram(ax, df, x_col, bin_number, x_label=None, y_label=None, lower_percentile=2.5, upper_percentile=97.5, color_code='base_fuel', statistic='count', include_zero=False, show_legend=False):
    df_copy = df.copy()
    
    if not include_zero:
        df_copy[x_col] = df_copy[x_col].replace(0, np.nan)

    lower_limit = df_copy[x_col].quantile(lower_percentile / 100)
    upper_limit = df_copy[x_col].quantile(upper_percentile / 100)

    valid_data = df_copy[x_col][(df_copy[x_col] >= lower_limit) & (df_copy[x_col] <= upper_limit)]

    # Get the corresponding color for each fuel category
    colors = [color_map_fuel.get(fuel, 'gray') for fuel in df_copy[color_code].unique()]

    # Set the hue_order to match the unique fuel categories and their corresponding colors
    hue_order = [fuel for fuel in df_copy[color_code].unique() if fuel in color_map_fuel]

    ax = sns.histplot(data=df_copy, x=valid_data, kde=False, bins=bin_number, hue=color_code, hue_order=hue_order, stat=statistic, multiple="stack", palette=colors, ax=ax, legend=show_legend)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=22)  # Set font size for x-axis label

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=22)  # Set font size for y-axis label

    ax.set_xlim(left=lower_limit, right=upper_limit)

    # Set font size for tick labels
    ax.tick_params(axis='both', labelsize=22)

    sns.despine()

def create_subplot_grid_histogram(df, subplot_positions, x_cols, x_labels, y_label=None, bin_number=20, lower_percentile=2.5, upper_percentile=97.5, statistic='count', color_code='base_fuel', include_zero=False, suptitle=None, sharex=False, sharey=False, column_titles=None, show_legend=True, figure_size=(12, 10), export_filename=None, export_format='png', dpi=300):
    num_subplots = len(subplot_positions)
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)

    # Create a dictionary to map subplot positions to their respective axes
    subplot_axes = {(pos[0], pos[1]): axes[pos[0], pos[1]] for pos in subplot_positions}

    # Define the parameters for each histogram subplot
    plot_params = [{'ax': subplot_axes[pos], 'x_col': col, 'x_label': label, 'y_label': y_label, 'bin_number': bin_number, 'lower_percentile': lower_percentile, 'upper_percentile': upper_percentile, 'statistic': statistic, 'color_code': color_code, 'include_zero': include_zero, 'show_legend': show_legend}
                   for pos, col, label in zip(subplot_positions, x_cols, x_labels)]

    # Plot each histogram subplot using the defined parameters
    for params in plot_params:
        create_subplot_histogram(df=df, **params)

    # Add a super title to the entire figure if suptitle is provided
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')

    # Add titles over the columns
    if column_titles:
        for col_index, title in enumerate(column_titles):
            axes[0, col_index].set_title(title, fontsize=22, fontweight='bold')
    
    # If sharey is True, remove y-axis labels on all subplots except the leftmost ones in each row
    if sharey:
        for row_index in range(num_rows):
            for col_index in range(num_cols):
                if col_index > 0:
                    axes[row_index, col_index].set_yticklabels([])

    # Add a legend for the color mapping at the bottom of the entire figure
    legend_labels = list(color_map_fuel.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map_fuel[label]) for label in legend_labels]
    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), prop={'size': 22}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))             
    
    # Adjust the layout
    plt.tight_layout()
    
    # Export the figure if export_filename is provided
    if export_filename:
        save_figure_path = os.path.join(save_figure_directory, export_filename)
        plt.savefig(save_figure_path, format=export_format, dpi=dpi)
    # Otherwise show the plot in Jupyter Notebook
    else:
        plt.show()

# ======================================================================================================================
# CO2 EMISSIONS ABATEMENT (BOXPLOTS)
# ======================================================================================================================

# LAST UPDATED DECEMBER 9, 2024
def subplot_grid_co2_abatement(dataframes, subplot_positions, epa_scc_values, x_cols, y_cols, hues, plot_titles=None, x_labels=None, y_labels=None, suptitle=None, figure_size=(12, 10), sharex=False, sharey=False):
    """
    Creates a grid of subplots to visualize CO2 abatement cost effectiveness across different datasets and scenarios.
    """
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
    axes = np.array(axes).reshape(num_rows, num_cols)  # Ensure axes is always 2D

    for idx, (df, epa_scc, x_col, y_col, hue) in enumerate(zip(dataframes, epa_scc_values, x_cols, y_cols, hues)):
        pos = subplot_positions[idx]
        ax = axes[pos[0], pos[1]]
        title = plot_titles[idx] if plot_titles else ""
        x_label = x_labels[idx] if x_labels else ""
        y_label = y_labels[idx] if y_labels else ""

        # Plot using the plot_co2_abatement function, passing the current axis to it
        plot_co2_abatement(df, x_col, y_col, hue, epa_scc, ax=ax)

        # Set custom labels and title if provided
        ax.set_xlabel(x_label, fontweight='bold', fontsize=18)
        ax.set_ylabel(y_label, fontweight='bold', fontsize=18)
        ax.set_title(title, fontweight='bold', fontsize=18)

        # Set font size for tick labels on the x-axis
        ax.tick_params(axis='x', labelsize=18)

        # Set font size for tick labels on the y-axis
        ax.tick_params(axis='y', labelsize=18)

    if suptitle:
        plt.suptitle(suptitle, fontweight='bold')

    # Create a consolidated legend by grabbing handles and labels from all subplots
    handles, labels = [], []
    for ax in axes.flatten():
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:  # Avoid duplicates
                handles.append(handle)
                labels.append(label)

    # # Add the consolidated legend outside the plots
    # fig.legend(handles, labels, loc='lower center', ncol=5, prop={'size': 18}, labelspacing=0.25, bbox_to_anchor=(0.5, -0.01))

    # # Adjust the layout
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to leave space for the suptitle

    # Add the consolidated legend outside the plots
    fig.legend(handles, labels, loc='lower center', ncol=5, prop={'size': 16}, labelspacing=0.25, handletextpad=1, columnspacing=1, bbox_to_anchor=(0.5, -0.05), bbox_transform=fig.transFigure)

    # Fine-tune the layout adjustment if needed
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjusted the rect to leave space for the suptitle and legend

    plt.show()

def plot_co2_abatement(df, x_col, y_col, hue, EPA_SCC_USD2023_PER_TON, ax=None):
    """
    Plots a boxplot of CO2 abatement cost effectiveness.

    Parameters:
    - df: DataFrame containing the data.
    - x_col: Column name for the x-axis.
    - y_col: Column name for the y-axis.
    - hue: Column name for the hue (categorical variable for color).
    - EPA_SCC_USD2023_PER_TON: Value for the red dashed line indicating SCC.
    - ax: Axis object to plot on. If None, creates a new plot.
    
    Returns:
    - None: Displays the plot.
    """
    # Filter out the 'Middle-to-Upper-Income' rows and create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    df_filtered = df_copy[df_copy[x_col] != 'Middle-to-Upper-Income']

    # If x_col is categorical, remove unused categories
    if df_filtered[x_col].dtype.name == 'category':
        df_filtered.loc[:, x_col] = df_filtered[x_col].cat.remove_unused_categories()

    # Color map for fuel types
    color_map_fuel = {
        'Electricity': 'seagreen',
        'Natural Gas': 'steelblue',
        'Propane': 'orange',
        'Fuel Oil': 'firebrick',
    }

    if ax is None:
        ax = plt.gca()

    # Create the boxplot
    sns.boxplot(
        data=df_filtered,
        x=x_col, 
        y=y_col, 
        hue=hue, 
        palette=color_map_fuel, 
        showfliers=False,
        width=0.8,
        ax=ax
    )

    # Add a red dashed line at the value of EPA_SCC_USD2023_PER_TON
    ax.axhline(y=EPA_SCC_USD2023_PER_TON, color='red', linestyle='--', linewidth=2, label=f'SCC (USD2023): ${int(round((EPA_SCC_USD2023_PER_TON), 0))}/mtCO2e')

    # Remove the individual legend for each subplot
    ax.legend_.remove()