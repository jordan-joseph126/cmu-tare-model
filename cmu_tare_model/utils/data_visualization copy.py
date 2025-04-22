import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ===================================================================================================================================================================================
# FUNCTIONS FOR DATA VISUALIZATION
# ===================================================================================================================================================================================
# FOR PUBLIC CLIMATE AND HEALTH IMPACTS NPV AND TOTAL NPV SENSITIVITY ANALYSIS
# ===================================================================================================================================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the updated visualization functions
# from histogram_visualization import create_subplot_grid_histogram

def iraRef_public_npv_sensitivity_grid(df, menu_mp=8, category='heating', scc='upper'):
    """
    Creates a grid of histograms comparing PUBLIC NPV values across different
    RCM models (AP2, EASIUR, InMAP) and CR functions (acs, h6c).
    
    Parameters:
    df: DataFrame containing the data
    menu_mp: Measure package identifier (default: 8)
    category: Equipment category (default: 'heating')
    scc: SCC assumption to use (default: 'upper')
    """
    # Set up subplot positions:
    # - Top row (acs): columns for AP2, EASIUR, InMAP
    # - Bottom row (h6c): columns for AP2, EASIUR, InMAP
    subplot_positions = [
        (0,0), (0,1), (0,2),  # Top row (acs)
        (1,0), (1,1), (1,2)   # Bottom row (h6c)
    ]
    
    # Define column names with sensitivity parameters
    x_cols = [
        # Top row (acs)
        'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_AP2_acs',
        'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_EASIUR_acs',
        'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_InMAP_acs',
        # Bottom row (h6c)
        'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_AP2_h6c',
        'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_EASIUR_h6c',
        'iraRef_mp{menu_mp}_{category}_public_npv_{scc}_InMAP_h6c'
    ]
    
    # Define x-axis labels for each subplot
    x_labels = [
        # Top row (acs)
        'Public NPV (AP2, acs)',
        'Public NPV (EASIUR, acs)',
        'Public NPV (InMAP, acs)',
        # Bottom row (h6c)
        'Public NPV (AP2, h6c)',
        'Public NPV (EASIUR, h6c)',
        'Public NPV (InMAP, h6c)'
    ]
    
    # Create the visualization
    create_subplot_grid_histogram(
        df=df,
        menu_mp=menu_mp,
        category=category,
        scc=scc,
        subplot_positions=subplot_positions,
        x_cols=x_cols,
        x_labels=x_labels,
        y_label='Count',
        bin_number=30,
        suptitle=f'IRA Reference Public NPV Sensitivity Analysis ({category.capitalize()})',
        column_titles=["AP2 Model", "EASIUR Model", "InMAP Model"],
        color_code=f'base_{category}_fuel',
        include_zero=False,
        figure_size=(18, 12),
        sharex='col',  # Share x-axis within columns
        sharey='row'   # Share y-axis within rows
    )

# # Example usage:
# # iraRef_public_npv_sensitivity_grid(df_euss_am_mp8_home_ap2, category='heating')

# # To run the analysis for different equipment categories:
# def run_all_equipment_sensitivity(df, menu_mp=8, scc='upper'):
#     """
#     Runs the sensitivity grid visualization for all equipment categories
    
#     Parameters:
#     df: DataFrame containing the data
#     menu_mp: Measure package identifier (default: 8)
#     scc: SCC assumption to use (default: 'upper')
#     """
#     for category in ['heating', 'waterHeating', 'clothesDrying', 'cooking']:
#         print(f"Creating visualization for {category}...")
#         iraRef_public_npv_sensitivity_grid(df, menu_mp, category, scc)

# Example:
# run_all_equipment_sensitivity(df_euss_am_mp8_home_ap2)

# Implementation of create_subplot_grid_histogram with sensitivity parameters
# (copy from the updated histogram_visualization.py file)

# Added base fuel color-coded legend
color_map_fuel = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'firebrick',
}

# Define a function to plot the histogram and percentile subplot with sensitivity parameters
def create_subplot_histogram(ax, df, x_col, bin_number, x_label=None, y_label=None, lower_percentile=2.5, 
                            upper_percentile=97.5, color_code='base_fuel', statistic='count', 
                            include_zero=False, show_legend=False, category=None, scc='upper', rcm_model='AP2', cr_function='h6c', menu_mp=8):
    df_copy = df.copy()
    
    # If the x_col contains sensitivity placeholders, replace them
    if '{menu_mp}' in x_col:
        x_col = x_col.replace('{menu_mp}', str(menu_mp))
    if '{category}' in x_col:
        x_col = x_col.replace('{category}', category)
    if '{scc}' in x_col:
        x_col = x_col.replace('{scc}', scc)
        
    # If the column doesn't exist, try fallback options
    if x_col not in df_copy.columns:
        # Define potential fallback patterns
        fallback_patterns = [
            x_col.replace(f'_{scc}', ''),
            x_col.replace(f'_public_npv_{scc}', '_public_npv'),
            x_col.replace(f'_public_npv_{scc}', '_climate_npv'),
            x_col.replace(f'mp{menu_mp}_', '')
        ]
        
        for fallback in fallback_patterns:
            if fallback in df_copy.columns:
                print(f"Using fallback column: {fallback} instead of {x_col}")
                x_col = fallback
                break
        
        # If still not found, draw an empty plot with a message
        if x_col not in df_copy.columns:
            ax.text(0.5, 0.5, f"Column not found: {x_col}", ha='center', va='center')
            if x_label is not None:
                ax.set_xlabel(x_label, fontsize=22)
            if y_label is not None:
                ax.set_ylabel(y_label, fontsize=22)
            ax.tick_params(axis='both', labelsize=22)
            sns.despine()
            return
    
    # If color_code includes category placeholders, replace them
    if isinstance(color_code, str) and '{category}' in color_code and category is not None:
        color_code = color_code.replace('{category}', category)
    
    # Handle missing color code column
    if color_code not in df_copy.columns:
        # Try to find a base fuel column
        if category is not None:
            potential_fuel_col = f'base_{category}_fuel'
            if potential_fuel_col in df_copy.columns:
                color_code = potential_fuel_col
                print(f"Using {color_code} for color coding")
            else:
                # No suitable column found, use a default color
                color_code = None
                print(f"No suitable fuel column found for color coding. Using default color.")
    
    if not include_zero:
        df_copy[x_col] = df_copy[x_col].replace(0, np.nan)

    # Skip if all values are NaN after filtering
    if df_copy[x_col].isna().all():
        ax.text(0.5, 0.5, "No valid data after filtering", ha='center', va='center')
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=22)
        if y_label is not None:
            ax.set_ylabel(y_label, fontsize=22)
        ax.tick_params(axis='both', labelsize=22)
        sns.despine()
        return

    lower_limit = df_copy[x_col].quantile(lower_percentile / 100)
    upper_limit = df_copy[x_col].quantile(upper_percentile / 100)

    valid_data = df_copy[x_col][(df_copy[x_col] >= lower_limit) & (df_copy[x_col] <= upper_limit)]

    # If no color coding or color_code column doesn't exist
    if color_code is None or color_code not in df_copy.columns:
        ax = sns.histplot(data=df_copy, x=valid_data, kde=False, bins=bin_number, 
                         stat=statistic, color='steelblue', ax=ax)
    else:
        # Get the corresponding color for each fuel category
        colors = [color_map_fuel.get(fuel, 'gray') for fuel in df_copy[color_code].unique()]

        # Set the hue_order to match the unique fuel categories and their corresponding colors
        hue_order = [fuel for fuel in df_copy[color_code].unique() if fuel in color_map_fuel]

        ax = sns.histplot(data=df_copy, x=valid_data, kde=False, bins=bin_number, 
                         hue=color_code, hue_order=hue_order, stat=statistic, 
                         multiple="stack", palette=colors, ax=ax, legend=show_legend)

    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=22)  # Set font size for x-axis label

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=22)  # Set font size for y-axis label

    ax.set_xlim(left=lower_limit, right=upper_limit)

    # Set font size for tick labels
    ax.tick_params(axis='both', labelsize=22)

    sns.despine()

def create_subplot_grid_histogram(df, subplot_positions, x_cols, x_labels, menu_mp=8, category=None, 
                                 scc='upper', rcm_model='AP2', cr_function='h6c', 
                                 y_label=None, bin_number=20, lower_percentile=2.5, upper_percentile=97.5, 
                                 statistic='count', color_code='base_fuel', include_zero=False, 
                                 suptitle=None, sharex=False, sharey=False, column_titles=None, 
                                 show_legend=True, figure_size=(12, 10), export_filename=None, 
                                 export_format='png', dpi=300, save_figure_directory=None):
    num_subplots = len(subplot_positions)
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)
    
    # Make sure axes is a 2D array even if there's only one row or column
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Create a dictionary to map subplot positions to their respective axes
    subplot_axes = {(pos[0], pos[1]): axes[pos[0], pos[1]] for pos in subplot_positions}

    # If category is provided and color_code is 'base_fuel', make it category-specific
    if category is not None and color_code == 'base_fuel':
        color_code = f'base_{category}_fuel'

    # Define the parameters for each histogram subplot
    plot_params = []
    for pos, col, label in zip(subplot_positions, x_cols, x_labels):
        params = {
            'ax': subplot_axes[pos], 
            'x_col': col, 
            'x_label': label, 
            'y_label': y_label, 
            'bin_number': bin_number, 
            'lower_percentile': lower_percentile, 
            'upper_percentile': upper_percentile, 
            'statistic': statistic, 
            'color_code': color_code, 
            'include_zero': include_zero, 
            'show_legend': show_legend,
            'category': category,
            'scc': scc,
            'rcm_model': rcm_model,
            'cr_function': cr_function,
            'menu_mp': menu_mp
        }
        plot_params.append(params)

    # Plot each histogram subplot using the defined parameters
    for params in plot_params:
        create_subplot_histogram(df=df, **params)

    # Add a super title to the entire figure if suptitle is provided
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold', fontsize=24)

    # Add titles over the columns
    if column_titles:
        for col_index, title in enumerate(column_titles):
            if col_index < num_cols:
                axes[0, col_index].set_title(title, fontsize=22, fontweight='bold')
    
    # Add row titles on the left side
    if num_rows >= 2:
        fig.text(0.01, 0.75, "ACS Function", fontsize=22, fontweight='bold', rotation=90, va='center')
        fig.text(0.01, 0.25, "H6C Function", fontsize=22, fontweight='bold', rotation=90, va='center')
    
    # If sharey is True, remove y-axis labels on all subplots except the leftmost ones in each row
    if sharey == 'row':
        for row_index in range(num_rows):
            for col_index in range(num_cols):
                if col_index > 0:
                    axes[row_index, col_index].set_ylabel('')
    
    # Only add the legend if there are valid color mappings
    if show_legend and color_code is not None and color_code in df.columns:
        # Get the unique fuel values that are actually in the data
        unique_fuels = df[color_code].unique()
        # Filter to only include fuels that are in our color map
        legend_labels = [fuel for fuel in unique_fuels if fuel in color_map_fuel]
        
        if legend_labels:
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map_fuel[label]) for label in legend_labels]
            fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), 
                      prop={'size': 22}, labelspacing=0.5, bbox_to_anchor=(0.5, -0.05))             
    
    # Adjust the layout
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])  # Adjust for suptitle, row labels, and legend
    
    # Export the figure if export_filename is provided
    if export_filename:
        if save_figure_directory:
            save_figure_path = os.path.join(save_figure_directory, export_filename)
        else:
            save_figure_path = export_filename
        plt.savefig(save_figure_path, format=export_format, dpi=dpi)
    # Otherwise show the plot in Jupyter Notebook
    else:
        plt.show()
