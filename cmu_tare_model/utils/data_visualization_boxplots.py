import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Union
from matplotlib.ticker import FuncFormatter

# Color mapping (keeping original style)
color_map_fuel = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'gray',  # Changed to gray for accessibility
}


def thousands_formatter(x: float, pos: int) -> str:
    """Format numbers to use K for thousands automatically.
    
    Formats numeric values for axis labels, converting values >= 1000 to 
    use 'K' suffix (e.g., 2500 becomes '2.5K'). Used as a matplotlib 
    FuncFormatter for cleaner axis labels.
    
    Args:
        x: Numeric value to format
        pos: Position parameter required by matplotlib FuncFormatter (unused)
        
    Returns:
        Formatted string representation of the number
    """
    # Validate input
    try:
        x_float = float(x)
    except (TypeError, ValueError):
        return str(x)
    
    if abs(x_float) >= 1000:
        return f'{x_float/1000:g}K'
    else:
        return f'{x_float:g}'


def create_subplot_boxplot(
    ax: plt.Axes, 
    df: pd.DataFrame, 
    y_col: str, 
    category_col: str, 
    hue_col: str,
    x_label: Optional[str] = None, 
    y_label: Optional[str] = None, 
    lower_percentile: float = 1, 
    upper_percentile: float = 99, 
    show_outliers: bool = False, 
    include_zero: bool = True,
    palette: Optional[Dict[str, str]] = None
) -> None:
    """Creates a boxplot on the provided axes using the specified DataFrame.
    
    This function creates vertical boxplots grouped by a category with colors based on a hue variable.
    It is designed for comparing distributions across different groups and categories.
    
    Args:
        ax: Matplotlib Axes object where the boxplot will be plotted
        df: DataFrame containing the data to plot
        y_col: Column name for y-axis (numeric data for vertical boxplots)
        category_col: Column name for categories (e.g., income level)
        hue_col: Column name for color grouping (e.g., fuel type)
        x_label: Optional label for x-axis
        y_label: Optional label for y-axis
        lower_percentile: Lower percentile for data range filtering (0-100)
        upper_percentile: Upper percentile for data range filtering (0-100)
        show_outliers: Whether to show outliers in the boxplot
        include_zero: Whether to include zero values in the visualization
        palette: Optional color mapping dictionary
        
    Raises:
        KeyError: If required columns don't exist in the DataFrame
    """
    # Minimal validation - check for column existence
    if y_col not in df.columns:
        raise KeyError(f"Column '{y_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    if category_col not in df.columns:
        raise KeyError(f"Column '{category_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    if hue_col not in df.columns:
        raise KeyError(f"Column '{hue_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Remove zero values if specified
    if not include_zero:
        df_copy = df_copy[df_copy[y_col] != 0]
    
    # Apply percentile filtering for numeric values
    lower_limit = df_copy[y_col].quantile(lower_percentile / 100)
    upper_limit = df_copy[y_col].quantile(upper_percentile / 100)
    df_copy = df_copy[(df_copy[y_col] >= lower_limit) & (df_copy[y_col] <= upper_limit)]
    
    # Use default palette if none provided
    if palette is None:
        palette = color_map_fuel
    
    # Get valid hue values that exist in both the data and palette
    hue_order = [val for val in df_copy[hue_col].unique() if val in palette]
    
    # Create the boxplot
    sns.boxplot(
        data=df_copy, 
        x=category_col, 
        y=y_col, 
        hue=hue_col,
        hue_order=hue_order,
        palette=palette,
        showfliers=show_outliers, 
        ax=ax
    )
    
    # Set labels if provided
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=22)
    
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=22)
    
    # Set font size for tick labels
    ax.tick_params(axis='both', labelsize=22)
    
    # Format y-axis labels to use K for thousands
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    # Remove top and right spines for cleaner appearance
    sns.despine()


def create_subplot_grid_boxplot(
    df: Optional[pd.DataFrame] = None,                    # Single DataFrame (backward compatible)
    dataframes: Optional[List[pd.DataFrame]] = None,      # Multiple DataFrames
    dataframe_indices: Optional[List[int]] = None,        # DataFrame mapping
    subplot_positions: List[Tuple[int, int]] = None, 
    y_cols: List[str] = None,                             # Numeric variables for y-axis
    category_col: str = 'income_level',                   # Default column for categories (x-axis)
    hue_col: str = 'base_fuel',                           # Default column for color coding
    x_labels: Optional[List[str]] = None, 
    y_labels: Optional[List[str]] = None, 
    lower_percentile: float = 1, 
    upper_percentile: float = 99, 
    show_outliers: bool = False, 
    include_zero: bool = True, 
    suptitle: Optional[str] = None, 
    sharex: bool = False, 
    sharey: bool = False, 
    subplot_titles: Optional[List[str]] = None, 
    show_legend: bool = True,                            
    figure_size: Tuple[int, int] = (12, 10),
    palette: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """Creates a grid of boxplots with multiple fuel types grouped by category.
    
    This function creates a customizable grid of boxplots specifically designed for comparing
    distributions across different categories (e.g., income levels) and groups (e.g., fuel types).
    
    Args:
        df: Single DataFrame for backward compatibility
        dataframes: List of DataFrames for multi-DataFrame visualizations
        dataframe_indices: List of indices mapping subplot positions to DataFrames
        subplot_positions: List of (row, column) tuples for subplot placement
        y_cols: List of column names for y-axis (numeric data) in each subplot
        category_col: Column name for categories on x-axis (e.g., income level)
        hue_col: Column name for color coding (e.g., fuel type)
        x_labels: Optional list of x-axis labels for each subplot
        y_labels: Optional list of y-axis labels for each subplot
        lower_percentile: Lower percentile for data range filtering (0-100)
        upper_percentile: Upper percentile for data range filtering (0-100)
        show_outliers: Whether to show outliers in the boxplots
        include_zero: Whether to include zero values in the visualization
        suptitle: Super title for the entire figure
        sharex: Whether to share x-axes across subplots
        sharey: Whether to share y-axes across subplots
        subplot_titles: Optional list of titles for each column
        show_legend: Whether to show the fuel type legend
        figure_size: Size of the figure as (width, height) in inches
        palette: Optional color mapping dictionary (defaults to color_map_fuel)
        
    Returns:
        Matplotlib Figure object containing the grid of boxplots
        
    Raises:
        ValueError: If input parameters are invalid or inconsistent
        KeyError: If specified columns don't exist in the DataFrame(s)
    """
    # Use default palette if none provided
    if palette is None:
        palette = color_map_fuel
        
    # Simple input validation
    if df is not None and dataframes is not None:
        raise ValueError("Specify either 'df' or 'dataframes', not both")
    if df is None and dataframes is None:
        raise ValueError("Must specify either 'df' or 'dataframes'")
    if subplot_positions is None or not subplot_positions:
        raise ValueError("subplot_positions must be provided and non-empty")
    if y_cols is None or not y_cols:
        raise ValueError("y_cols must be provided and non-empty")
    if len(subplot_positions) != len(y_cols):
        raise ValueError(f"Number of subplot positions ({len(subplot_positions)}) must match number of y columns ({len(y_cols)})")
    
    # Set up DataFrame handling
    if df is not None:
        # Single DataFrame mode (original behavior)
        df_list = [df]
        df_indices = [0] * len(subplot_positions)
    else:
        # Multiple DataFrame mode
        df_list = dataframes
        if dataframe_indices is None:
            df_indices = [i % len(dataframes) for i in range(len(subplot_positions))]
        else:
            if len(dataframe_indices) != len(subplot_positions):
                raise ValueError("dataframe_indices length must match number of subplots")
            df_indices = dataframe_indices
    
    # Grid setup logic
    num_subplots = len(subplot_positions)
    num_cols = max(pos[1] for pos in subplot_positions) + 1
    num_rows = max(pos[0] for pos in subplot_positions) + 1

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figure_size, sharex=sharex, sharey=sharey)

    # Ensure axes is always 2D for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])  # Single subplot
    elif num_rows == 1:
        axes = np.array([axes])    # Single row, multiple columns
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])  # Multiple rows, single column

    # Create dictionary mapping subplot positions to axes
    subplot_axes = {(pos[0], pos[1]): axes[pos[0], pos[1]] for pos in subplot_positions}

    # Create boxplots for each subplot position
    for i, (pos, y_col) in enumerate(zip(subplot_positions, y_cols)):
        # Get the appropriate DataFrame for this subplot
        current_df = df_list[df_indices[i]]
        
        # Check column existence
        if y_col not in current_df.columns:
            raise KeyError(f"Column '{y_col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)}")
        if category_col not in current_df.columns:
            raise KeyError(f"Column '{category_col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)}")
        if hue_col not in current_df.columns:
            raise KeyError(f"Column '{hue_col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)}")
        
        # Get labels
        x_label = x_labels[i] if x_labels and i < len(x_labels) else None
        y_label = y_labels[i] if y_labels and i < len(y_labels) else None
        
        # Create boxplot
        create_subplot_boxplot(
            ax=subplot_axes[pos],
            df=current_df,
            y_col=y_col,
            category_col=category_col,
            hue_col=hue_col,
            x_label=x_label,
            y_label=y_label if pos[1] == 0 or not sharey else None,  # Only show y-label on leftmost plots if sharey
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            show_outliers=show_outliers,
            include_zero=include_zero,
            palette=palette
        )
        
        # If this is the last subplot and show_legend is False, remove the legend
        if not show_legend:
            if subplot_axes[pos].get_legend() is not None:
                subplot_axes[pos].get_legend().remove()

    # Add column titles if provided
    if subplot_titles:
        for col_index, title in enumerate(subplot_titles):
            if col_index < num_cols:
                axes[0, col_index].set_title(title, fontsize=22, fontweight='bold')

    # Add super title if provided
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold', fontsize=22)

    # Handle legend - only keep one legend for the entire figure
    if show_legend:
        # Remove individual subplot legends
        for pos in subplot_positions:
            if subplot_axes[pos].get_legend() is not None:
                subplot_axes[pos].get_legend().remove()
        
        # Create a consolidated legend at the bottom of the figure
        legend_labels = list(palette.keys())
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=palette[label]) for label in legend_labels]
        fig.legend(
            legend_handles, 
            legend_labels, 
            loc='lower center', 
            ncol=len(legend_labels), 
            prop={'size': 22}, 
            labelspacing=0.5, 
            bbox_to_anchor=(0.5, -0.05)  # Space between legend and x-ticks
        )
        # Adjust layout with room for legend
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Reserve space for legend and title
    else:
        # Adjust layout without needing space for legend
        plt.tight_layout(rect=[0, 0, 1, 0.95] if suptitle else [0, 0, 1, 1])
    
    return fig


# Example usage of the create_subplot_grid_boxplot function
'''
fig_climate_damages = create_subplot_grid_boxplot(
    df=df_outputs_basic_home_easiur,  # Single DataFrame with all data
    subplot_positions=[(0, 0), (0, 1), (0, 2)],
    y_cols=[
        f'{scenario_prefix}{category}_avoided_damages_climate_lrmer_lower',
        f'{scenario_prefix}{category}_avoided_damages_climate_lrmer_central',
        f'{scenario_prefix}{category}_avoided_damages_climate_lrmer_upper'
    ],
    category_col='lmi_or_mui',
    hue_col=f'base_{category}_fuel',
    sharex=True,
    sharey=True,
    subplot_titles=['Lower Bound', 'Central Estimate', 'Upper Bound'],
    x_labels=['', '', ''],
    y_labels=['Avoided Lifetime Climate Damages [$USD-2023]', '', ''],
    figure_size=(18, 12),
    show_outliers=False,
    palette=color_map_fuel
)
'''
