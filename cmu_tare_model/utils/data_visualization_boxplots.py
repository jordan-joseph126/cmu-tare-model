import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Union
from matplotlib.ticker import FuncFormatter

from cmu_tare_model.constants import COLOR_MAP_FUEL

def thousands_formatter(
        x: float,
        pos: int) -> str:
    """Format numbers to use K for thousands automatically.
    
    Formats numeric values for axis labels, converting values >= 1000 to 
    use 'K' suffix (e.g., 2500 becomes '2.5K'). Used as a matplotlib 
    FuncFormatter for cleaner axis labels.
    
    Args:
        x: Numeric value to format
        pos: Position parameter required by matplotlib FuncFormatter (unused)
        
    Returns:
        Formatted string representation of the number
        
    Raises:
        TypeError: If x cannot be converted to a numeric value
    """
    # Validate input
    try:
        x_float = float(x)
    except (TypeError, ValueError):
        raise TypeError(f"Invalid input for formatting: {x}. Must be a numeric value.")
    
    # Format with 'K' for thousands
    if abs(x_float) >= 1000:
        return f'{x_float/1000:g}K'
    else:
        return f'{x_float:g}'


def create_subplot_grid_boxplot(
        df: Optional[pd.DataFrame] = None,
        dataframes: Optional[List[pd.DataFrame]] = None,
        dataframe_indices: Optional[List[int]] = None,
        subplot_positions: List[Tuple[int, int]] = None,
        y_cols: List[str] = None,
        category_col: Optional[str] = None,
        hue_col: str = 'base_fuel',
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        lower_percentile: float = 1,
        upper_percentile: float = 99,
        show_outliers: bool = False,
        include_zero: bool = True,
        subplot_titles: Optional[List[str]] = None,
        suptitle: Optional[str] = None,
        figure_size: Tuple[int, int] = (12, 10),
        sharex: bool = False,
        sharey: bool = False,
        show_legend: bool = True,
        show_xtick_labels: bool = True) -> plt.Figure:
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
        subplot_titles: Optional list of titles for each subplot
        suptitle: Super title for the entire figure
        figure_size: Size of the figure as (width, height) in inches
        sharex: Whether to share x-axes across subplots
        sharey: Whether to share y-axes across subplots
        show_legend: Whether to show the fuel type legend
        show_xtick_labels: Whether to show x-tick labels (useful for large grids)
        
    Returns:
        Matplotlib Figure object containing the grid of boxplots
        
    Raises:
        ValueError: If input parameters are invalid or inconsistent
        KeyError: If specified columns don't exist in the DataFrame(s)
    """
    # Use default palette
    palette = COLOR_MAP_FUEL
        
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
        axes = np.array([[axes]])  # Single subplot
    elif num_rows == 1:
        axes = np.array([axes])    # Single row, multiple columns
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])  # Multiple rows, single column

    # Create dictionary mapping subplot positions to axes
    subplot_axes = {(pos[0], pos[1]): axes[pos[0], pos[1]] for pos in subplot_positions}

    # Create boxplots for each subplot position
    for i, (pos, y_col) in enumerate(zip(subplot_positions, y_cols)):
        ax = subplot_axes[pos]
        
        # Get the appropriate DataFrame for this subplot
        current_df = df_list[df_indices[i]]
        
        # Check column existence
        if y_col not in current_df.columns:
            raise KeyError(f"Column '{y_col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)}")
        if category_col is not None and category_col not in current_df.columns:
            raise KeyError(f"Column '{category_col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)}")
        if hue_col not in current_df.columns:
            raise KeyError(f"Column '{hue_col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)}")

        # MEMORY FIX: Extract only needed columns, then apply filters
        required_cols = [y_col, hue_col]
        if category_col is not None:
            required_cols.append(category_col)
        df_plot = current_df[required_cols].copy()

        # Remove zero values if specified
        if not include_zero:
            df_plot = df_plot[df_plot[y_col] != 0]
        
        # Apply percentile filtering for numeric values
        lower_limit = df_plot[y_col].quantile(lower_percentile / 100)
        upper_limit = df_plot[y_col].quantile(upper_percentile / 100)
        df_plot = df_plot[(df_plot[y_col] >= lower_limit) & (df_plot[y_col] <= upper_limit)]
        
        # Get valid hue values that exist in both the data and palette
        hue_order = [val for val in df_plot[hue_col].unique() if val in palette]
              
        # Create the boxplot
        if category_col is not None:
            sns.boxplot(
                data=df_plot,
                x=category_col,
                y=y_col,
                hue=hue_col,
                hue_order=hue_order,
                palette=palette,
                showfliers=show_outliers,
                ax=ax
            )
        
        else:
            sns.boxplot(
                data=df_plot,
                x=hue_col,
                y=y_col,
                hue=hue_col,
                order=hue_order,
                hue_order=hue_order,
                palette=palette,
                showfliers=show_outliers,
                legend=False,
                ax=ax
            )

        # Add horizontal reference line at y=0
        ax.axhline(y=0, color='black', linestyle='--', linewidth=3, alpha=0.8, zorder=10)

        # Get labels
        x_label = x_labels[i] if x_labels and i < len(x_labels) else None
        y_label = y_labels[i] if y_labels and i < len(y_labels) else None
        
        # Set labels if provided
        if x_label is not None:
            ax.set_xlabel(x_label, fontsize=24)
        
        if y_label is not None:
            ax.set_ylabel(y_label, fontsize=24, fontweight='bold')
        
        # Set font size for tick labels
        ax.tick_params(axis='both', labelsize=24)

        # Hide x-tick labels if requested
        if show_xtick_labels:
            # Rotate x-tick labels conditionally
            ax.tick_params(axis='x', rotation=0)
        else:
            ax.set_xticklabels([])

        # Format y-axis labels to use K for thousands
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        
        # Remove top and right spines for cleaner appearance
        sns.despine()

    # Add column titles if provided
    if subplot_titles:
        for col_index, title in enumerate(subplot_titles):
            if col_index < num_cols:
                axes[0, col_index].set_title(title, fontsize=24, fontweight='bold')

    # Add super title if provided
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold', fontsize=24)

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
