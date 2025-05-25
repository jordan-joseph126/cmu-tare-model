import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any, Union

"""
Enhanced Histogram Visualization Module with Multiple DataFrame Support

This module provides enhanced histogram visualization functions following the same
architectural patterns as the adoption potential visualization refactoring, now
with support for multiple DataFrames to enable complex sensitivity analyses.

Key improvements:
- Multiple DataFrame support with flexible mapping
- Comprehensive type annotations
- Robust error handling with informative messages
- Backward compatibility with single DataFrame usage
- Flexible parameter design
- Proper figure management and return types
"""

# ===================================================================================================================================================================================
# ENHANCED HISTOGRAM FUNCTIONS WITH MULTIPLE DATAFRAME SUPPORT
# ===================================================================================================================================================================================

# Color mapping for fuel types (moved from global to function parameter with default)
DEFAULT_COLOR_MAP_FUEL = {
    'Electricity': 'seagreen',
    'Natural Gas': 'steelblue',
    'Propane': 'orange',
    'Fuel Oil': 'firebrick',
}


def create_subplot_histogram(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    bin_number: int = 20,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
    color_code: str = 'base_fuel',
    statistic: str = 'count',
    include_zero: bool = False,
    show_legend: bool = False,
    color_map: Optional[Dict[str, str]] = None
) -> None:
    """
    Creates a histogram visualization on the provided axes with enhanced error handling.
    
    This function applies the same architectural improvements as the adoption potential
    refactoring: comprehensive type hints, robust error handling, and flexible
    parameter design.
    
    Args:
        ax: Matplotlib axes to plot on
        df: DataFrame containing the data to visualize
        x_col: Column name for the data to plot
        bin_number: Number of histogram bins
        x_label: Optional label for x-axis
        y_label: Optional label for y-axis
        lower_percentile: Lower percentile for data range filtering
        upper_percentile: Upper percentile for data range filtering
        color_code: Column name for color coding (usually fuel type)
        statistic: Statistic to compute ('count', 'density', 'probability', etc.)
        include_zero: Whether to include zero values in the visualization
        show_legend: Whether to show legend on this subplot
        color_map: Optional color mapping dictionary (uses default if None)
        
    Returns:
        None. The plot is created on the provided axes.
        
    Raises:
        ValueError: If required columns are not found or data is invalid
        TypeError: If input parameters are of wrong type
    """
    # Input validation (following adoption potential pattern)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if not isinstance(ax, plt.Axes):
        raise TypeError("ax must be a matplotlib Axes object")
    
    # Validate required columns exist
    if x_col not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Column '{x_col}' not found in DataFrame. "
                        f"Available columns: {available_cols}")
    
    if color_code not in df.columns:
        available_cols = list(df.columns)
        raise ValueError(f"Color coding column '{color_code}' not found in DataFrame. "
                        f"Available columns: {available_cols}")
    
    # Use default color map if none provided
    if color_map is None:
        color_map = DEFAULT_COLOR_MAP_FUEL.copy()
    
    # Data preparation (following adoption potential defensive copying pattern)
    df_copy = df.copy()
    
    # Handle zero values
    if not include_zero:
        df_copy[x_col] = df_copy[x_col].replace(0, np.nan)
    
    # Remove any non-finite values
    initial_count = len(df_copy)
    df_copy = df_copy[df_copy[x_col].notna() & np.isfinite(df_copy[x_col])]
    final_count = len(df_copy)
    
    if final_count == 0:
        raise ValueError(f"No valid data remaining after filtering for column '{x_col}'. "
                        f"Original count: {initial_count}")
    
    if final_count < initial_count * 0.1:  # Less than 10% of data remaining
        print(f"Warning: Only {final_count}/{initial_count} ({final_count/initial_count*100:.1f}%) "
              f"of data points are valid for column '{x_col}'")
    
    # Calculate percentile limits
    try:
        lower_limit = df_copy[x_col].quantile(lower_percentile / 100)
        upper_limit = df_copy[x_col].quantile(upper_percentile / 100)
    except Exception as e:
        raise ValueError(f"Error calculating percentiles for column '{x_col}': {str(e)}")
    
    # Filter data to percentile range
    valid_data = df_copy[x_col][(df_copy[x_col] >= lower_limit) & (df_copy[x_col] <= upper_limit)]
    
    if len(valid_data) == 0:
        raise ValueError(f"No data points within percentile range [{lower_percentile}, {upper_percentile}] "
                        f"for column '{x_col}'")
    
    # Prepare color mapping (following adoption potential color handling pattern)
    unique_categories = df_copy[color_code].unique()
    colors = [color_map.get(fuel, 'gray') for fuel in unique_categories]
    hue_order = [fuel for fuel in unique_categories if fuel in color_map]
    
    if not hue_order:
        print(f"Warning: No colors defined for categories {unique_categories}. Using default colors.")
        hue_order = list(unique_categories)
        colors = plt.cm.Set1(np.linspace(0, 1, len(hue_order)))
    
    # Create the histogram
    try:
        sns.histplot(
            data=df_copy, 
            x=valid_data, 
            kde=False, 
            bins=bin_number, 
            hue=color_code, 
            hue_order=hue_order, 
            stat=statistic, 
            multiple="stack", 
            palette=colors, 
            ax=ax, 
            legend=show_legend
        )
    except Exception as e:
        raise ValueError(f"Error creating histogram for column '{x_col}': {str(e)}")
    
    # Set labels and formatting
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=22)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=22)
    
    # Set axis limits and styling
    ax.set_xlim(left=lower_limit, right=upper_limit)
    ax.tick_params(axis='both', labelsize=22)
    sns.despine(ax=ax)


def create_subplot_grid_histogram(
    df: Optional[pd.DataFrame] = None,
    dataframes: Optional[List[pd.DataFrame]] = None,
    dataframe_indices: Optional[List[int]] = None,
    subplot_positions: List[Tuple[int, int]] = None,
    x_cols: List[str] = None,
    x_labels: Optional[List[str]] = None,
    y_label: Optional[str] = None,  # Keep for backward compatibility
    y_labels: Optional[List[str]] = None,  # NEW: Individual y-labels
    bin_number: int = 20,
    lower_percentile: float = 2.5,
    upper_percentile: float = 97.5,
    statistic: str = 'count',
    color_code: str = 'base_fuel',
    include_zero: bool = False,
    suptitle: Optional[str] = None,
    sharex: bool = False,
    sharey: bool = False,
    column_titles: Optional[List[str]] = None,
    show_legend: bool = False,  # CHANGED: Default to False for cleaner multi-panel appearance
    figure_size: Tuple[int, int] = (12, 10),
    color_map: Optional[Dict[str, str]] = None
) -> plt.Figure:
    """
    Creates a grid of histogram subplots with support for multiple DataFrames.
    
    ENHANCED with 4 key improvements:
    1. Individual y-labels for each subplot (similar to x_labels functionality)
    2. Proper y-axis tick visibility and formatting
    3. Cleaner legend management (no individual subplot legends by default)
    4. Optimized whitespace management for better space utilization
    
    Args:
        df: Single DataFrame for backward compatibility (mutually exclusive with dataframes)
        dataframes: List of DataFrames for multiple-source visualization
        dataframe_indices: List of indices mapping each subplot to a DataFrame in dataframes list
        subplot_positions: List of (row, col) tuples specifying subplot positions
        x_cols: List of column names for x-axis data in each subplot
        x_labels: Optional list of x-axis labels for each subplot
        y_label: Optional label for y-axis (applied to all subplots if y_labels not provided)
        y_labels: Optional list of y-axis labels for each subplot (takes precedence over y_label)
        bin_number: Number of histogram bins for all subplots
        lower_percentile: Lower percentile for data range filtering
        upper_percentile: Upper percentile for data range filtering
        statistic: Statistic to compute for all histograms
        color_code: Column name for color coding
        include_zero: Whether to include zero values
        suptitle: Optional super title for the entire figure
        sharex: Whether subplots should share x-axis
        sharey: Whether subplots should share y-axis
        column_titles: Optional list of column titles
        show_legend: Whether to show individual subplot legends (default: False for cleaner multi-panel appearance)
                    Note: Main legend is always shown at figure level regardless of this setting
        figure_size: Figure size as (width, height) in inches
        color_map: Optional color mapping dictionary
        
    Returns:
        Matplotlib Figure object containing the visualization
        
    Raises:
        ValueError: If inputs are incompatible or improperly formatted
        TypeError: If input parameters are of wrong type
    """
    # ========== INPUT VALIDATION AND COMPATIBILITY LOGIC ==========
    
    # Validate that we have either single or multiple DataFrame setup, but not both
    if df is not None and dataframes is not None:
        raise ValueError("Cannot specify both 'df' and 'dataframes'. Use either single or multiple DataFrame mode.")
    
    if df is None and dataframes is None:
        raise ValueError("Must specify either 'df' (single DataFrame) or 'dataframes' (multiple DataFrames).")
    
    # Validate required parameters exist
    if subplot_positions is None:
        raise ValueError("subplot_positions is required")
    if x_cols is None:
        raise ValueError("x_cols is required")
    
    # Set up DataFrame list and indices based on input mode
    if df is not None:
        # Single DataFrame mode (backward compatibility)
        df_list = [df]
        df_indices = [0] * len(subplot_positions)  # All subplots use the same DataFrame
        
    else:
        # Multiple DataFrame mode
        if not isinstance(dataframes, list):
            raise TypeError("dataframes must be a list of pandas DataFrames")
        
        if not dataframes:
            raise ValueError("dataframes list cannot be empty")
        
        # Validate all items in dataframes are actually DataFrames
        for i, dataframe in enumerate(dataframes):
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError(f"dataframes[{i}] must be a pandas DataFrame, got {type(dataframe)}")
            if dataframe.empty:
                raise ValueError(f"dataframes[{i}] is empty")
        
        df_list = dataframes
        
        # Handle dataframe_indices
        if dataframe_indices is None:
            # Default: positional mapping (subplot i uses dataframe i % len(dataframes))
            df_indices = [i % len(dataframes) for i in range(len(subplot_positions))]
        else:
            if not isinstance(dataframe_indices, list):
                raise TypeError("dataframe_indices must be a list of integers")
            
            if len(dataframe_indices) != len(subplot_positions):
                raise ValueError(f"Length mismatch: dataframe_indices ({len(dataframe_indices)}) "
                               f"must match subplot_positions ({len(subplot_positions)})")
            
            # Validate that all indices are valid
            for i, idx in enumerate(dataframe_indices):
                if not isinstance(idx, int):
                    raise TypeError(f"dataframe_indices[{i}] must be an integer, got {type(idx)}")
                if idx < 0 or idx >= len(dataframes):
                    raise ValueError(f"dataframe_indices[{i}] = {idx} is out of range. "
                                   f"Must be between 0 and {len(dataframes)-1}")
            
            df_indices = dataframe_indices
    
    # ========== STANDARD INPUT VALIDATION ==========
    
    if not isinstance(subplot_positions, list):
        raise TypeError("subplot_positions must be a list")
    
    if not isinstance(x_cols, list):
        raise TypeError("x_cols must be a list")
    
    # Validate input lengths
    if len(subplot_positions) != len(x_cols):
        raise ValueError(f"Length mismatch: subplot_positions ({len(subplot_positions)}) and "
                        f"x_cols ({len(x_cols)}) must have the same length")
    
    if x_labels and len(x_labels) != len(subplot_positions):
        raise ValueError(f"Length mismatch: x_labels ({len(x_labels)}) must match "
                        f"subplot_positions ({len(subplot_positions)})")
    
    # NEW: Validate y_labels length
    if y_labels and len(y_labels) != len(subplot_positions):
        raise ValueError(f"Length mismatch: y_labels ({len(y_labels)}) must match "
                        f"subplot_positions ({len(subplot_positions)})")
    
    # Use default color map if none provided
    if color_map is None:
        color_map = DEFAULT_COLOR_MAP_FUEL.copy()
    
    # ========== COLUMN VALIDATION ACROSS DATAFRAMES ==========
    
    # Validate that all required columns exist in their respective DataFrames
    for idx, (x_col, df_idx) in enumerate(zip(x_cols, df_indices)):
        current_df = df_list[df_idx]
        
        # Check x_col exists
        if x_col not in current_df.columns:
            available_cols = list(current_df.columns)
            raise ValueError(f"Subplot {idx}: Column '{x_col}' not found in DataFrame {df_idx}. "
                           f"Available columns: {available_cols}")
        
        # Check color_code exists
        if color_code not in current_df.columns:
            available_cols = list(current_df.columns)
            raise ValueError(f"Subplot {idx}: Color coding column '{color_code}' not found in DataFrame {df_idx}. "
                           f"Available columns: {available_cols}")
    
    # ========== GRID SETUP ==========
    
    # Determine grid dimensions
    if not subplot_positions:
        raise ValueError("subplot_positions cannot be empty")
    
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
    
    # Ensure axes is always 2D for consistent indexing
    if num_rows == 1 and num_cols == 1:
        axes = np.array([[axes]])
    elif num_rows == 1:
        axes = np.array([axes])
    elif num_cols == 1:
        axes = np.array([[ax] for ax in axes])
    else:
        axes = np.array(axes)
    
    # ========== CREATE SUBPLOTS ==========
    
    # Create plots for each subplot
    for idx, (pos, x_col, df_idx) in enumerate(zip(subplot_positions, x_cols, df_indices)):
        try:
            # Get the subplot position and axes
            row, col = pos
            ax = axes[row, col]
            
            # Get the appropriate DataFrame for this subplot
            current_df = df_list[df_idx]
            
            # Get labels (UPDATED: Individual y-label logic)
            x_label = x_labels[idx] if x_labels and idx < len(x_labels) else None
            
            # NEW: Individual y-label logic
            if y_labels and idx < len(y_labels):
                current_y_label = y_labels[idx]
            else:
                current_y_label = y_label  # Fallback to global y_label for backward compatibility
            
            # Create the histogram subplot
            create_subplot_histogram(
                ax=ax,
                df=current_df,  # Use the DataFrame specific to this subplot
                x_col=x_col,
                bin_number=bin_number,
                x_label=x_label,
                y_label=current_y_label,  # Use the determined y-label
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
                color_code=color_code,
                statistic=statistic,
                include_zero=include_zero,
                show_legend=show_legend,
                color_map=color_map
            )
            
            # UPDATED: Fix missing Y-axis ticks (ISSUE 2 FIX)
            # WHY: Seaborn histplot may not show y-ticks clearly in subplot grids
            # HOW: Ensure y-axis visibility and proper tick formatting, but let matplotlib handle tick placement
            
            # Make sure y-axis ticks are visible with proper formatting
            ax.tick_params(axis='y', which='major', labelsize=20, direction='out', length=6, width=1)
            
            # Ensure y-axis is visible (let matplotlib handle automatic tick placement)
            ax.spines['left'].set_visible(True)
            ax.yaxis.set_visible(True)
            
        except Exception as e:
            print(f"Error creating subplot at position {pos} for column '{x_col}' from DataFrame {df_idx}: {str(e)}")
            # Create an error plot
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
    
    # ========== FIGURE FORMATTING ==========
    
    # Add super title
    if suptitle:
        fig.suptitle(suptitle, fontweight='bold', fontsize=22)
    
    # Add column titles
    if column_titles:
        if len(column_titles) != num_cols:
            print(f"Warning: Number of column titles ({len(column_titles)}) doesn't match "
                  f"number of columns ({num_cols})")
        else:
            for col_index, title in enumerate(column_titles):
                if col_index < num_cols:
                    axes[0, col_index].set_title(title, fontsize=22, fontweight='bold')
    
    # Handle shared y-axis labels
    if sharey:
        for row_index in range(num_rows):
            for col_index in range(num_cols):
                if col_index > 0:
                    axes[row_index, col_index].set_yticklabels([])
    
    # UPDATED: Optimize whitespace management (ISSUE 4 FIX)
    # WHY: Reduce excessive whitespace between plot area and legend for better space utilization
    # HOW: Fine-tune layout parameters for optimal balance between readability and efficiency
    
    # Add global legend with optimized positioning
    legend_labels = list(color_map.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[label]) for label in legend_labels]
    
    # Position legend closer to plot area (reduced bbox_to_anchor y-value)
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='lower center', 
        ncol=len(legend_labels), 
        prop={'size': 20}, 
        labelspacing=0.5, 
        bbox_to_anchor=(0.5, -0.01)  # CHANGED: Reduced from -0.05 to -0.01
    )
    
    # Optimize layout with reduced bottom margin
    plt.tight_layout(rect=[0, 0.01, 1, 0.98])  # CHANGED: Reduced bottom from 0.12 to 0.01
    fig.subplots_adjust(bottom=0.15)  # CHANGED: Reduced from 0.25 to 0.15
    
    # Reduce excessive padding between x-tick labels and axis labels
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i, j].xaxis.labelpad = 10  # CHANGED: Reduced from 20 to 10
    
    return fig
