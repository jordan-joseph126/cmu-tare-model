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
        
    Raises:
        TypeError: If x cannot be converted to a numeric value
    """
    # Validate input
    try:
        x_float = float(x)
    except (TypeError, ValueError):
        return str(x)  # Return string representation instead of raising error
    
    if abs(x_float) >= 1000:
        return f'{x_float/1000:g}K'
    else:
        return f'{x_float:g}'


def create_subplot_histogram(
    ax: plt.Axes, 
    df: pd.DataFrame, 
    x_col: str, 
    bin_number: Union[int, str] = 'auto',  # CHANGE: Allow 'auto' for intelligent binning
    x_label: Optional[str] = None, 
    y_label: Optional[str] = None, 
    lower_percentile: Optional[float] = 2.5,  # CHANGE: Allow None 
    upper_percentile: Optional[float] = 97.5,  # CHANGE: Allow None
    color_code: str = 'base_fuel', 
    statistic: str = 'count', 
    include_zero: bool = False, 
    show_legend: bool = False
) -> None:
    """Creates a histogram on the provided axes using the specified DataFrame.
    
    This function is designed to be used within a grid of subplots. It can optionally
    limit the display range based on percentiles while showing all data, and creates 
    a stacked histogram by fuel type with consistent formatting.
    
    Args:
        ax: Matplotlib Axes object where the histogram will be plotted
        df: DataFrame containing the data to plot
        x_col: Column name in df for x-axis data
        bin_number: Number of bins for the histogram, or 'auto' for intelligent binning
        x_label: Optional label for x-axis
        y_label: Optional label for y-axis
        lower_percentile: Lower percentile for display range (0-100), or None for no limit
        upper_percentile: Upper percentile for display range (0-100), or None for no limit
        color_code: Column name for color coding (usually fuel type)
        statistic: Statistic to compute ('count', 'density', 'probability', etc.)
        include_zero: Whether to include zero values in the visualization
        show_legend: Whether to show legend on this subplot
        
    Raises:
        KeyError: If x_col or color_code columns don't exist in the DataFrame
    """
    # Minimal validation - just check for column existence
    if x_col not in df.columns:
        raise KeyError(f"Column '{x_col}' not found in DataFrame. Available columns: {list(df.columns)}")
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Handle zero values (keep original approach - correct for multi-appliance data)
    if not include_zero:
        df_copy[x_col] = df_copy[x_col].replace(0, np.nan)

    # FIXED: Calculate percentile limits for DISPLAY only (don't filter the data)
    if lower_percentile is not None and upper_percentile is not None:
        lower_limit = df_copy[x_col].quantile(lower_percentile / 100)
        upper_limit = df_copy[x_col].quantile(upper_percentile / 100)
    else:
        lower_limit = None
        upper_limit = None

    # Set the hue_order to match the unique fuel categories that exist in color_map
    hue_order = [fuel for fuel in df_copy[color_code].unique() if fuel in color_map_fuel]
    
    # Get colors only for the fuels that actually exist in the data (prevents palette warning)
    colors = [color_map_fuel[fuel] for fuel in hue_order]

    # FIXED: Create the histogram using column name consistently (no filtered Series)
    sns.histplot(
        data=df_copy,       # Full DataFrame (NaN values automatically excluded)
        x=x_col,           # Column name (not filtered Series!)
        kde=False, 
        bins=bin_number,   # Now supports 'auto' 
        hue=color_code, 
        hue_order=hue_order, 
        stat=statistic, 
        multiple="stack", 
        palette=colors, 
        ax=ax, 
        legend=show_legend
    )

    # Set labels if provided
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=22)

    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=22)

    # FIXED: Set axis limits for DISPLAY only (optional, based on percentiles)
    if lower_limit is not None and upper_limit is not None:
        ax.set_xlim(left=lower_limit, right=upper_limit)
    
    ax.tick_params(axis='both', labelsize=22)

    # Add vertical reference line at x=0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=3, alpha=0.8, zorder=10)
    
    # Format axis labels to use K for thousands
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    
    # Remove top and right spines for cleaner appearance
    sns.despine()


def create_subplot_grid_histogram(
    df: Optional[pd.DataFrame] = None,                    # Single DataFrame (backward compatible)
    dataframes: Optional[List[pd.DataFrame]] = None,      # Multiple DataFrames
    dataframe_indices: Optional[List[int]] = None,        # DataFrame mapping
    subplot_positions: Optional[List[Tuple[int, int]]] = None, 
    x_cols: Optional[List[str]] = None, 
    x_labels: Optional[List[str]] = None, 
    y_label: Optional[str] = None,
    y_labels: Optional[List[str]] = None,                 # Individual y-labels
    bin_number: Union[int, str] = 'auto',                 # CHANGE: Default to 'auto' for intelligent binning
    lower_percentile: Optional[float] = None,             # CHANGE: Default to None (show full range)
    upper_percentile: Optional[float] = None,             # CHANGE: Default to None (show full range)
    statistic: str = 'count', 
    color_code: str = 'base_fuel', 
    include_zero: bool = False,                           # Keep original default (zero → NaN is correct)
    suptitle: Optional[str] = None, 
    sharex: bool = False, 
    sharey: bool = False, 
    subplot_titles: Optional[List[str]] = None, 
    show_legend: bool = False,                            # Changed default to False
    figure_size: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Creates a grid of histograms with support for single or multiple DataFrames.
    
    This function creates a customizable grid of histograms with flexible DataFrame handling.
    It supports both single DataFrame (backward compatible) and multiple DataFrame modes.
    
    UPDATED: Now shows full data range by default, with optional display range limiting.
    
    Args:
        df: Single DataFrame for backward compatibility
        dataframes: List of DataFrames for multi-DataFrame visualizations
        dataframe_indices: List of indices mapping subplot positions to DataFrames
        subplot_positions: List of (row, column) tuples for subplot placement
        x_cols: List of column names for x-axis data in each subplot
        x_labels: Optional list of x-axis labels for each subplot
        y_label: Global y-axis label applied to all subplots (if y_labels not provided)
        y_labels: Optional list of individual y-axis labels for each subplot
        bin_number: Number of bins for the histograms, or 'auto' for intelligent binning
        lower_percentile: Lower percentile for display range (0-100), or None for full range
        upper_percentile: Upper percentile for display range (0-100), or None for full range
        statistic: Statistic to compute ('count', 'density', 'probability', etc.)
        color_code: Column name for color coding (usually fuel type)
        include_zero: Whether to include zero values in the visualization
        suptitle: Super title for the entire figure
        sharex: Whether to share x-axes across subplots
        sharey: Whether to share y-axes across subplots
        subplot_titles: Optional list of titles for each column
        show_legend: Whether to show legend for each subplot
        figure_size: Size of the figure as (width, height) in inches
        
    Returns:
        Matplotlib Figure object containing the grid of histograms
        
    Raises:
        ValueError: If input parameters are invalid or inconsistent
        KeyError: If specified columns don't exist in the DataFrame(s)
    """
    # Simple input validation (original style - minimal)
    if df is not None and dataframes is not None:
        raise ValueError("Specify either 'df' or 'dataframes', not both")
    if df is None and dataframes is None:
        raise ValueError("Must specify either 'df' or 'dataframes'")
    if subplot_positions is None or not subplot_positions:
        raise ValueError("subplot_positions must be provided and non-empty")
    if x_cols is None or not x_cols:
        raise ValueError("x_cols must be provided and non-empty")
    if len(subplot_positions) != len(x_cols):
        raise ValueError(f"Number of subplot positions ({len(subplot_positions)}) must match number of x columns ({len(x_cols)})")
    
    # Set up DataFrame handling (same as before)
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
    
    # Original grid setup logic (unchanged)
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

    # Original dictionary mapping approach
    subplot_axes = {(pos[0], pos[1]): axes[pos[0], pos[1]] for pos in subplot_positions}

    # Create histograms with updated function
    for i, (pos, x_col) in enumerate(zip(subplot_positions, x_cols)):
        # Get the appropriate DataFrame for this subplot
        current_df = df_list[df_indices[i]]
        
        # Check column existence just once before calling helper function
        if x_col not in current_df.columns:
            raise KeyError(f"Column '{x_col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)}")
        
        # Get labels
        x_label = x_labels[i] if x_labels else None
        current_y_label = y_labels[i] if y_labels else y_label
        
        # Create histogram using updated function
        create_subplot_histogram(
            ax=subplot_axes[pos],
            df=current_df,
            x_col=x_col,
            bin_number=bin_number,          # Now supports 'auto'
            x_label=x_label,
            y_label=current_y_label,
            lower_percentile=lower_percentile,  # Now defaults to None
            upper_percentile=upper_percentile,  # Now defaults to None
            statistic=statistic,
            color_code=color_code,
            include_zero=include_zero,
            show_legend=show_legend
        )

    # Original title logic (unchanged)
    if suptitle:
        plt.suptitle(suptitle, fontweight='bold', fontsize=22)

    # FIXED: Apply subplot titles to individual subplot positions, not just top row  
    if subplot_titles:
        for i, (pos, title) in enumerate(zip(subplot_positions, subplot_titles)):
            if i < len(subplot_titles):
                subplot_axes[pos].set_title(title, fontsize=22, fontweight='bold')

    # Original legend logic below (unchanged)
    legend_labels = list(color_map_fuel.keys())
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map_fuel[label]) for label in legend_labels]
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='lower center', 
        ncol=len(legend_labels), 
        prop={'size': 22}, 
        labelspacing=0.5, 
        bbox_to_anchor=(0.5, -0.05)
    )             
    
    # Original layout logic with room for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig


def print_positive_percentages_complete(
    df: Optional[pd.DataFrame] = None,
    dataframes: Optional[List[pd.DataFrame]] = None,
    dataframe_indices: Optional[List[int]] = None,
    column_names: List[str] = None,
    subplot_titles: Optional[List[str]] = None,
    fuel_column: str = 'base_fuel'
):
    """
    Print the percentage of positive values for each specified column across single or multiple DataFrames,
    with a breakdown by fuel type and information about missing/invalid data.
    
    Args:
        df: Single DataFrame for backward compatibility
        dataframes: List of DataFrames for multi-DataFrame analysis
        dataframe_indices: List of indices mapping columns to DataFrames
        column_names: List of column names to analyze
        subplot_titles: Optional list of display titles for each column
        fuel_column: Column name containing fuel type information
    """
    # Input validation
    if df is not None and dataframes is not None:
        print("Error: Specify either 'df' or 'dataframes', not both")
        return
    if df is None and dataframes is None:
        print("Error: Must specify either 'df' or 'dataframes'")
        return
    if column_names is None or not column_names:
        print("Error: column_names must be provided and non-empty")
        return
    
    # Set up DataFrame handling
    if df is not None:
        # Single DataFrame mode (original behavior)
        df_list = [df]
        df_indices = [0] * len(column_names)
    else:
        # Multiple DataFrame mode
        df_list = dataframes
        if dataframe_indices is None:
            df_indices = [i % len(dataframes) for i in range(len(column_names))]
        else:
            if len(dataframe_indices) != len(column_names):
                print("Error: dataframe_indices length must match number of columns")
                return
            df_indices = dataframe_indices
    
    # Get the known fuel types we want to report on (hard-coded to avoid sorting issues)
    # These should match the names in your color_map_fuel dictionary
    known_fuel_types = ['Electricity', 'Natural Gas', 'Propane', 'Fuel Oil']
    
    for i, col in enumerate(column_names):
        # Get the appropriate DataFrame for this column
        current_df = df_list[df_indices[i]]
        
        # Check if column exists in the current DataFrame
        if col not in current_df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)[:10]} ...")
            continue
            
        # Check if fuel column exists in the current DataFrame
        if fuel_column not in current_df.columns:
            print(f"Warning: Fuel column '{fuel_column}' not found in DataFrame at index {df_indices[i]}. Available columns: {list(current_df.columns)[:10]} ...")
            continue
        
        # Get display name
        display_name = subplot_titles[i] if subplot_titles and i < len(subplot_titles) else col
        
        print(f"\n===== {display_name} =====")
        
        # Get total number of rows in the DataFrame
        total_df_rows = len(current_df)
        
        # Calculate overall statistics for non-NaN values
        total_non_nan = current_df[col].count()
        positive_count = (current_df[col] > 0).sum()
        positive_percentage = (positive_count / total_non_nan * 100) if total_non_nan > 0 else 0
        
        print(f"All: {positive_percentage:.1f}% positive ({positive_count:,} of {total_non_nan:,} values)")
        
        # Track the total rows accounted for by known fuel types
        accounted_rows = 0
        
        # Calculate statistics for each known fuel type
        for fuel in known_fuel_types:
            # Filter by fuel type
            fuel_data = current_df[current_df[fuel_column] == fuel]
            
            # Skip if no data for this fuel type
            if len(fuel_data) == 0:
                continue
                
            # Count rows for this fuel type
            fuel_rows = len(fuel_data)
            accounted_rows += fuel_rows
                
            # Count non-NaN values for this column
            fuel_total = fuel_data[col].count()
            
            # Count positive values
            fuel_positive = (fuel_data[col] > 0).sum()
            
            # Calculate percentage
            fuel_percentage = (fuel_positive / fuel_total * 100) if fuel_total > 0 else 0
            
            # Print results
            print(f"{fuel}: {fuel_percentage:.1f}% positive ({fuel_positive:,} of {fuel_total:,} values)")
        
        # Calculate remaining/unaccounted rows (either "Other Fuel" or NaN values)
        unaccounted_rows = total_df_rows - accounted_rows
        unaccounted_percentage = (unaccounted_rows / total_df_rows * 100)
        
        print(f"\nHomes with \"Other Fuel\" (Invalid Fuel/Tech) make up the remaining {unaccounted_rows:,} of {total_df_rows:,} total dataframe rows ({unaccounted_percentage:.1f}%).")
        print(f"NaN values in column: {total_df_rows - total_non_nan:,} rows ({((total_df_rows - total_non_nan) / total_df_rows * 100):.1f}% of dataframe)")
