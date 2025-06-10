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


def create_subplot_grid_histogram(
    df: Optional[pd.DataFrame] = None,
    dataframes: Optional[List[pd.DataFrame]] = None,
    dataframe_indices: Optional[List[int]] = None,
    subplot_positions: List[Tuple[int, int]] = None,
    x_cols: List[str] = None,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    bin_number: Union[int, str] = 'auto',
    lower_percentile: Optional[float] = None,
    upper_percentile: Optional[float] = None,
    subplot_titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
    figure_size: Tuple[int, int] = (12, 8),
    sharex: bool = False,
    sharey: bool = False,
    color_code: Optional[str] = None,
    statistic: str = 'count',
    include_zero: bool = False,
    show_legend: bool = False
) -> plt.Figure:
    """
    Creates a grid of histograms with support for single or multiple DataFrames.

    Creates multi-panel histogram visualizations with consistent formatting and 
    intelligent bin calculation. Handles large datasets efficiently and provides 
    accurate bin counts when percentile filtering is applied. Supports both single 
    DataFrame (backward compatible) and multiple DataFrame modes.

    Args:
        df: Single DataFrame for backward compatibility.
        dataframes: List of DataFrames for multi-DataFrame visualizations.
        dataframe_indices: List of indices mapping subplot positions to DataFrames.
        subplot_positions: List of (row, col) tuples for subplot placement.
        x_cols: Column names for x-axis on each subplot.
        x_labels: Optional list of labels for x-axes.
        y_labels: Optional list of labels for y-axes.
        bin_number: Number of bins (int) or 'auto' for data-size-appropriate binning.
            For large datasets (100K+ rows), 'auto' defaults to 40 bins.
        lower_percentile: Lower percentile for display range (0-100), or None for full range.
            When specified with upper_percentile, bins are calculated within this range.
        upper_percentile: Upper percentile for display range (0-100), or None for full range.
            When specified with lower_percentile, bins are calculated within this range.
        subplot_titles: Titles for each subplot.
        suptitle: Super title for the entire figure.
        figure_size: Tuple for figure size (width, height) in inches.
        sharex: Whether to share x-axis across subplots.
        sharey: Whether to share y-axis across subplots.
        color_code: Column name for hue stacking by category.
        statistic: Statistic to compute ('count', 'density', 'frequency', 'probability').
        include_zero: Whether to include zero values in the histogram.
        show_legend: Whether to show legend on individual subplots.

    Returns:
        Matplotlib Figure containing the histogram grid.

    Raises:
        ValueError: If input parameters are invalid or inconsistent.
        KeyError: If specified columns don't exist in the DataFrame(s).
    """
    # Input validation
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
    
    # Prepare figure and axes
    nrows = max(r for r, c in subplot_positions) + 1
    ncols = max(c for r, c in subplot_positions) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figure_size, sharex=sharex, sharey=sharey)
    
    # Ensure axes is always 2D array for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # SOLUTION 1: Intelligent bin calculation for 'auto' setting
    # Replace seaborn's 'auto' with data-size-appropriate defaults for large datasets
    if bin_number == 'auto':
        # Sample first column from first DataFrame to estimate appropriate bin count
        first_df = df_list[0]
        first_col = x_cols[0]
        if first_col in first_df.columns:
            sample_data = first_df[first_col].dropna()
            n_valid = len(sample_data)
            
            if n_valid > 100000:
                bin_number = 40  # Conservative bin count for large datasets
            elif n_valid > 10000:
                bin_number = 35
            else:
                bin_number = 30  # Let seaborn handle smaller datasets

    # Set up color mapping
    if color_code:
        # Get unique values across all DataFrames for consistent color mapping
        all_fuel_values = set()
        for df_current in df_list:
            if color_code in df_current.columns:
                all_fuel_values.update(df_current[color_code].dropna().unique())
        
        hue_order = [fuel for fuel in all_fuel_values if fuel in COLOR_MAP_FUEL]
        colors = [COLOR_MAP_FUEL[fuel] for fuel in hue_order]
        palette = dict(zip(hue_order, colors))
    else:
        hue_order = None
        palette = None

    # Plot each subplot
    for idx, (pos, x_col) in enumerate(zip(subplot_positions, x_cols)):
        row, col = pos
        ax = axes[row, col]
        
        # Get the appropriate DataFrame for this subplot (no copying here)
        current_df = df_list[df_indices[idx]]
        
        # Check column existence
        if x_col not in current_df.columns:
            raise KeyError(f"Column '{x_col}' not found in DataFrame at index {df_indices[idx]}. Available columns: {list(current_df.columns)}")
        
        # MEMORY-EFFICIENT: Copy only when needed for modifications
        df_plot = current_df.copy()
        
        # Handle zero values if requested
        if not include_zero:
            if x_col in df_plot.columns:
                df_plot[x_col] = df_plot[x_col].replace(0, np.nan)
        
        # SOLUTION 2: Calculate bins based on display range when percentiles are specified
        # This ensures the specified number of bins appears within the visible range
        if lower_percentile is not None and upper_percentile is not None:
            # Calculate the actual display limits
            lower_limit = df_plot[x_col].quantile(lower_percentile / 100)
            upper_limit = df_plot[x_col].quantile(upper_percentile / 100)
            
            # Create bins spanning only the display range
            if isinstance(bin_number, int):
                # Create exactly bin_number bins within the percentile range
                display_bins = np.linspace(lower_limit, upper_limit, bin_number + 1)
            else:
                # Fallback for non-integer bin specifications
                display_bins = bin_number
        else:
            # Use original bin specification for full data range
            display_bins = bin_number
        
        # Create histogram with properly calculated bins
        sns.histplot(
            data=df_plot,
            x=x_col,
            hue=color_code,
            hue_order=hue_order,
            multiple='stack',
            bins=display_bins,  # Use calculated bins for proper display range
            stat=statistic,
            palette=palette,
            ax=ax,
            legend=show_legend
        )
        
        # Set labels
        ax.set_xlabel(x_labels[idx] if x_labels and idx < len(x_labels) else x_col, fontsize=22)
        ax.set_ylabel(y_labels[idx] if y_labels and idx < len(y_labels) else 'Count', fontsize=22)
        
        # Set title
        if subplot_titles and idx < len(subplot_titles):
            ax.set_title(subplot_titles[idx], fontsize=22, fontweight='bold')
        
        # Set display limits based on percentiles (now matches bin calculation)
        if lower_percentile is not None and upper_percentile is not None:
            ax.set_xlim(left=lower_limit, right=upper_limit)
        
        # Add reference line at x=0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=3, alpha=0.8, zorder=10)
        
        # Format axis labels
        ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))
        ax.tick_params(axis='both', labelsize=22)
        
        # Remove legend from individual subplots (we'll add a global legend)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        # Clean up appearance
        sns.despine(ax=ax)

    # Hide unused subplots
    for row in range(nrows):
        for col in range(ncols):
            if (row, col) not in subplot_positions:
                axes[row, col].set_visible(False)

    # Add super title
    if suptitle:
        fig.suptitle(suptitle, fontweight='bold', fontsize=22)

    # Add global legend
    if color_code and palette:
        legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
        fig.legend(
            legend_handles, 
            hue_order, 
            loc='lower center', 
            ncol=len(hue_order), 
            prop={'size': 22}, 
            bbox_to_anchor=(0.5, -0.05)
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95 if suptitle else 1])
    else:
        plt.tight_layout()

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
    # These should match the names in your COLOR_MAP_FUEL dictionary
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
