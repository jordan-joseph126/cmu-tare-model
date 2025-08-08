# 5-Step Validation Framework Reference Guide

## Overview
This framework ensures consistent handling of valid and invalid data across all calculation modules. Each step builds upon the previous one to maintain data integrity.

## Step 1: Mask Initialization
**Purpose**: Identify which rows contain valid data for processing.

**Implementation**:
```python
df_copy, valid_mask, all_columns_to_mask, category_columns_to_mask = initialize_validation_tracking(
    df_copy, category, menu_mp, verbose=verbose)
```

**Key Components**:
- `valid_mask`: Boolean Series indicating which rows should be processed
- `all_columns_to_mask`: List of columns that will need masking in the final output
- `category_columns_to_mask`: Category-specific columns to mask

**Common Issues**:
- Missing call to initialize_validation_tracking()
- Not passing the valid_mask to subsequent functions
- Not tracking columns that need masking

## Step 2: Series Initialization
**Purpose**: Create result columns with appropriate initial values (zeros for valid homes, NaN for invalid homes).

**Implementation**:
```python
discounted_savings_template = create_retrofit_only_series(df_measure_costs, valid_mask)
```

**Key Components**:
- Result series should be initialized with zeros for valid homes
- Invalid homes should be set to NaN at the outset
- For multiple result columns, initialize each one

**Common Issues**:
- Using default Series/DataFrame initialization that doesn't respect valid_mask
- Setting invalid homes to zero instead of NaN
- Not using create_retrofit_only_series() utility

## Step 3: Valid-Only Calculation
**Purpose**: Perform calculations only on valid rows to improve performance and avoid errors.

**Implementation**:
```python
# Calculate only for valid homes
calculation_mask = valid_mask & df[required_column].notna()
result_for_valid_homes = perform_calculation(df.loc[calculation_mask])
```

**Key Components**:
- Apply calculations only to rows where valid_mask is True
- Further refine calculation mask when required columns might be missing
- Avoid wasting computation on invalid data

**Common Issues**:
- Performing calculations on all rows regardless of validity
- Not checking for missing values in required columns

## Step 4: Valid-Only Updates
**Purpose**: Update result columns only for valid rows, using efficient collection patterns.

**Implementation**:
```python
# Using list-based collection instead of incremental updates
yearly_avoided_costs.append(avoided_costs)

# Later converts list to DataFrame and sums
avoided_costs_df = pd.concat(yearly_avoided_costs, axis=1)
total_discounted_savings = avoided_costs_df.sum(axis=1)
```

**Key Components**:
- Collect results in lists or dictionaries before final assembly
- Avoid incremental updates to DataFrames which are inefficient
- Only update valid rows in the final result

**Common Issues**:
- Using inefficient incremental updates
- Not respecting the valid_mask when updating results
- Overwriting previously calculated values

## Step 5: Final Masking
**Purpose**: Ensure all invalid data is properly masked in the final output.

**Implementation**:
```python
df_result = apply_temporary_validation_and_mask(df_copy, df_new_columns, all_columns_to_mask, verbose=verbose)
```

**Key Components**:
- Apply masking to all tracked columns
- Ensure consistent NaN values for invalid homes
- Use provided utility functions for consistency

**Common Issues**:
- Forgetting to apply final masking
- Not tracking all columns that need masking
- Manually applying masks inconsistently
