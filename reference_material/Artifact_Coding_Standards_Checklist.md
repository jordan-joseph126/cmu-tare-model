# Code Review Checklist for Validation Framework

## Validation Framework Implementation Checklist

### Step 1: Mask Initialization
- [ ] Calls `initialize_validation_tracking()` with appropriate parameters
- [ ] Stores and passes `valid_mask` to subsequent functions
- [ ] Properly tracks columns that need masking
- [ ] Handles category-specific validation if needed

### Step 2: Series Initialization
- [ ] Uses `create_retrofit_only_series()` for result columns
- [ ] Sets appropriate initial values (typically zeros) for valid homes
- [ ] Sets NaN for invalid homes
- [ ] Initializes all required result columns

### Step 3: Valid-Only Calculation
- [ ] Performs calculations only on rows where `valid_mask` is True
- [ ] Uses compound masks when necessary (e.g., `valid_mask & df[col].notna()`)
- [ ] Avoids unnecessary calculations for invalid homes
- [ ] Properly handles missing values in required columns

### Step 4: Valid-Only Updates
- [ ] Uses list-based collection pattern instead of incremental updates
- [ ] Updates only valid homes with calculated values
- [ ] Uses proper DataFrame indexing (e.g., `.loc[valid_mask]`)
- [ ] Preserves NaN values for invalid homes

### Step 5: Final Masking
- [ ] Calls `apply_temporary_validation_and_mask()` or `apply_final_masking()`
- [ ] Passes all tracked columns for masking
- [ ] Ensures consistent NaN values for invalid homes
- [ ] Returns properly masked DataFrame

## Code Quality Checklist

### Documentation
- [ ] Google-style docstrings for all functions
- [ ] Type hints for parameters and return values
- [ ] Strategic comments explaining WHY, not WHAT
- [ ] Updated documentation to reflect validation framework

### Error Handling
- [ ] Validates inputs with informative error messages
- [ ] Uses specific exception types in try/except blocks
- [ ] Follows "fail fast, fail clearly" principle
- [ ] Handles edge cases appropriately

### Performance
- [ ] Avoids inefficient operations (e.g., DataFrame updates in loops)
- [ ] Uses vectorized operations when possible
- [ ] Avoids duplicate calculations
- [ ] Handles large datasets efficiently

### Maintainability
- [ ] Consistent variable naming conventions
- [ ] Modular design with appropriate function boundaries
- [ ] Minimizes code duplication
- [ ] Clear data flow through the program

## Original vs. Modified Code Comparison
For each module, note:
- Original implementation of each validation step
- Modified implementation with improvements
- Rationale for changes

## Edge Cases to Check
- [ ] Empty DataFrames
- [ ] All homes invalid
- [ ] Missing required columns
- [ ] Zero values in calculation fields
- [ ] Different equipment categories
- [ ] Different sensitivity parameters
