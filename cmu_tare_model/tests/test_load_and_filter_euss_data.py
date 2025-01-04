"""
Below is an example prompt you can copy and paste into ChatGPT to request **pytest tests** for the code. It includes context, objectives, and constraints—following best practices in prompt engineering. 

Feel free to adjust the details (e.g., if you have specific data schemas or additional rules) before sending it to ChatGPT.

---
# ========================================================================================================
# PROMPT FOR LOAD_AND_FILTER_EUSS_DATA PYTEST TESTS
# ========================================================================================================

```plaintext
You are provided with a Python module containing several functions:

1. `standardize_fuel_name(fuel_desc)` - Standardizes fuel descriptions to known categories.
2. `preprocess_fuel_data(df, column_name)` - Applies standardization to a specified DataFrame column.
3. `apply_fuel_filter(df, category, enable)` - Filters a DataFrame by fuel types if enabled.
4. `apply_technology_filter(df, category, enable)` - Filters a DataFrame by technology types if enabled.
5. `debug_filters(df, filter_name)` - Prints how many rows remain after filtering.
6. `extract_city_name(row)` - Extracts the city name from a "ST, City" string.
7. `df_enduse_refactored(df_baseline, fuel_filter='Yes', tech_filter='Yes')` - Creates a new DataFrame (`df_enduse`) from `df_baseline` and applies optional filters.
8. `df_enduse_compare(df_mp, input_mp, menu_mp, df_baseline, df_cooking_range)` - Creates a comparison DataFrame (`df_compare`) by merging multiple data sources.

**Context:**
- Each function has specific responsibilities like data preprocessing, filtering, or DataFrame construction.  
- Edge cases include missing columns, empty DataFrames, incorrect formats, NaN values, etc.  
- We want to ensure that all functions behave correctly, raise expected errors, or handle unexpected inputs gracefully.

**Objectives:**
- Generate a complete Python file with **pytest** test functions that cover:
  - Normal (“happy path”) cases.
  - Edge cases (empty strings, missing columns, different data types, etc.).
  - Exception handling (`KeyError` or other errors when columns are missing or data is invalid).
  - Boundary conditions (e.g., filtering enabled vs. disabled, empty DataFrame, unknown input_mp values).
- Use **fixtures** where beneficial to avoid code repetition and keep tests organized.
- Provide short **docstrings** or inline comments to clarify the purpose of each test.

**Constraints & Requirements:**
- Tests must use **pytest** conventions (`test_*` function names).
- Include examples of `pytest.raises` for error conditions.
- Use **parametrization** where it helps test multiple values succinctly.
- Do not perform any file I/O (no reading/writing actual files).
- The final answer should be a **ready-to-run Python test file**.

**What I want from you:**
- A single Python script (or code block) with all the pytest test functions.
- Each test should cover the relevant function from the provided code.
- Use minimal but sufficient mock or sample data to illustrate each test scenario, including DataFrame creation for filtering tests.

Now, please generate the **pytest test suite** in a single Python file, satisfying these requirements. 
```

---

When you paste the above prompt into ChatGPT, it should generate a thorough **pytest**-based test suite that you can copy and run immediately with:

```bash
pytest test_file_name.py
```

Remember to tailor the sample data or columns in the tests if your real data schema differs from the example references.
"""