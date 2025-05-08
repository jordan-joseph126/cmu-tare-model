# Specialized Prompt Templates for Python Code Analysis

These specialized prompts are designed to help you get the most effective assistance from Claude when working with your research code. Each template follows prompt engineering best practices to ensure thorough, educational responses rather than just quick solutions.

## Table of Contents
1. [General Code Review and Troubleshooting](#1-general-code-review-and-troubleshooting)
2. [Data Validation Framework Review](#2-data-validation-framework-review)
3. [Public Impact Analysis Implementation](#3-public-impact-analysis-implementation)
4. [Private Impact Analysis Implementation](#4-private-impact-analysis-implementation) 
5. [Pytest Test Suite Generation](#5-pytest-test-suite-generation)
6. [Debugging Assistance](#6-debugging-assistance)
7. [Conversation Continuation](#7-conversation-continuation)

---

## 1. General Code Review and Troubleshooting

```
I need a comprehensive code review of my updated Python module. Please analyze the code to:

1. Confirm the code is performing as expected based on the provided output
2. Verify it follows the same logic as the previous version (performing identical calculations)
3. Ensure it maintains the same variable names and naming conventions
4. Thoroughly review against coding standards:
   - Google-style docstrings
   - Appropriate type hints (including complex types from typing module)
   - Strategic comments (focusing on WHY, not WHAT)
   - Proper error handling with informative messages

Here is the updated module code:
```python
[PASTE CODE HERE]
```

Here is the previous version for comparison:
```python
[PASTE PREVIOUS CODE HERE]
```

Here is sample output from the module's execution:
```
[PASTE OUTPUT HERE]
```

Additional relevant files:
- utility files (`calculation_utils.py` and `validation_framework.py`) provide supporting functions
- [REFERENCE ANY DEPENDENT MODULES]
- [REFERENCE ANY CONSTANTS OR CONFIG FILES]

Please provide a step-by-step analysis with:
- Relevant code snippets (with line numbers and surrounding context)
- Identification of any standards violations or inconsistencies
- Suggested improvements with both original and replacement code
- Explanation of WHY each change improves the code

Focus especially on maintaining the core computational logic while improving readability, maintainability, and error handling. Ensure strict adherence to all coding standards and instructions.

Before concluding, please review your entire response to ensure completeness and accuracy. I value thoroughness and correctness over speed.
```

---

## 2. Data Validation Framework Review

```
I need a detailed validation of my data validation framework implementation. Please analyze the code to:

1. Confirm consistent implementation of the five-step validation pattern:
   - Step 1: Mask initialization with initialize_validation_tracking()
   - Step 2: Series initialization with create_retrofit_only_series()
   - Step 3: Valid-only calculation (calculations only for valid homes)
   - Step 4: Valid-only updates (updates only for valid homes)
   - Step 5: Final masking with apply_final_masking()

2. Compare the utility functions across different files to ensure consistent implementation

3. Review the updated files against the originals to verify:
   - Core computational logic remains unchanged
   - Data validation is properly integrated
   - Edge cases are handled appropriately

Here are the utility functions:
```python
[PASTE UTILITY FUNCTIONS]
```

Here is the updated module that implements the validation framework:
```python
[PASTE UPDATED MODULE]
```

Here is the original module for comparison:
```python
[PASTE ORIGINAL MODULE]
```

Here is sample output from the module's execution:
```
[PASTE OUTPUT HERE]
```

Additional relevant files:
- utility files (`calculation_utils.py` and `validation_framework.py`) provide supporting functions
- [ANY OTHER RELEVANT FILES]

Please provide a comprehensive analysis with:
- Code snippets demonstrating each validation step (with line numbers and context)
- Comparison of implementation across different functions and files
- Identification of any inconsistencies in the validation pattern
- Recommendations for improvements with original and replacement code

Ensure strict adherence to all coding standards, including documentation, type hints, and error handling requirements for any recommendations.

Before concluding, please review your entire response to ensure completeness and accuracy. I value thoroughness and correctness over speed.
```

---

## 3. Public Impact Analysis Implementation

```
I need assistance implementing a public impact analysis module for energy retrofits. The module should calculate climate and health impacts of different retrofit scenarios. Please help me develop this module following these requirements:

1. The module should implement the five-step validation framework:
   - Step 1: Mask initialization with initialize_validation_tracking()
   - Step 2: Series initialization with create_retrofit_only_series()
   - Step 3: Valid-only calculation (calculations only for valid homes)
   - Step 4: Valid-only updates (updates only for valid homes)
   - Step 5: Final masking with apply_final_masking()

2. The implementation should calculate these key metrics:
   - Annual emissions (CO2e, SO2, NOx, PM2.5) for baseline and retrofit scenarios
   - Lifetime emissions for different equipment types
   - Avoided emissions (baseline - retrofit) for different policy scenarios
   - Climate damages based on social cost of carbon
   - Health damages based on emission factors and health impact models

3. The module should support analysis across different policy scenarios:
   - "No Inflation Reduction Act" scenario
   - "AEO2023 Reference Case" scenario

Here are the utility functions I'll be using:
```python
[PASTE UTILITY FUNCTIONS]
```

Here is my current implementation draft:
```python
[PASTE DRAFT CODE]
```

Here's an example of how I intend to call this module:
```python
[PASTE EXAMPLE USAGE CODE]
```

Additional relevant files:
- `modeling_params.py` for parameter definitions
- `constants.py` for system constants
- `validation_framework.py` for core validation utilities
- [ANY OTHER RELEVANT FILES]

Please provide:
- A step-by-step implementation plan
- Detailed code blocks (with comments) for key functions
- Recommendations for edge case handling
- Explanation of WHY each implementation choice is made
- Advice on performance optimization

Ensure strict adherence to all coding standards, including Google-style docstrings, appropriate type hints, strategic comments, and proper error handling.

Before concluding, please review your entire response to ensure completeness and accuracy. I value thoroughness and correctness over speed.
```

---

## 4. Private Impact Analysis Implementation

```
I need assistance implementing a private impact analysis module for energy retrofits. The module should calculate financial impacts for homeowners under different retrofit scenarios. Please help me develop this module following these requirements:

1. The module should implement the five-step validation framework:
   - Step 1: Mask initialization with initialize_validation_tracking()
   - Step 2: Series initialization with create_retrofit_only_series()
   - Step 3: Valid-only calculation (calculations only for valid homes)
   - Step 4: Valid-only updates (updates only for valid homes)
   - Step 5: Final masking with apply_final_masking()

2. The implementation should calculate these key financial metrics:
   - Installation costs for different equipment types
   - Replacement costs for future equipment replacements
   - Annual operating costs for baseline and retrofit scenarios
   - Lifetime operating costs over equipment lifetime
   - Applicable rebates and incentives
   - Net present value (NPV) with different discount rates
   - Simple payback period

3. The module should support analysis across different policy scenarios:
   - "No Inflation Reduction Act" scenario
   - "AEO2023 Reference Case" scenario (including relevant rebates)

Here are the utility functions I'll be using:
```python
[PASTE UTILITY FUNCTIONS]
```

Here is my current implementation draft:
```python
[PASTE DRAFT CODE]
```

Here's an example of how I intend to call this module:
```python
[PASTE EXAMPLE USAGE CODE]
```

Additional relevant files:
- `modeling_params.py` for parameter definitions
- `constants.py` for system constants
- `validation_framework.py` for core validation utilities
- [ANY OTHER RELEVANT FILES]

Please provide:
- A step-by-step implementation plan
- Detailed code blocks (with comments) for key functions
- Recommendations for edge case handling
- Explanation of WHY each implementation choice is made
- Advice on performance optimization

Ensure strict adherence to all coding standards, including Google-style docstrings, appropriate type hints, strategic comments, and proper error handling.

Before concluding, please review your entire response to ensure completeness and accuracy. I value thoroughness and correctness over speed.
```

---

## 5. Pytest Test Suite Generation

```
I need a comprehensive pytest test suite to validate my implementation. Please create tests that:

1. Verify correct implementation of all five validation steps:
   - Mask initialization tests
   - Series initialization tests
   - Valid-only calculation tests
   - Valid-only updates tests
   - Final masking tests

2. Include multiple test types:
   - Unit tests for individual functions
   - Integration tests for complete validation flow
   - Parametrized tests for different categories and parameters
   - Edge case tests for boundary conditions

3. Provide test fixtures for:
   - Sample DataFrames with valid and invalid homes
   - Mock validation flags and masks
   - Various equipment categories and parameter combinations

Here is the module to test:
```python
[PASTE MODULE CODE]
```

Additional relevant files:
- `validation_framework.py` for core validation utilities
- `calculation_utils.py` for utility functions
- `constants.py` for system constants
- [ANY OTHER RELEVANT FILES]

Please generate a complete pytest test file with:
- Proper pytest imports and structure
- Comprehensive docstrings following Google-style conventions
- Descriptive test names following pytest conventions
- Appropriate assertions with meaningful failure messages
- Test coverage for both success and failure paths

Follow all coding standards, with particular attention to documentation, type hints, and error handling for test utility functions.

Focus especially on testing the validation framework's behavior with different equipment categories, sensitivity parameters, and edge cases.

Before concluding, please review your entire response to ensure completeness and accuracy. I value thoroughness and correctness over speed.
```

---

## 6. Debugging Assistance

```
I'm encountering an issue with my implementation of the validation framework. Please help me diagnose and fix the problem. Here are the details:

1. Expected behavior:
   [DESCRIBE WHAT THE CODE SHOULD DO]

2. Actual behavior:
   [DESCRIBE THE ISSUE YOU'RE EXPERIENCING]

3. Error message (if any):
```
[PASTE ERROR MESSAGE]
```

4. The relevant code section where I believe the issue might be:
```python
[PASTE CODE SECTION WITH ~5 LINES BEFORE AND AFTER]
```

5. Complete module code:
```python
[PASTE COMPLETE MODULE CODE]
```

6. Example input and output:
```
[PASTE EXAMPLE INPUT/OUTPUT]
```

Additional relevant files:
- `validation_framework.py` for core validation utilities
- `calculation_utils.py` for utility functions
- [ANY OTHER RELEVANT FILES]

Please help me:
1. Identify the root cause of the issue
2. Provide a step-by-step explanation of what's happening
3. Suggest a fix with both the original and corrected code
4. Explain WHY the issue occurred and how the fix addresses it
5. Recommend any additional improvements to prevent similar issues

Focus on educational guidance that helps me understand the problem, not just a quick fix.

Before concluding, please review your entire response to ensure completeness and accuracy. I value thoroughness and correctness over speed.
```

---

## 7. Conversation Continuation

```
I'd like to continue our code review in a new conversation. Please summarize our previous discussion about the [SPECIFIC MODULE] module, including:

1. Key issues identified and their resolutions
2. Important code changes we made
3. Remaining areas for improvement

Based on our previous work, in this new conversation I would like to:
[LIST OF THINGS TO DO]

Here's a reminder of the most recent version of the code:
```python
[PASTE LATEST CODE]
```

Additional relevant files:
- [REFERENCE ANY UTILITY FILES/MODULES]
- [REFERENCE ANY DEPENDENT MODULES]
- [REFERENCE ANY CONSTANTS OR CONFIG FILES]

Please begin by summarizing our previous conversation, including relevant code snippets that will help provide context. Then proceed with a step-by-step review of the remaining tasks.

As always, I prefer your response to include detailed code snippets (original code, replacement code, and line numbers/surrounding code) to provide adequate context for each change or suggestion. Ensure all recommendations follow the coding standards.

Before concluding, please review your entire response to ensure completeness and accuracy. I value thoroughness and correctness over speed.
```

---

## Prompt Engineering Tips

### 1. Provide Sufficient Context
- Always include relevant code snippets with surrounding context (5 lines above/below)
- Reference other modules or files that interact with the code being reviewed
- Specify the expected behavior and current output

### 2. Be Specific About Your Requirements
- Clearly state what you want Claude to focus on (e.g., specific validation steps, edge cases)
- Mention the specific coding standards you want to adhere to
- Specify what format you want the response in

### 3. Request Educational Explanations
- Ask for explanations of WHY solutions work, not just WHAT they are
- Request step-by-step guidance for complex problems
- Use phrases like "help me understand" to encourage detailed explanations

### 4. Structure Your Prompts
- Use numbered or bulleted lists for multiple requirements
- Include clear section headers
- Divide complex questions into logical parts

### 5. Iterative Refinement
- Start with broader prompts and refine based on responses
- Be ready to follow up with clarifying questions
- Save effective prompts as templates for future use
