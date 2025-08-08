# Python Code Analysis Prompt Engineering Project

## PROJECT OVERVIEW
This project focuses on developing effective prompt engineering strategies for Python code analysis, validation, and enhancement in research contexts. You will create and refine specialized prompts that enable Claude to assist with improving research code through educational guidance rather than simple solutions.

## OBJECTIVES
1. Create specialized prompts for analyzing different components of research code
2. Adapt existing prompts to incorporate prompt engineering best practices
3. Test prompts against sample code files to evaluate effectiveness
4. Document successful prompt patterns and techniques

## CORE PROMPT ENGINEERING PRINCIPLES

### 1. Specificity and Context
- Include detailed information about the code's purpose, dependencies, and requirements
- Provide relevant code snippets with surrounding context (5 lines above/below)
- Mention expected input/output behavior and data structures

### 2. Structured Instructions
- Break down complex tasks into sequential steps
- Use clear section headers to organize the prompt
- Number items in lists to ensure all points are addressed

### 3. Educational Focus
- Request explanations for WHY solutions work, not just WHAT they are
- Ask for step-by-step breakdowns of complex operations
- Encourage comparing multiple approaches when appropriate

### 4. Clear Output Format
- Specify exactly how you want the response formatted
- Request relevant code snippets with line numbers
- Ask for original code alongside improved versions for comparison

### 5. Technical Precision
- Include specific technical requirements (docstring format, typehints, etc.)
- Attach reference standards for code quality
- Specify validation framework requirements when relevant

## PROJECT DOMAINS

### 1. Data Validation Framework
This domain focuses on the 5-step data validation framework:
1. Mask Initialization with `initialize_validation_tracking()`
2. Series Initialization with `create_retrofit_only_series()`
3. Valid-Only Calculation for qualifying homes
4. Valid-Only Updates with `.loc[valid_mask]`
5. Final Masking with `apply_final_masking()`

### 2. Public Impact Analysis
This domain focuses on analyzing the public impacts of energy retrofits:
- Emission factors for electricity and fossil fuels
- Climate damage calculations
- Health damage assessments
- Calculation of social costs and benefits

### 3. Private Impact Analysis
This domain focuses on analyzing private/individual impacts:
- Installation and replacement costs
- Operational savings
- Rebate calculations
- Net present value (NPV) determinations
- Payback period analysis

### 4. Test Suite Development
This domain focuses on creating comprehensive test suites:
- Unit tests for individual functions
- Integration tests for complete workflows
- Parametrized tests for different scenarios
- Edge case testing for boundary conditions

## PROMPT TEMPLATES TO DEVELOP

For each domain, develop specialized prompt templates for the following purposes:

### 1. Code Review Prompt
- Analyzes existing code against standards
- Identifies inconsistencies and potential bugs
- Suggests improvements while maintaining core logic
- Provides educational explanation for each suggestion

### 2. Implementation Guidance Prompt
- Helps implement new features or enhancements
- Provides step-by-step guidance for complex operations
- Ensures consistent application of the validation framework
- Includes thorough documentation and error handling

### 3. Test Generation Prompt
- Creates comprehensive test suites for specific modules
- Covers all key functionality with appropriate assertions
- Includes fixtures and mocks for testing dependencies
- Follows best practices for pytest implementation

### 4. Debugging Assistance Prompt
- Helps identify and resolve specific issues
- Provides systematic debugging approaches
- Suggests diagnostic tests and log points
- Explains underlying causes of bugs

## PROJECT DELIVERABLES

1. **Domain-Specific Prompt Templates**
   - At least one specialized prompt for each domain-purpose combination
   - Include placeholders for code snippets, file references, etc.
   - Document the rationale behind each prompt's structure

2. **Prompt Engineering Guidelines**
   - Document effective patterns and techniques discovered
   - Provide examples of before/after prompts showing improvements
   - Include recommendations for adapting prompts for different scenarios

3. **Example Conversations**
   - Provide sample conversations using the developed prompts
   - Include analysis of what worked well and what could be improved
   - Demonstrate how to iterate on prompts based on initial responses

## TECHNICAL REQUIREMENTS

All prompts should incorporate these technical aspects:

### Documentation Standards
Prompts should request Google-Style docstrings:
```python
def example_function(param1: int, param2: str) -> bool:
    """One-line summary of function purpose.

    Longer description if needed with more details.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this error is raised.
    """
```

### Typehints
Prompts should emphasize:
- All function parameters and return values need appropriate typehints
- Use of typing module for complex types (List, Dict, Optional, Union, etc.)

### Comments
Prompts should encourage:
- Inline comments only for non-obvious logic
- Focus on WHY, not WHAT the code is doing

### Error Handling
Prompts should request:
- Input validation with informative error messages
- Appropriate try/except blocks with specific exception types
- Following the principle of "fail fast, fail clearly"

## EVALUATION CRITERIA

Prompts will be evaluated based on:
1. **Clarity**: How clear and unambiguous are the instructions?
2. **Specificity**: How well does the prompt target the specific task?
3. **Structure**: How well-organized is the prompt?
4. **Educational Value**: How well does the prompt encourage learning?
5. **Completeness**: How thoroughly does the prompt cover all requirements?
6. **Adaptability**: How easily can the prompt be adapted for similar tasks?

## REFERENCES

The project's filesystem includes:
- validation_framework.py: Core utilities for the 5-step validation
- calculation_utils.py: Utilities for equipment calculations
- constants.py: System constants and configuration values
- modeling_params.py: Parameter definitions for different scenarios
- test_pytest_template.py: Example test structure for validation framework

## GETTING STARTED

1. Review the existing code files to understand the project structure
2. Start with adapting the existing prompt templates to incorporate prompt engineering best practices
3. Test each prompt with sample code from the project
4. Refine prompts based on the quality of responses received
5. Document your process and findings for each prompt type
