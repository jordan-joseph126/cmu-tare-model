# tests/test_fizzbuzz.py

import pytest
from fizzbuzz import fizzbuzz

@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        # Test cases divisible by 15 (both 3 and 5)
        (15, "fizz buzz"),
        (30, "fizz buzz"),
        (45, "fizz buzz"),
        (60, "fizz buzz"),
        (0, "fizz buzz"),  # Edge case: zero is divisible by any non-zero integer

        # Test cases divisible by 3 only
        (3, "fizz"),
        (6, "fizz"),
        (9, "fizz"),
        (12, "fizz"),
        (-3, "fizz"),      # Negative number divisible by 3
        (-6, "fizz"),

        # Test cases divisible by 5 only
        (5, "buzz"),
        (10, "buzz"),
        (20, "buzz"),
        (25, "buzz"),
        (-5, "buzz"),      # Negative number divisible by 5
        (-10, "buzz"),

        # Test cases not divisible by 3 or 5
        (1, 1),
        (2, 2),
        (4, 4),
        (7, 7),
        (11, 11),
        (14, 14),
        (16, 16),
        (17, 17),
        (-1, -1),          # Negative number not divisible by 3 or 5
        (-2, -2),
        (-4, -4),
    ]
)
def test_fizzbuzz(input_value, expected_output):
    assert fizzbuzz(input_value) == expected_output

@pytest.mark.parametrize(
    "invalid_input",
    [
        "15",    # String input
        3.5,     # Float input
        None,    # NoneType
        [],      # Empty list
        {},      # Empty dict
        [3],     # List with integer
        (5,),    # Tuple with integer
    ]
)
# Add invalid input tests
def test_fizzbuzz_invalid_inputs(invalid_input):
    with pytest.raises(ValueError) as exc_info:
        fizzbuzz(invalid_input)
    assert str(exc_info.value) == "Input must be an integer."

# Add doctest integration
def test_doctests():
    import doctest
    from fizzbuzz import fizzbuzz

    result = doctest.testmod(fizzbuzz)
    assert result.failed == 0, f"Doctests failed: {result}"