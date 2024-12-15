# test_math_module_updated.py
import unittest
from math_module import add, subtract, multiply, divide, mean, median, mode

class TestArithmeticFunctions(unittest.TestCase):
    """Test cases for arithmetic functions: add, subtract, multiply, divide."""

    # Tests for the add function
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)
        self.assertEqual(add(1.5, 2.5), 4.0)
        self.assertEqual(add(-1.5, -2.5), -4.0)
        # Additional tests for different data types
        self.assertEqual(add("Hello ", "World"), "Hello World")
        self.assertEqual(add([1, 2], [3, 4]), [1, 2, 3, 4])

    # Tests for the subtract function
    def test_subtract(self):
        self.assertEqual(subtract(5, 3), 2)
        self.assertEqual(subtract(0, 0), 0)
        self.assertEqual(subtract(-1, -1), 0)
        self.assertEqual(subtract(2.5, 1.5), 1.0)
        self.assertEqual(subtract(-2.5, -1.5), -1.0)
        # Additional test for invalid data types
        with self.assertRaises(TypeError):
            subtract("Hello", 5)

    # Tests for the multiply function
    def test_multiply(self):
        self.assertEqual(multiply(2, 3), 6)
        self.assertEqual(multiply(-1, 1), -1)
        self.assertEqual(multiply(0, 100), 0)
        self.assertEqual(multiply(2.5, 4), 10.0)
        self.assertEqual(multiply(-2.5, -4), 10.0)
        # Additional tests for different data types
        self.assertEqual(multiply("Repeat", 3), "RepeatRepeatRepeat")
        self.assertEqual(multiply([1], 3), [1, 1, 1])

    # Tests for the divide function
    def test_divide(self):
        self.assertEqual(divide(10, 2), 5)
        self.assertEqual(divide(-6, 3), -2)
        self.assertAlmostEqual(divide(5, 2), 2.5)
        self.assertAlmostEqual(divide(-5, -2), 2.5)
        with self.assertRaises(ValueError):
            divide(5, 0)
        with self.assertRaises(ValueError):
            divide(0, 0)
        # Additional test for invalid data types
        with self.assertRaises(TypeError):
            divide("10", 2)

class TestStatisticalFunctions(unittest.TestCase):
    """Test cases for statistical functions: mean, median, mode."""

    # Tests for the mean function
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4, 5]), 3)
        self.assertEqual(mean([2, 2, 2, 2]), 2)
        self.assertAlmostEqual(mean([1.5, 2.5, 3.5]), 2.5)
        self.assertAlmostEqual(mean([-1, 1]), 0)
        with self.assertRaises(ValueError):
            mean([])

    # Tests for the median function
    def test_median(self):
        self.assertEqual(median([1, 3, 2, 4, 5]), 3)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(median([1.5, 3.5, 2.5]), 2.5)
        self.assertEqual(median([7]), 7)
        self.assertAlmostEqual(median([-1, -2, -3, -4]), -2.5)
        with self.assertRaises(ValueError):
            median([])
        # Additional test for even-length list with non-numeric data
        with self.assertRaises(TypeError):
            median(["x", "y", "z", "w"])

    # Tests for the mode function
    def test_mode(self):
        self.assertEqual(mode([1, 2, 2, 3, 3]), [2, 3])
        self.assertEqual(mode([1, 1, 1, 2, 3]), [1])
        self.assertEqual(mode([4]), [4])
        self.assertEqual(mode([]), [])  # Assuming mode([]) returns []
        self.assertEqual(mode([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])  # All unique
        self.assertEqual(mode([0, 0, 0, 0]), [0])
        # Additional test for non-numeric data
        self.assertEqual(mode(["apple", "banana", "apple"]), ["apple"])

    # Optional: Additional tests for robustness can be added here

# Creating Test Suites using unittest.TestLoader.loadTestsFromTestCase()
def arithmetic_suite():
    """Test suite for arithmetic functions."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestArithmeticFunctions)
    return suite

def statistical_suite():
    """Test suite for statistical functions."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestStatisticalFunctions)
    return suite

def all_tests_suite():
    """Combined test suite for all functions."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(arithmetic_suite())
    suite.addTests(statistical_suite())
    return suite

# Running the Test Suites
if __name__ == '__main__':
    # Create a test runner that will run the test suites
    runner = unittest.TextTestRunner(verbosity=2)

    # Run Arithmetic Functions Test Suite
    print("\nRunning Arithmetic Functions Test Suite:")
    runner.run(arithmetic_suite())

    # Run Statistical Functions Test Suite
    print("\nRunning Statistical Functions Test Suite:")
    runner.run(statistical_suite())

    # Alternatively, to run all tests together:
    # print("\nRunning All Test Suites:")
    # runner.run(all_tests_suite())