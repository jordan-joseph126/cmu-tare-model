# test_math_module.py
import unittest
from math_module import add, subtract, multiply, divide, mean, median, mode

class TestMathFunctions(unittest.TestCase):
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

    # Optional: Additional tests for robustness
    def test_data_types(self):
        # Testing with different data types has been integrated into other test methods
        pass  # This method can be removed or kept for additional data type tests

if __name__ == '__main__':
    unittest.main()