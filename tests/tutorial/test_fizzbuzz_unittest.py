import unittest

def fizzbuzz(number):
    if number % 15 == 0:
        return "fizz buzz"
    elif number % 3 == 0:
        return "fizz"
    elif number % 5 == 0:
        return "buzz"
    else:
        return number

class TestFizzBuzz(unittest.TestCase):
    def test_fizzbuzz_multiple_of_15(self):
        self.assertEqual(fizzbuzz(15), "fizz buzz")
        self.assertEqual(fizzbuzz(30), "fizz buzz")
        self.assertEqual(fizzbuzz(45), "fizz buzz")

    def test_fizzbuzz_multiple_of_3(self):
        self.assertEqual(fizzbuzz(3), "fizz")
        self.assertEqual(fizzbuzz(6), "fizz")
        self.assertEqual(fizzbuzz(9), "fizz")

    def test_fizzbuzz_multiple_of_5(self):
        self.assertEqual(fizzbuzz(5), "buzz")
        self.assertEqual(fizzbuzz(10), "buzz")
        self.assertEqual(fizzbuzz(20), "buzz")

    def test_fizzbuzz_non_multiple(self):
        self.assertEqual(fizzbuzz(1), 1)
        self.assertEqual(fizzbuzz(2), 2)
        self.assertEqual(fizzbuzz(4), 4)

    def test_fizzbuzz_edge_cases(self):
        self.assertEqual(fizzbuzz(0), "fizz buzz")  # Edge case: 0 is divisible by all numbers

if __name__ == "__main__":
    unittest.main()