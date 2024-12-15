# fizzbuzz.py

def fizzbuzz(number):
    """
    Returns "fizz", "buzz", "fizz buzz", or the number itself based on divisibility rules.

    - Returns "fizz" if the number is divisible by 3.
    - Returns "buzz" if the number is divisible by 5.
    - Returns "fizz buzz" if the number is divisible by both 3 and 5.
    - Otherwise, returns the number itself.

    Examples:
    >>> fizzbuzz(3)
    'fizz'
    >>> fizzbuzz(5)
    'buzz'
    >>> fizzbuzz(15)
    'fizz buzz'
    >>> fizzbuzz(7)
    7
    >>> fizzbuzz(30)
    'fizz buzz'
    >>> fizzbuzz(9)
    'fizz'
    >>> fizzbuzz(10)
    'buzz'
    >>> fizzbuzz(4)
    4
    """
    
    if not isinstance(number, int):
        raise ValueError("Input must be an integer.")
    
    if number % 15 == 0:
        return "fizz buzz"
    elif number % 3 == 0:
        return "fizz"
    elif number % 5 == 0:
        return "buzz"
    else:
        return number

# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()