# math_module.py
from collections import Counter
from typing import List, Any

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

def mean(data: List[float]) -> float:
    if not data:
        raise ValueError("Mean requires at least one data point.")
    return sum(data) / len(data)

def median(data: List[float]) -> float:
    if not data:
        raise ValueError("Median requires at least one data point.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    index = n // 2
    if n % 2:
        return sorted_data[index]
    return (sorted_data[index - 1] + sorted_data[index]) / 2

def mode(data: List[Any]) -> List[Any]:
    if not data:
        return []
    c = Counter(data)
    highest_freq = c.most_common(1)[0][1]
    return [k for k, v in c.items() if v == highest_freq]
