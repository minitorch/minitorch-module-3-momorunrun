"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers"""
    return x * y
    # raise NotImplementedError


def id(x: float) -> float:
    """Returns the input unchanged"""
    return x
    # raise NotImplementedError


def add(x: float, y: float) -> float:
    """Adds two numbers"""
    return x + y
    # raise NotImplementedError


def neg(x: float) -> float:
    """Negates a number"""
    return -x
    # raise NotImplementedError


def lt(x: float, y: float) -> bool:
    """Checks if one umber is less than another"""
    return x < y
    # raise NotImplementedError


def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal"""
    return x == y
    # raise NotImplementedError


def max(x: float, y: float) -> float:
    """Returns the larger of two numbers"""
    # return x if x > y else y
    if x > y:
        return x
    else:
        return y
    # raise NotImplementedError


def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value"""
    return abs(x - y) < 1e-2
    # raise NotImplementedError


def sigmoid(x: float) -> float:
    """Calculates the sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1 + math.exp(x))
    # raise NotImplementedError


def relu(x: float) -> float:
    """Applies the ReLU activation function"""
    # return max(0.0, x)
    return x if x > 0.0 else 0.0
    # raise NotImplementedError


def log(x: float) -> float:
    """Calculates the natural logarithm"""
    return math.log(x)
    # raise NotImplementedError


def exp(x: float) -> float:
    """Calculates the exponential function"""
    return math.exp(x)
    # raise NotImplementedError


def inv(x: float) -> float:
    """Calculates the reciprocal"""
    return 1 / x
    # raise NotImplementedError


def log_back(x: float, k: float) -> float:
    """Computes the derivative of log times a second arg"""
    return k / x
    # raise NotImplementedError


def inv_back(x: float, k: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -1.0 * k / (x**2)
    # raise NotImplementedError


def relu_back(x: float, d: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    return d if x > 0 else 0
    # raise NotImplementedError


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float], x: Iterable[float]) -> Iterable[float]:
    """Higher-order function that applies a given function to each element of an iterable"""
    # def map(fn: Callable[[float], float], x: Iterable[float]):
    return [fn(a) for a in x]
    # raise NotImplementedError


def zipWith(
    fn: Callable[[float, float], float], x: Iterable[float], y: Iterable[float]
) -> Iterable[float]:
    """Higher-order function that combines elements from two iterables using a given function"""
    return [fn(a, b) for a, b in zip(x, y)]
    # raise NotImplementedError


def reduce(fn: Callable[[float, float], float], start: float, x: Iterable) -> float:
    """Higher-order function that reduces an iterable to a single value using a given function"""
    acc = start
    for a in x:
        acc = fn(acc, a)
    return acc
    # raise NotImplementedError


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg, ls)
    # raise NotImplementedError


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add, ls1, ls2)
    # raise NotImplementedError


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, 0, ls)
    # raise NotImplementedError


def prod(ls: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, 1.0, ls)
    # raise NotImplementedError
