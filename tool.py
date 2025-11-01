"""A tool module: small helpers used as callable tools by the Agent. Define simple arithmetic functions to be used as tools by the Agent. """

# Addition


def addition(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b

# Subtraction


def subtraction(a: int, b: int) -> int:
    """Return a - b."""
    return a - b

# Multiplication


def multiplication(a: int, b: int) -> int:
    """Return the product of a and b."""
    return a * b

# Division


def division(a: int, b: int) -> float:
    """Return a / b as a float. Raises ZeroDivisionError if b == 0."""
    if b == 0:
        raise ZeroDivisionError("division by zero")
    return a / b
