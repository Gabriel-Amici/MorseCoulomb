"""
Module for implementation of quadrature rules
"""

import numpy as np
from scipy.special import roots_legendre


def gauss_legendre_quadrature(func: callable, a: float, b: float, N: int) -> float:
    """
    Gauss-Legendre Quadrature for numerical integration.

    Parameters:
    func : callable
        The function to integrate.
    a : float
        The lower limit of integration.
    b : float
        The upper limit of integration.
    N : int
        The number of quadrature points (degree of the Legendre polynomial).
    """

    # Compute the roots and weights for the N-th degree Legendre polynomial
    roots, weights = roots_legendre(N)
    
    # Transform the roots to the interval [a, b]
    transformed_roots = 0.5 * (b - a) * roots + 0.5 * (b + a)
    
    # Evaluate the function at the transformed roots
    func_values = func(transformed_roots)
    
    # Compute the quadrature result
    result = 0.5 * (b - a) * np.dot(weights, func_values)
    
    return result

def simpson13(func: callable, a: float, b: float, N: int | None = None, dx: float = 1.e-5):
    """
    General Simpson 1/3 quadrature for callable `func`

    Parameters:
        func : callable
            The function to integrate.
        a : float
            The lower limit of integration.
        b : float
            The upper limit of integration.
        N : int | None
            The number of intervals. Overrides `dx` if provided. 
            Must be even for Simpson's rule.
        dx : float
            The extent of each interval.
    """
    if N is not None:
        n = N
    else:
        n = round((b - a) / dx)

    # Simpson 1/3 requires an even number of intervals
    if n % 2:
        n += 1

    x = np.linspace(a, b, n + 1)
    y = func(x)

    dx_actual = x[1] - x[0]

    return (dx_actual / 3) * (
        y[0]
        + y[-1]
        + 4 * np.sum(y[1:-1:2])
        + 2 * np.sum(y[2:-1:2])
    )