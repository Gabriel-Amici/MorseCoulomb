import numpy as np
from numba import njit, vectorize, prange
from collections import namedtuple

FieldParams = namedtuple('FieldParams', [
    'amplitude',
    'frequency',
    'form',           # 'cos', 'sin', 'tan', ...
    'envelope',       # 'linear', 'quadratic', 'cubic'
    'rampup_time',
    'rampdown_time',
    'operation_time', # -1 = runs forever
], defaults=[1.0, 1.0, 'cos', 'linear', 0.0, 0.0, -1.0])


@njit
def inter_period( y2 : float, y1 : float, yo : float, x1 : float, dx : float ): #interpolation function
    
    """interpolate points"""

    
    slope = ( y2 - y1 )/dx
    period = (yo - y1)/slope +  x1

    return period


@njit
def _envelope(t_norm: float, shape: str) -> float:
    """Normalized envelope: t_norm in [0,1] -> [0,1].
    Same shape used for rampup and rampdown."""
    if shape == 'linear':
        return t_norm
    elif shape == 'quadratic':
        return t_norm ** 2
    elif shape == 'cubic':
        return t_norm ** 3
    else:
        return t_norm  # fallback


@njit
def _oscillation(form: str, x: float) -> float:
    if form == 'cos':
        return np.cos(x)
    elif form == 'sin':
        return np.sin(x)
    elif form == 'tan':
        return np.tan(x)
    elif form == 'exp':
        return np.exp(x)
    elif form == 'log':
        return np.log(x)
    elif form == 'sqrt':
        return np.sqrt(x)
    elif form == 'abs':
        return np.abs(x)
    elif form == 'sign':
        return np.sign(x)
    else:
        return np.cos(x)  # fallback


@njit
def external_field_scalar(time: float, params) -> float:
    # hard zeros outside [0, operation_time]
    if time < 0.0:
        return 0.0
    if params.operation_time > 0.0 and time > params.operation_time:
        return 0.0

    # envelope coefficient
    if params.rampup_time > 0.0 and time < params.rampup_time:
        env = _envelope(time / params.rampup_time, params.envelope)
    elif params.operation_time > 0.0 and params.rampdown_time > 0.0 \
            and time > params.operation_time - params.rampdown_time:
        env = _envelope((params.operation_time - time) / params.rampdown_time, params.envelope)
    else:
        env = 1.0

    return params.amplitude * env * _oscillation(params.form, params.frequency * time)


@njit(parallel=True)
def external_field_array(time: np.ndarray, params) -> np.ndarray:
    out = np.empty(len(time))
    for i in prange(len(time)):
        out[i] = external_field_scalar(time[i], params)
    return out

@njit
def external_field(F0, omg, t):
    return F0 * np.cos(omg*t)

@njit
def chebyshev_nodes(a: float, b: float, N: int) -> np.ndarray:
    """
    Generate Chebyshev nodes for numerical interpolation over an interval [a, b].
    
    Chebyshev nodes are the roots of Chebyshev polynomials of the first kind,
    mapped to the interval [a, b]. They are useful for polynomial interpolation
    as they minimize Runge's phenomenon (oscillations at interval edges).
    
    Parameters
    ----------
    a : float
        Lower bound of the interval
    b : float
        Upper bound of the interval (must be greater than a)
    N : int
        Number of nodes minus 1 (N+1 total nodes will be generated)
    
    Returns
    -------
    np.ndarray
        Array of N+1 Chebyshev nodes distributed over [a, b]
    
    Examples
    --------
    >>> Chebyshev_nodes(-1, 1, 4)
    array([ 1.        ,  0.70710678,  0.        , -0.70710678, -1.        ])
    >>> Chebyshev_nodes(0, 2, 2)
    array([2. , 1. , 0. ])
    """
    delta_theta = np.pi/N
    thetas = np.array([i*delta_theta for i in range(N+1)])

    r = (b-a)/2
    mu = (b+a)/2

    nodes = mu + r*np.cos(thetas)

    return nodes


@njit
def lagrange_polynomial(x: float, j: int, points: np.ndarray) -> float:
    """
    Compute the j-th Lagrange basis polynomial evaluated at point x.
    
    This function implements a single Lagrange basis polynomial used in
    Lagrange interpolation, which passes through 1 at points[j, 0] and 0
    at all other points[m, 0] where m != j.
    
    Parameters
    ----------
    x : float
        Point at which to evaluate the polynomial
    j : int
        Index of the basis polynomial (0 <= j < n)
    points : np.ndarray
        2D array of shape (n, 2) containing n points, where
        points[:, 0] are x-coordinates and points[:, 1] are y-coordinates
    
    Returns
    -------
    float
        Value of the j-th Lagrange basis polynomial at x
    
    Examples
    --------
    >>> points = np.array([[0, 0], [1, 1], [2, 4]])
    >>> lagrange_polynomial(0.5, 1, points)
    # Returns value of L_1(0.5) for points (0,0), (1,1), (2,4)
    """
    n = len(points)
    res = 1
    for m in range(n):
        if m != j:
            res *= (x - points[m, 0])/(points[j, 0] - points[m, 0])
    return res


@njit
def polynomial_fit(r: float, points: np.ndarray) -> float:
    """
    Evaluate a Lagrange interpolating polynomial at point r.
    
    Constructs and evaluates the full Lagrange interpolating polynomial
    that passes through all given points using the Lagrange basis polynomials.
    
    Parameters
    ----------
    r : float
        Point at which to evaluate the interpolating polynomial
    points : np.ndarray
        2D array of shape (n, 2) containing n points, where
        points[:, 0] are x-coordinates and points[:, 1] are y-coordinates
    
    Returns
    -------
    float
        Value of the interpolating polynomial at r
    
    Examples
    --------
    >>> points = np.array([[0, 0], [1, 1], [2, 4]])
    >>> polynomial_fit(1.5, points)
    # Returns interpolated value at r = 1.5
    """
    n = len(points)
    res = 0
    for j in range(n):
        res += (points[j, 1])*lagrange_polynomial(r, j, points)
    return res

VEC_P = vectorize(polynomial_fit)


@njit
def bisection_method(angle: float, points: np.ndarray, a: float, b: float, tol: float = 1.e-7, max_iter: int = 1000) -> float:
    """
    Find a root of the equation polynomial_fit(r, points) - angle = 0 using bisection method.
    
    This function solves for r where the interpolating polynomial equals a given angle,
    using the bisection method over the interval [a, b].
    
    Parameters
    ----------
    angle : float
        Target value to find the root for (i.e., solve P(r) = angle)
    points : np.ndarray
        2D array of shape (n, 2) containing n points, where
        points[:, 0] are x-coordinates and points[:, 1] are y-coordinates
    a : float
        Lower bound of the search interval
    b : float
        Upper bound of the search interval (must be greater than a)
    tol : float, optional
        Tolerance for convergence (default: 1.e-7)
    max_iter : int, optional
        Maximum number of iterations (default: 1000)
    
    Returns
    -------
    float
        Approximate root r where polynomial_fit(r, points) ≈ angle
    
    Examples
    --------
    >>> points = np.array([[0, 0], [1, 1], [2, 4]])
    >>> bisection_method(2.25, points, 0, 2)
    # Returns r where P(r) ≈ 2.25 in [0, 2]
    """
    f = lambda r : polynomial_fit(r, points) - angle

    if abs(f(a)) < tol:
        return a
    elif abs(f(b)) < tol:
        return b

    iterations = 0

    while (b - a) / 2 > tol and iterations < max_iter:
        c = (a + b) / 2
        if abs(f(c)) < tol:
            return c

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

        iterations += 1

    x0 = (a + b) / 2
    
    return x0

@njit
def check_no_duplicates(arr):
    # Convert the array to a set
    unique_elements = set(arr)
    
    # Compare the lengths
    return len(unique_elements) == len(arr)


def diagonalize_matrix(M):
    M_values, M_vectors = np.linalg.eig(M)
    return M_vectors, M_values, np.linalg.inv(M_vectors)