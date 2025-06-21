import numpy as np
from numba import njit, vectorize

@njit
def sC_potential(alpha: float, r: float) -> float:

    """Calculates the 1D soft-Coulomb potencial given the position r"""

    pot = -1/np.sqrt(r*r + alpha*alpha)
    return pot


def sC_potential_vec(alpha: float, r: np.ndarray) -> np.ndarray:
    """Calculates the 1D Morse-Coulomb potential for an array of positions r."""
    
    soft_coulomb = -1 / np.sqrt(r * r + alpha * alpha)
    return soft_coulomb

@njit
def sC_potential_1st_derivative(alpha: float, r: float) -> float:
    
    """Calculates the first order derivative of V(r) at a given point r"""

    pot = r*(r*r + alpha*alpha)**(-3/2)
    return pot

@njit
def sC_return_points(alpha: float, E: float) -> np.ndarray:

    """Calculates the return points of a particle based on alpha and the total Energy"""
    
    rM = np.sqrt( 1/(E**2) - alpha**2 )
    rm = -rM
    
    return np.array([rm, rM])


@njit
def sC_total_energy( alpha: float, r: float, p: float ) -> float:

    return (p**2)/2 + sC_potential(alpha, r)

