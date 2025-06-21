import numpy as np
from numba import njit, vectorize

@njit
def nMsC_potential(alpha: float, r: float) -> float:

    """Calculates the 1D Morse-Coulomb potencial given the position r"""
    

    beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = -alpha/np.sqrt(r*r + alpha*alpha)
    else:
        pot = ( np.exp( -2*beta*r) -2*np.exp( -beta*r ) )
    return pot


def nMsC_potential_vec(alpha: float, r: np.ndarray) -> np.ndarray:
    """Calculates the 1D Morse-Coulomb potential for an array of positions r."""
    beta = 1 / (alpha * np.sqrt(2))
    
    coulomb = -alpha/ np.sqrt(r * r + alpha * alpha)
    morse = (np.exp(-2 * beta * r) - 2 * np.exp(-beta * r))
    
    return np.where(r > 0, coulomb, morse)

@njit
def nMsC_potential_1st_derivative(alpha: float, r: float) -> float:
    
    """Calculates the first order derivative of V(r) at a given point r"""

    beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = alpha*r*(r*r + alpha*alpha)**(-3/2)
    else:
        pot = 2*beta*np.exp( -2*beta*r)*( np.exp( beta*r ) - 1 )
    return pot

@njit
def nMsC_return_points(alpha: float, E: float) -> np.ndarray:

    """Calculates the return points of a particle based on alpha and the total Energy"""
    
    rm = -alpha*np.sqrt(2)*np.log( np.sqrt( E + 1 ) + 1 )
    rM = alpha*np.sqrt( 1/(E**2) - 1 )
    
    return np.array([rm, rM])


@njit
def nMsC_total_energy( alpha: float, r: float, p: float ) -> float:

    return (p**2)/2 + nMsC_potential(alpha, r)

