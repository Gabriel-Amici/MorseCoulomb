import numpy as np
from numba import njit, vectorize

@njit
def MsC_potential(alpha: float, r: float) -> float:

    """Calculates the 1D Morse-Coulomb potencial given the position r"""
    

    D = 1/alpha ; beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = -1/np.sqrt(r*r + alpha*alpha)
    else:
        pot = D*( np.exp( -2*beta*r) -2*np.exp( -beta*r ) )
    return pot


def MsC_potential_vec(alpha: float, r: np.ndarray) -> np.ndarray:
    """Calculates the 1D Morse-Coulomb potential for an array of positions r."""
    D = 1 / alpha
    beta = 1 / (alpha * np.sqrt(2))
    
    coulomb = -1 / np.sqrt(r * r + alpha * alpha)
    morse = D * (np.exp(-2 * beta * r) - 2 * np.exp(-beta * r))
    
    return np.where(r > 0, coulomb, morse)

@njit
def MsC_potential_1st_derivative(alpha: float, r: float) -> float:
    
    """Calculates the first order derivative of V(r) at a given point r"""

    D = 1/alpha ; beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = r*(r*r + alpha*alpha)**(-3/2)
    else:
        pot = 2*D*beta*np.exp( -2*beta*r)*( np.exp( beta*r ) - 1 )
    return pot

@njit
def MsC_return_points(alpha: float, E: float) -> np.ndarray:

    """Calculates the return points of a particle based on alpha and the total Energy"""
    
    rm = -alpha*np.sqrt(2)*np.log( np.sqrt( alpha*E + 1 ) + 1 )
    rM = np.sqrt( 1/(E**2) - alpha**2 )
    
    return np.array([rm, rM])


@njit
def MsC_total_energy( alpha: float, r: float, p: float ) -> float:

    return (p**2)/2 + MsC_potential(alpha, r)

