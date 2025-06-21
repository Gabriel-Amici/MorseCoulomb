import numpy as np
from numba import njit
from .msc_potential import MsC_return_points

@njit
def C_potential(r: float) -> float:

    """Calculates the 1D Coulomb potencial given the position r"""

    if r > 0:
        pot = -1/r
    else:
        pot = np.inf
    return pot

@njit
def C_potential_vec(r: float) -> float:

    """Calculates the 1D Coulomb potencial given the position r"""

    return np.where(r>0, -1/r, np.inf)

@njit
def C_return_points(E: float) -> np.ndarray:
    return MsC_return_points(0, E)

@njit
def C_transformed_return_points(Ei: float) -> np.ndarray:

    """Calculates the return points of a the transformed hamiltonian of a Coulomb potential based on the total Energy"""
    
    K = 4
    p2 = -Ei

    q1M = np.sqrt( K/(4*p2) )
    q1m = -q1M
    
    return np.array([q1m, q1M])


@njit
def C_total_energy(r: float, p: float) -> float:

    return (p**2)/2 + C_potential(r)


@njit
def C_energy_levels( n: int = 1 ) -> float:

    """ Calculates the energy at level n using Balmer series"""

    return -1/(2*n**2)
