from ..potentials.coulomb_potential import C_energy_levels
from scipy.special import genlaguerre
import numpy as np

def Lag(n, x):
    """
    Associated Laguerre polynomial L^1_{n-1}(x)
    """
    # L_n-1^1 is a generalized Laguerre polynomial with alpha=1
    return genlaguerre(n-1, 1)(x)


def C_eigstate(r, n):
    """
    Computes the Coulomb wavefunction φ_n(r)
    """
    # Pre-factor
    prefactor = (2 * r) / np.sqrt(n**5)
    
    # Exponential term
    exp_term = np.exp(-r/n)
    
    # Laguerre polynomial term
    #laguerre_term = Lag(n, 2 * r / n)
    laguerre_term = genlaguerre(n-1, 1)(2*r/n)
    
    func = prefactor * exp_term * laguerre_term

    # Complete wavefunction
    return np.where(r>0, func, np.zeros_like(func))