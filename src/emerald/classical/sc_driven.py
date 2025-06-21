import numpy as np
from numba import njit
from ..potentials.sc_potential import sC_potential_1st_derivative, sC_return_points
from ..utils import external_field

@njit
def sC_diff_system(alpha: float, t: float, X: np.ndarray, F_0: float, Omg: float) -> np.ndarray:

    """Differencial Equations system whose solution describes motion of a particle in the Morse-Coulomb potential."""


    r, p = X[0], X[1]

    f1 = p
    f2 = - sC_potential_1st_derivative(alpha, r) - external_field(F_0, Omg, t)

    return np.array([ f1, f2 ])


@njit('f8[:](f8, f8, f8[:], f8, f8, f8)')
def sC_driven_runge_kutta_4(alpha: float, t: float, X: np.ndarray, F_0: float, Omg: float, dt:float = 1.e-4) -> np.ndarray:

    """Order 4 Runge-Kutta method for solving differencial equation systems"""
    

    k1 = sC_diff_system(alpha, t,         X,            F_0, Omg)
    k2 = sC_diff_system(alpha, t + 0.5*dt, X + 0.5*dt*k1, F_0, Omg)
    k3 = sC_diff_system(alpha, t + 0.5*dt, X + 0.5*dt*k2, F_0, Omg)
    k4 = sC_diff_system(alpha, t + dt,     X + dt*k3,     F_0, Omg)

    return ( X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) )



