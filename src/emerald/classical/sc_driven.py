import numpy as np
from numba import njit
from ..potentials.sc_potential import sC_potential_1st_derivative, sC_return_points
from ..utils import external_field_scalar

@njit
def sC_diff_system(alpha: float, t: float, X: np.ndarray, field_params) -> np.ndarray:

    """Differencial Equations system whose solution describes motion of a particle in the Morse-Coulomb potential."""


    r, p = X[0], X[1]

    f1 = p
    f2 = - sC_potential_1st_derivative(alpha, r) - external_field_scalar(t, field_params)

    return np.array([ f1, f2 ])


@njit
def sC_driven_runge_kutta_4(alpha: float, t: float, X: np.ndarray, field_params, dt:float = 1.e-4) -> np.ndarray:

    """Order 4 Runge-Kutta method for solving differencial equation systems"""
    

    k1 = sC_diff_system(alpha, t, X, field_params)
    k2 = sC_diff_system(alpha, t + 0.5*dt, X + 0.5*dt*k1, field_params)
    k3 = sC_diff_system(alpha, t + 0.5*dt, X + 0.5*dt*k2, field_params)
    k4 = sC_diff_system(alpha, t + dt, X + dt*k3, field_params)

    return ( X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) )

@njit
def sC_driven_trajectory(alpha: float, time_array: np.ndarray, X0: np.ndarray, field_params, dt: float = 1.e-4) -> np.ndarray:

    time_step = time_array[1] - time_array[0]
    steps_per_time_step = int(round(time_step / dt)) if dt < time_step else 1
    integration_step = time_step / steps_per_time_step  # exact subdivision

    trajectory = np.zeros((len(time_array), len(X0)))
    trajectory[0] = X0

    for i in range(1, len(time_array)):
        intermediate_state = trajectory[i-1]
        for j in range(steps_per_time_step):
            time = time_array[i-1] + j * integration_step
            intermediate_state = sC_driven_runge_kutta_4(alpha, time, intermediate_state, field_params, integration_step)
        trajectory[i] = intermediate_state

    return trajectory