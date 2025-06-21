import numpy as np
from tqdm import tqdm
from ..utils import  external_field

def wavefunction_stationary_evolution(time_grid, energies, initial_state):
    
    N = len(initial_state)
    N_T = len(time_grid)
    delta_t = time_grid[1] - time_grid[0]

    wavefunc_history = np.empty( (N, N_T), dtype=np.complex128 )
    wavefunc_history[:, 0] = initial_state.astype(np.complex128)

    exp_H0 = np.exp( -1j*energies*delta_t )

    for tau in tqdm(range(1, N_T)):
        wavefunc_history[:, tau] = exp_H0*wavefunc_history[:, tau-1] #multiplicação pela exp(H0)
    return wavefunc_history


def interaction_matrix(position_grid, base_states):
    N = len(position_grid)
    dr = position_grid[1] - position_grid[0]
    
    # Compute weights
    w = np.full(N, 2.0)
    w[1::2] = 4.0
    w[0] = 1.0
    w[-1] = 1.0
    
    # Efficient matrix computation
    temp = (w * position_grid)[:, None] * base_states
    Xi = (dr / 3) * np.dot(base_states.T, temp)

    Xi_values, Xi_vectors = np.linalg.eigh(Xi)

    return Xi_vectors, Xi_values, np.linalg.inv(Xi_vectors)

def braket(position_grid, states, wavefunction):
    dr = position_grid[1] - position_grid[0]
    position_grid = np.ones_like(position_grid)
    N = len(position_grid)
    
    # Compute weights
    w = np.full(N, 2.0)
    w[1::2] = 4.0
    w[0] = 1.0
    w[-1] = 1.0
    
    # Efficient matrix computation
    temp = (w * position_grid)[:, None] * wavefunction
    coefficients = (dr / 3) * np.dot(states.T, temp)

    return coefficients

import numpy as np

def expansion_coefficients(position_grid, states, psi):
    """
    Calculate the coefficients of expansion of a wavefunction in a basis of states.
    
    Parameters:
    - position_grid: 1D array of shape (N,), the position grid points.
    - states: 2D array of shape (N, M), basis states (each column is a state).
    - psi: 1D array of shape (N,), the wavefunction.
    
    Returns:
    - c: 1D array of shape (M,), the expansion coefficients.
    """
    # Number of grid points
    N = len(position_grid)
    
    # Grid spacing (assumes uniform grid)
    dr = position_grid[1] - position_grid[0]
    
    # Define Simpson's rule weights
    w = np.full(N, 2.0)  # Default weight is 2
    w[1::2] = 4.0       # Odd indices get weight 4
    w[0] = 1.0          # First point gets weight 1
    w[-1] = 1.0         # Last point gets weight 1
    w *= dr / 3.0       # Scale by dr/3 for Simpson's rule
    
    # Compute the integrand: conjugate of states times psi
    # states.conj() has shape (N, M), psi[:, None] has shape (N, 1)
    integrand = np.conj(states) * psi[:, None]
    
    # Compute coefficients by summing over the grid with weights
    # w[:, None] has shape (N, 1), integrand has shape (N, M)
    c = np.sum(w[:, None] * integrand, axis=0)
    
    return c

def wavefunction_evolution(time_grid, energies, initial_state, Xi_values, Xi_vectors, Xi_inv, F_0, Omg):
    
    n = len(initial_state)
    N_T = len(time_grid)
    delta_t = time_grid[1] - time_grid[0]

    wavefunc_history = np.empty( (n, N_T), dtype=np.complex128 )
    wavefunc_history[:, 0] = initial_state.astype(np.complex128)

    exp_H0 = np.exp( -1j*energies*delta_t/2 )

    temp_wavefunc = np.empty(n, dtype=np.complex128)

    for tau in tqdm(range(1, N_T)):
        t = time_grid[tau]
        exp_Xi = np.exp( -1j*external_field(F_0, Omg, t + delta_t/2)*Xi_values*delta_t)
        
        temp_wavefunc = exp_H0*wavefunc_history[:, tau-1] #multiplicação pela exp(H0)

        temp_wavefunc_2 = Xi_inv@temp_wavefunc #multiplicação pela inversa dos autovetores de Xi 

        temp_wavefunc_2 *= exp_Xi #multiplicação pelo exponencial da Xi diagonalizada

        temp_wavefunc_3 = Xi_vectors@temp_wavefunc_2 #multiplicação pelos autovetores de Xi

        temp_wavefunc_3 *= exp_H0

        wavefunc_history[:, tau] = temp_wavefunc_3.copy()

    return wavefunc_history