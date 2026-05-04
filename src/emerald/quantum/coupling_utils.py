import numpy as np
from numba import njit
from tqdm import tqdm


@njit
def _evolve_step(state, exp_H0, exp_Xi, Xi_vecs, Xi_conjT):
    """Single split-operator step: U(state) = e^-iH₀δ/2 · V · e^-iεΞδ · V† · e^-iH₀δ/2 · state"""
    tmp = exp_H0 * state
    tmp = Xi_conjT @ tmp
    tmp = exp_Xi * tmp
    tmp = Xi_vecs @ tmp
    return exp_H0 * tmp


@njit
def _inverse_evolve_step(state, exp_H0, exp_Xi, Xi_vecs, Xi_conjT):
    """Inverse split-operator step: U†(state) = e^iH₀δ/2 · V · e^iεΞδ · V† · e^iH₀δ/2 · state"""
    tmp = exp_H0 * state
    tmp = Xi_conjT @ tmp
    tmp = exp_Xi * tmp
    tmp = Xi_vecs @ tmp
    return exp_H0 * tmp


def wavefunction_stationary_evolution(time_grid, energies, initial_state):

    N = len(initial_state)
    N_T = len(time_grid)
    delta_t = time_grid[1] - time_grid[0]

    wavefunc_history = np.empty((N, N_T), dtype=np.complex128)
    wavefunc_history[:, 0] = initial_state.astype(np.complex128)

    exp_H0 = np.exp(-1j * energies * delta_t)

    for tau in tqdm(range(1, N_T)):
        wavefunc_history[:, tau] = exp_H0 * wavefunc_history[:, tau - 1]
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
    # Cast to complex128: Numba requires matching dtypes for matrix-vector ops with complex states
    Xi_vectors_c = Xi_vectors.astype(np.complex128)

    return Xi_vectors_c, Xi_values, Xi_vectors_c.conj().T

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

@njit
def _wavefunction_evolution_core(N, N_T, delta_t, energies, Xi_vals, Xi_vecs, Xi_conjT, field_array, initial_state):
    """Numba-accelerated core of wavefunction_evolution. Returns (N, N_T) history array."""
    wavefunc_history = np.empty((N, N_T), dtype=np.complex128)
    wavefunc_history[:, 0] = initial_state

    exp_H0 = np.exp(-1j * energies * delta_t / 2)

    for tau in range(1, N_T):
        exp_Xi = np.exp(-1j * field_array[tau] * Xi_vals * delta_t)
        wavefunc_history[:, tau] = _evolve_step(
            wavefunc_history[:, tau - 1], exp_H0, exp_Xi, Xi_vecs, Xi_conjT
        )
    return wavefunc_history


def wavefunction_evolution(time_grid, energies, initial_state, Xi_values, Xi_vectors, Xi_conjT, field_array):
    """Evolve a wavefunction under a time-dependent field using the split-operator method.

    Parameters
    ----------
    time_grid : array
        1D time grid.
    energies : array
        Eigenvalues of H0.
    initial_state : array
        Initial wavefunction in the energy eigenbasis.
    Xi_values : array
        Eigenvalues of the interaction matrix Ξ.
    Xi_vectors : array
        Eigenvectors of Ξ (columns).
    Xi_conjT : array
        Conjugate transpose of Xi_vectors (= inverse for Hermitian Ξ).
    field_array : array
        Precomputed external field values at each time grid point.

    Returns
    -------
    wavefunc_history : array (N, N_T)
        Full evolution history.
    """
    N = len(initial_state)
    N_T = len(time_grid)
    delta_t = time_grid[1] - time_grid[0]

    return _wavefunction_evolution_core(
        N, N_T, delta_t, energies, Xi_values, Xi_vectors, Xi_conjT, field_array, initial_state
    )