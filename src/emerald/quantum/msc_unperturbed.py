import numpy as np
from numba import njit
from scipy.linalg import toeplitz
from ..potentials.msc_potential import MsC_potential_vec
from tqdm import tqdm


def MsC_hamiltonian(alpha, position_grid):
    # Grid setup
    
    N = len(position_grid)

    N += int(N % 2 == 0)  # Ensure N is odd
    
    
    r_grid = np.linspace(position_grid[0], position_grid[-1], N)
    L = position_grid[-1] - position_grid[0]
    
    n = (N - 1) // 2

    # Compute the kinetic energy first row
    l = np.arange(n)
    g = 2 * (np.pi * l / L) ** 2  # g[0] = 0 naturally
    theta = 2 * np.pi / (N - 1)
    k = np.arange(N)
    C = np.cos(l[:, None] * k * theta)  # Shape: (n, N)
    t = (2 / (N - 1)) * np.dot(g, C)  # First row of T

    # Construct Hamiltonian
    T = toeplitz(t)  # Symmetric Toeplitz matrix
    V = np.diag(MsC_potential_vec(alpha, r_grid))  # Diagonal potential
    H = T + V

    return H, position_grid


def MsC_eigstates(H, position_grid, base_size):

    delta_r = position_grid[1] - position_grid[0]

    # Eigendecomposition
    eig_energies, eig_states = np.linalg.eigh(H)
    sort_indexes = np.argsort(eig_energies)
    eig_energies = eig_energies[sort_indexes]
    eig_states = eig_states[:, sort_indexes] / np.sqrt(delta_r)

    return eig_energies[:base_size], eig_states[:, :base_size]
