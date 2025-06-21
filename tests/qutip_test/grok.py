import numpy as np
import scipy.sparse as sparse
import qutip as qt

def MsC_potential_vec(alpha, r):
    """
    Vectorized Morse-Coulomb potential for a 1D quantum system.
    
    Args:
        alpha (float): Potential parameter controlling the shape.
        r (np.ndarray): Position array.
    
    Returns:
        np.ndarray: Potential values at each position.
    """
    D = 1 / alpha
    beta = 1 / (alpha * np.sqrt(2))
    return np.where(r > 0,
                    -1 / np.sqrt(r**2 + alpha**2),
                    D * (np.exp(-2 * beta * r) - 2 * np.exp(-beta * r)))

def MsC_return_points(alpha: float, E: float) -> np.ndarray:
    """
    Calculates the return points of a particle based on alpha and the total Energy.
    
    Args:
        alpha (float): Potential parameter.
        E (float): Energy level for classical turning points.
    
    Returns:
        np.ndarray: Array of [rm, rM], the minimum and maximum turning points.
    """
    rm = -alpha * np.sqrt(2) * np.log(np.sqrt(alpha * E + 1) + 1)
    rM = np.sqrt(1 / (E**2) - alpha**2)
    return np.array([rm, rM])

def compute_spectrum(alpha, N, E_ref=-0.5, extend_factor=2.0, k=10):
    """
    Compute the energy spectrum for the Morse-Coulomb potential using QuTiP,
    with a grid based on classical turning points.
    
    Args:
        alpha (float): Potential parameter.
        N (int): Number of grid points.
        E_ref (float): Reference energy to estimate turning points (default: -0.5).
        extend_factor (float): Factor to extend grid beyond turning points (default: 2.0).
        k (int): Number of eigenvalues to compute (default: 10).
    
    Returns:
        tuple: (energies, x), where energies is the spectrum and x is the grid.
    """
    # Compute classical turning points for the reference energy
    rm, rM = MsC_return_points(alpha, E_ref)
    
    # Extend the grid beyond turning points to capture wavefunction decay
    L_left = rm * extend_factor  # Negative side (Morse decays exponentially)
    L_right = rM * extend_factor # Positive side (Coulomb-like decay)
    
    # Define asymmetric position grid
    x = np.linspace(L_left, L_right, N)
    dx = x[1] - x[0]  # Grid spacing
    
    # Potential energy operator (diagonal matrix)
    V_diag = MsC_potential_vec(alpha, x)
    V = sparse.diags(V_diag)
    
    # Kinetic energy operator: T = - (1/2) d²/dx²
    diagonal = np.ones(N) * (1 / dx**2)          # Diagonal: 1/dx²
    off_diagonal = np.ones(N-1) * (-0.5 / dx**2) # Off-diagonals: -1/(2 dx²)
    T = sparse.diags([off_diagonal, diagonal, off_diagonal], [-1, 0, 1])
    
    # Total Hamiltonian: H = T + V
    H = T + V
    
    # Convert to QuTiP quantum object
    H_qt = qt.Qobj(H)
    
    # Compute the k lowest eigenvalues
    energies = H_qt.eigenenergies(sparse=True, eigvals=k)
    
    return energies, x

# Example usage
if __name__ == "__main__":
    # Parameters
    alpha = 0.001     # Potential parameter
    N = 2000         # Number of grid points
    E_ref = -0.005    # Reference energy (e.g., near expected ground state)
    extend_factor = 2.0  # Extend grid by this factor beyond turning points
    k = 10          # Number of energy levels to compute
    
    # Compute the energy spectrum and get the grid
    energies, x_grid = compute_spectrum(alpha, N, E_ref, extend_factor, k)
    
    # Print results
    print(f"Grid range: [{x_grid[0]:.3f}, {x_grid[-1]:.3f}]")
    print(f"Energy spectrum for alpha = {alpha}:")
    for i, energy in enumerate(energies):
        print(f"E_{i} = {energy:.6f}")
