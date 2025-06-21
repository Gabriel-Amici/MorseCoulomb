import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

def MsC_potential(alpha: float, r: float) -> float:
    """Calculates the 1D Morse-Coulomb potential given the position r"""
    D = 1/alpha
    beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = -1/np.sqrt(r*r + alpha*alpha)
    else:
        pot = D*(np.exp(-2*beta*r) - 2*np.exp(-beta*r))
    return pot

def MsC_return_points(alpha: float, E: float) -> np.ndarray:
    """Calculates the return points of a particle based on alpha and the total Energy"""
    rm = -alpha*np.sqrt(2)*np.log(np.sqrt(alpha*E + 1) + 1)
    rM = np.sqrt(1/(E**2) - alpha**2)
    
    return np.array([rm, rM])

def create_adaptive_grid(alpha, energy_range, n_points, padding_factor=1.5):
    """
    Create an adaptive grid based on classical return points for multiple energies
    
    Parameters:
    -----------
    alpha : float
        Parameter for the Morse-Coulomb potential
    energy_range : list or array
        Energies to consider for determining grid extent
    n_points : int
        Number of grid points
    padding_factor : float
        Factor to extend the grid beyond the return points
        
    Returns:
    --------
    grid : ndarray
        Position grid optimized for the potential
    """
    # Find minimum and maximum return points across all energies
    r_min = float('inf')
    r_max = float('-inf')
    
    print("Calculating return points for grid construction:")
    for energy in energy_range:
        try:
            rm, rM = MsC_return_points(alpha, energy)
            print(f"  Energy {energy:.4f}: r_min = {rm:.4f}, r_max = {rM:.4f}")
            r_min = min(r_min, rm)
            r_max = max(r_max, rM)
        except Exception as e:
            print(f"  Could not calculate return points for E={energy}: {e}")
    
    # Apply padding
    grid_range = r_max - r_min
    r_min = r_min
    r_max = r_max + (padding_factor - 1) * grid_range
    
    print(f"Final grid: r_min = {r_min:.4f}, r_max = {r_max:.4f}")
    
    # Create uniform grid
    grid = np.linspace(r_min, r_max, n_points)
    
    return grid

def evaluate_potential_on_grid(grid, potential_func, alpha):
    """
    Evaluate the potential function on the grid points
    """
    V = np.zeros_like(grid)
    for i, r in enumerate(grid):
        V[i] = potential_func(alpha, r)
    return V

def fourier_grid_hamiltonian_1d(grid, potential_values, mass=1.0, n_eigvals=10):
    """
    Implements the 1D Fourier Grid Hamiltonian method with a custom grid
    
    Parameters:
    -----------
    grid : ndarray
        Position grid
    potential_values : ndarray
        Potential energy values at each grid point
    mass : float, optional
        Particle mass (default: 1.0, atomic units)
    n_eigvals : int, optional
        Number of eigenvalues/eigenfunctions to compute
        
    Returns:
    --------
    eigvals : ndarray
        Eigenvalues (energies)
    eigvecs : ndarray
        Eigenvectors (wavefunctions)
    """
    n_points = len(grid)
    
    # Calculate grid spacing (can be non-uniform)
    dx = grid[1] - grid[0]
    
    # Check if grid is uniform
    if not np.allclose(np.diff(grid), dx):
        raise ValueError("This implementation requires a uniform grid")
    
    # Calculate potential energy matrix (diagonal in position basis)
    V = np.diag(potential_values)
    
    # Set up kinetic energy operator using Fourier method
    def kinetic_operator(wavefunction):
        # Define the wavenumbers for the FFT
        k = 2.0 * np.pi * fftpack.fftfreq(n_points, dx)
        
        # Perform FFT, multiply by k^2, and perform inverse FFT
        ft_wavefunction = fftpack.fft(wavefunction)
        ft_wavefunction = ft_wavefunction * (k**2 / (2.0 * mass))
        return fftpack.ifft(ft_wavefunction).real
    
    # Create the Hamiltonian matrix
    H = np.zeros((n_points, n_points), dtype=complex)
    
    # For each column of the Hamiltonian
    for i in range(n_points):
        # Create a basis vector (delta function at position i)
        basis_vector = np.zeros(n_points)
        basis_vector[i] = 1.0
        
        # Apply kinetic operator to the basis vector
        K_basis = kinetic_operator(basis_vector)
        
        # Store the result in the Hamiltonian matrix
        H[:, i] = K_basis
    
    # Make sure the kinetic energy matrix is Hermitian
    H = 0.5 * (H + H.conj().T)
    
    # Add the potential energy
    H = H + V
    
    # Solve the eigenvalue problem using np.linalg.eigh (more robust than eigsh)
    # Convert to real if the imaginary part is negligible
    if np.allclose(H.imag, 0, atol=1e-10):
        H = H.real
    
    # Solve the full eigenvalue problem and then take the first n_eigvals
    eigvals, eigvecs = np.linalg.eigh(H)
    eigvals = eigvals[:n_eigvals]
    eigvecs = eigvecs[:, :n_eigvals]
    
    # Sort eigenvalues and eigenvectors (should already be sorted, but just to be sure)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return eigvals, eigvecs

def plot_results(grid, eigvals, eigvecs, potential_values, alpha, title="Morse-Coulomb Potential"):
    """Plot the eigenfunctions and eigenvalues"""
    plt.figure(figsize=(12, 8))
    
    # Plot potential
    plt.plot(grid, potential_values, 'k--', label='Potential')
    
    # Plot eigenfunctions and eigenvalues
    for i in range(len(eigvals)):
        # Scale eigenfunction for visibility
        scale_factor = 0.1
        scaled_eigenfunction = scale_factor * eigvecs[:, i] + eigvals[i]
        
        plt.plot(grid, scaled_eigenfunction, label=f'E{i} = {eigvals[i]:.6f}')
        plt.axhline(y=eigvals[i], color=f'C{i}', linestyle=':')
    
    plt.title(f"{title} (α = {alpha})")
    plt.xlabel('Position')
    plt.ylabel('Energy / Wavefunction')
    plt.legend()
    plt.ylim(-2, 0)
    plt.xlim(-1, 30)
    plt.grid(True)
    
    # Add a zoom inset for the bound states
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(plt.gca(), width="40%", height="30%", loc=1)
    axins.plot(grid, potential_values, 'k--')
    for i in range(min(5, len(eigvals))):
        if eigvals[i] < 0:  # Only show bound states
            scale_factor = 0.05
            scaled_eigenfunction = scale_factor * eigvecs[:, i] + eigvals[i]
            axins.plot(grid, scaled_eigenfunction)
            axins.axhline(y=eigvals[i], color=f'C{i}', linestyle=':')
    
    # Set limits for inset plot - focusing on bound states
    bound_indices = np.where(eigvals < 0)[0]
    if len(bound_indices) > 0:
        negative_pot_indices = np.where(potential_values < 0)[0]
        r_min_bound = grid[negative_pot_indices[0]]
        r_max_bound = grid[negative_pot_indices[-1]]
        axins.set_xlim(r_min_bound, r_max_bound)
        axins.set_ylim(1.5*min(potential_values), 0)
        axins.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_morse_coulomb_potential(alpha, n_points=5000, n_eigvals=15, mass=1.0):
    """
    Complete analysis of the Morse-Coulomb potential
    """
    print(f"Analyzing Morse-Coulomb potential with α = {alpha}")
    
    # Create energy range for grid determination
    # Consider both negative energies (bound states) and positive energies
    energy_range = np.array([
        -0.001,   # Bound states
        ])
    
    # Create grid based on classical return points
    grid = create_adaptive_grid(alpha, energy_range, n_points)
    
    # Evaluate potential on grid
    potential_values = evaluate_potential_on_grid(grid, MsC_potential, alpha)
    
    # Calculate eigenvalues and eigenvectors
    eigvals, eigvecs = fourier_grid_hamiltonian_1d(grid, potential_values, mass, n_eigvals)
    
    # Print results
    print("\nEigenvalues (Energy levels):")
    for i, energy in enumerate(eigvals):
        print(f"E{i} = {energy:.8f}")
    
    # Calculate number of bound states (E < 0)
    n_bound = np.sum(eigvals < 0)
    print(f"\nNumber of bound states: {n_bound}")
    
    # Plot results
    plot_results(grid, eigvals, eigvecs, potential_values, alpha)
    
    return grid, eigvals, eigvecs, potential_values

# Run the analysis
if __name__ == "__main__":
    # Parameter for the Morse-Coulomb potential
    alpha = 0.0001
    
    # Run analysis
    grid, eigvals, eigvecs, potential_values = analyze_morse_coulomb_potential(alpha)
