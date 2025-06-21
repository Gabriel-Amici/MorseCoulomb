import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

def fourier_grid_hamiltonian_1d(x_min, x_max, n_points, potential_func, mass=1.0, n_eigvals=5):
    """
    Implements the 1D Fourier Grid Hamiltonian method to solve the time-independent Schrödinger equation.
    
    Parameters:
    -----------
    x_min : float
        Minimum value of position grid
    x_max : float
        Maximum value of position grid
    n_points : int
        Number of grid points
    potential_func : function
        A function that takes position x and returns potential energy V(x)
    mass : float, optional
        Particle mass (default: 1.0, atomic units)
    n_eigvals : int, optional
        Number of eigenvalues/eigenfunctions to compute (default: 5)
        
    Returns:
    --------
    x : ndarray
        Position grid
    eigvals : ndarray
        Eigenvalues (energies)
    eigvecs : ndarray
        Eigenvectors (wavefunctions)
    """
    # Set up position grid
    x = np.linspace(x_min, x_max, n_points)
    dx = x[1] - x[0]
    
    # Calculate potential energy matrix (diagonal in position basis)
    V = np.diag(potential_func(x))
    
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
    
    # Solve the eigenvalue problem (use eigsh for symmetric matrices if H is real)
    if np.allclose(H.imag, 0):
        H = H.real
        eigvals, eigvecs = eigsh(H, k=min(n_eigvals, n_points-1), which='SM')
    else:
        # Use np.linalg.eigh for complex Hermitian matrices
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = eigvals[:n_eigvals]
        eigvecs = eigvecs[:, :n_eigvals]
    
    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    return x, eigvals, eigvecs

# Example: Harmonic oscillator potential
def harmonic_oscillator(x, k=1.0):
    """Harmonic oscillator potential: V(x) = 0.5 * k * x^2"""
    return 0.5 * k * x**2

def plot_results(x, eigvals, eigvecs, potential_func, title="Quantum Harmonic Oscillator"):
    """Plot the eigenfunctions and eigenvalues"""
    plt.figure(figsize=(12, 8))
    
    # Plot potential
    V = potential_func(x)
    plt.plot(x, V, 'k--', label='Potential')
    
    # Plot eigenfunctions and eigenvalues
    for i in range(len(eigvals)):
        # Scale eigenfunction for visibility
        scale_factor = 1.5
        scaled_eigenfunction = scale_factor * eigvecs[:, i] + eigvals[i]
        
        plt.plot(x, scaled_eigenfunction, label=f'E{i} = {eigvals[i]:.4f}')
        plt.axhline(y=eigvals[i], color=f'C{i}', linestyle=':')
    
    plt.title(title)
    plt.xlabel('Position')
    plt.ylabel('Energy / Wavefunction')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters
    x_min = -10.0
    x_max = 10.0
    n_points = 1000
    mass = 1.0
    n_eigvals = 5
    
    # Solve for harmonic oscillator
    x, eigvals, eigvecs = fourier_grid_hamiltonian_1d(
        x_min, x_max, n_points, 
        harmonic_oscillator, 
        mass=mass, 
        n_eigvals=n_eigvals
    )
    
    # Plot results
    plot_results(x, eigvals, eigvecs, harmonic_oscillator)
    
    # Compare with analytical results
    print("Numerical eigenvalues:", eigvals)
    print("Analytical eigenvalues (harmonic oscillator):", 
          np.array([n + 0.5 for n in range(n_eigvals)]))
    
    # Example: Anharmonic oscillator
    def anharmonic_oscillator(x, k=1.0, alpha=0.1):
        """Anharmonic oscillator: V(x) = 0.5 * k * x^2 + alpha * x^4"""
        return 0.5 * k * x**2 + alpha * x**4
    
    # Solve for anharmonic oscillator
    x, eigvals_anharm, eigvecs_anharm = fourier_grid_hamiltonian_1d(
        x_min, x_max, n_points, 
        lambda x: anharmonic_oscillator(x), 
        mass=mass, 
        n_eigvals=n_eigvals
    )
    
    # Plot anharmonic results
    plot_results(x, eigvals_anharm, eigvecs_anharm, 
                lambda x: anharmonic_oscillator(x), 
                title="Anharmonic Oscillator")