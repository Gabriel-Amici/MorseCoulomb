import numpy as np
from qmsolve import Hamiltonian, SingleParticle, Eigenstates

def MsC_potential_vec(alpha, r):
    """
    Vectorized Morse-Coulomb potential with overflow protection.
    
    Args:
        alpha (float): Potential parameter.
        r (np.ndarray): Position array.
    
    Returns:
        np.ndarray: Potential values at each position.
    """
    D = 1 / alpha
    beta = 1 / (alpha * np.sqrt(2))
    # Cap exponential terms to avoid overflow
    r_cutoff = -50 / beta  # Where exp(-beta * r) ≈ exp(50) ≈ 5e21, reasonable limit
    r_safe = np.where(r < r_cutoff, r_cutoff, r)
    morse = D * (np.exp(-2 * beta * r_safe) - 2 * np.exp(-beta * r_safe))
    coulomb = -1 / np.sqrt(r**2 + alpha**2)
    return np.where(r > 0, coulomb, morse)

def MsC_return_points(alpha: float, E: float) -> np.ndarray:
    """
    Calculates classical turning points.
    
    Args:
        alpha (float): Potential parameter.
        E (float): Reference energy.
    
    Returns:
        np.ndarray: [rm, rM], minimum and maximum turning points.
    """
    rm = -alpha * np.sqrt(2) * np.log(np.sqrt(alpha * E + 1) + 1)
    rM = np.sqrt(1 / (E**2) - alpha**2)
    return np.array([rm, rM])

def compute_spectrum(alpha, N, E_ref=-0.5, extend_factor=1.5, k=10):
    """
    Compute energy spectrum using qmsolve with a grid based on turning points.
    
    Args:
        alpha (float): Potential parameter.
        N (int): Number of grid points.
        E_ref (float): Reference energy for turning points.
        extend_factor (float): Grid extension factor.
        k (int): Number of eigenvalues to compute.
    
    Returns:
        Eigenstates: Object containing energies and wavefunctions.
    """
    # Compute turning points
    rm, rM = MsC_return_points(alpha, E_ref)
    
    # Define grid with reduced extension to avoid extreme values
    L_left = max(rm * extend_factor, -20.0)  # Cap at -20 if too negative
    L_right = rM * extend_factor
    extent = L_right - L_left
    
    # Define potential for qmsolve
    def potential(particle):
        return MsC_potential_vec(alpha, particle.x + L_left)  # Shift grid to [0, extent]
    
    # Create Hamiltonian (qmsolve shifts grid internally, we offset x)
    H = Hamiltonian(particles=SingleParticle(),
                    potential=potential,
                    spatial_ndim=1,
                    N=N,
                    extent=extent)
    
    # Solve for eigenstates
    print("Computing...")
    eigenstates = H.solve(max_states=k)
    return eigenstates, L_left, L_right

def main():
    # Parameters
    alpha = 0.1
    N = 15000
    E_ref = -0.001
    extend_factor = 2  # Reduced to limit Morse growth
    k = 10
    
    # Compute spectrum
    eigenstates, L_left, L_right = compute_spectrum(alpha, N, E_ref, extend_factor, k)
    
    # Print results
    print(f"Grid range: [{L_left:.3f}, {L_right:.3f}]")
    print(f"Energy spectrum for alpha = {alpha}:")
    for i, energy in enumerate(eigenstates.energies):
        print(f"E_{i} = {energy:.6f}")
    
    '''# Sample wavefunction values (first 5 points)
    print("\nSample wavefunction values (first 5 points):")
    for i in range(min(3, k)):  # First 3 states
        wf = eigenstates[i]
        print(f"State {i}: {wf[:5]}")'''

if __name__ == "__main__":
    main()