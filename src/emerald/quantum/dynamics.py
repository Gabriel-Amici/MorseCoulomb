from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from .representations import Representation


def hamiltonian(representation: Representation, potential: callable, left_bc, right_bc):
    """
    Builds the hamiltonian matrix in the given `representation` of a quantum
    system composed of an electron interacting with the given `potential`.

    Parameters
    ----------
    representation : `Representation`
        The given basis on which to express the hamiltonian;

    potential : `Potential`
        Callable representing the interacting potential energy;

    left_bc | right_bc : `str` or `(alpha, beta)`
        Right and left boundary conditions: `'dirichlet'`, `'neumann'`
        or general Robin condition `alpha*f(c) + beta*f'(c)`.
    
    Returns
    -------
    H : `np.ndarray`
        Hamiltonian matrix in the given representation;
    S : `np.ndarray`
        Overlap matrix of the basis. Identity if orthogonal.
    """

    # Hamiltonian operator in this representation
    H = representation.kinetic + representation.braket(potential)

    # Overlap term
    S = representation.overlap

    # Basis constraints
    C = representation.constrained_basis(left_bc, right_bc)

    # Constrain the matrices to boundary conditions
    H = C.T @ H @ C
    S = C.T @ S @ C

    return H, S

def solve_eigenproblem(H, S):
    """
    Solves the generalized eigenvalue problem H @ c = E S @ c for the given
    Hamiltonian and overlap matrices.

    Parameters
    ----------
    H : `np.ndarray`
        Hamiltonian matrix;

    S : `np.ndarray`
        Overlap matrix of the basis. Identity if orthogonal.

    Returns
    -------
    energies : `np.ndarray`
        Eigenvalues of the system;

    states : `np.ndarray`
        Eigenvectors of the system.
    """
    if S is np.eye(S.shape[0]):
        # Standard eigenvalue problem
        energies, states = eigh(H)
    else:
        energies, states = eigh(H, S)

    return energies, states

@dataclass
class Spectrum:
    """
    Represents the spectrum of a quantum system, including its eigenvalues
    and eigenstates.
    """
    representation: Representation
    energies: np.ndarray
    states:   np.ndarray
    base_size: int

    def wavefunction(self, n: int):
        """
        Returns a callable for the n-th wavefunction of the quantum system
        """

        return self.representation.reconstruct(self.states[:self.base_size, n])

    @classmethod
    def from_representation(cls, representation: Representation, 
                            potential: callable, left_bc, right_bc):

        H, S = hamiltonian(representation, potential, left_bc, right_bc)

        energies, states = solve_eigenproblem(H, S)

        return cls(representation=representation,
                   energies=energies,
                   states=states,
                   base_size=len(energies))