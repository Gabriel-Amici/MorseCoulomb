"""
Stationary description of a quantum system: Hamiltonian, eigenproblem and
`Spectrum`.

Conventions
-----------
* Atomic units (hbar = 1).
* A `Spectrum` stores the eigenstates of H0 as columns of `states`, expressed
  in the *constrained* basis (the basis after boundary conditions were
  imposed). `constraint` maps them back to the raw basis of the representation.
"""

from dataclasses import dataclass, field, replace

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

    The eigenvectors returned by `scipy.linalg.eigh` are S-orthonormal, i.e.
    `states.T @ S @ states = 1`. `Spectrum.matrix` relies on this.

    Parameters
    ----------
    H : `np.ndarray`
        Hamiltonian matrix;

    S : `np.ndarray`
        Overlap matrix of the basis. Identity if orthogonal.

    Returns
    -------
    energies : `np.ndarray`
        Eigenvalues of the system, in ascending order;

    states : `np.ndarray`
        Eigenvectors of the system (one per column).
    """
    # NOTE: the original `S is np.eye(...)` compared object identity and was
    # therefore never true. Compare values instead.
    if np.allclose(S, np.eye(S.shape[0])):
        # Standard eigenvalue problem
        energies, states = eigh(H)
    else:
        energies, states = eigh(H, S)

    return energies, states


@dataclass
class Spectrum:
    """
    Eigenvalues and eigenstates of a static Hamiltonian H0, expressed in a
    given `Representation`.

    A Spectrum is the *stationary* object of the quantum side of the library.
    It also defines the basis (the eigenbasis of H0) in which time-dependent
    wavepackets are expanded, see `emerald.quantum.evolution`.

    Parameters
    ----------
    representation : `Representation`
        Basis in which the eigenproblem was solved (e.g. `BSplineBasis`);

    energies : `np.ndarray`, shape (n_states,)
        Eigenenergies in ascending order;

    states : `np.ndarray`, shape (n_constrained, n_states)
        Eigenvectors, one per column, in the *constrained* basis, rescaled
        to physical wavefunctions (see `Representation.normalization`);

    constraint : `np.ndarray`, shape (n_functions, n_constrained)
        Matrix C such that `C @ states[:, n]` are the coefficients of state
        `n` in the raw basis of the representation (see
        `Representation.constrained_basis`);

        `states` is stored in the constrained basis *after* normalization, so
        the reconstruction order matters: `wavefunction` applies C to the
        normalized coefficients, never the other way round.

    threshold : `float`
        Energy separating bound from continuum states. States with
        `energy < threshold` are considered bound. Defaults to 0.

    Notes
    -----
    Typical workflow::

        spec = Spectrum.from_representation(basis, potential, 'dirichlet', 'dirichlet')
        spec.n_bound                         # inspect how many bound states there are
        spec = spec.truncate(spec.n_bound + 100)   # keep bound + some pseudo-continuum
    """
    representation: Representation
    energies: np.ndarray
    states: np.ndarray
    constraint: np.ndarray
    threshold: float = 0.0
    # Bookkeeping only, so that `__post_init__` normalizes exactly once and
    # `truncate` (which round-trips through `dataclasses.replace`) does not
    # rescale an already-normalized spectrum.
    _normalized: bool = field(default=False, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Basic inspection
    # ------------------------------------------------------------------

    def __post_init__(self):
        """
        Rescale `states` from the internal normalization of the eigenproblem
        to physical wavefunctions, if the representation asks for it.
        """
        if self._normalized:
            return

        normalization = getattr(self.representation, "normalization", None)
        if normalization is not None:
            self.states = normalization * self.states

        self._normalized = True

    @property
    def n_states(self) -> int:
        """Number of eigenstates currently kept in the spectrum."""
        return len(self.energies)

    @property
    def bound(self) -> np.ndarray:
        """Boolean mask, True for bound states (`energy < threshold`)."""
        return self.energies < self.threshold

    @property
    def n_bound(self) -> int:
        """Number of bound states among the ones kept."""
        return int(np.count_nonzero(self.bound))

    def truncate(self, n_states: int) -> "Spectrum":
        """
        Returns a new `Spectrum` keeping only the `n_states` lowest states.

        The box states above the threshold are only a discretization of the
        continuum, so in practice one keeps the bound states plus as many
        pseudo-continuum states as the problem requires (converge the result
        with respect to `n_states`).

        Every operator built afterwards with `matrix` is the *projection* of
        the operator onto the kept subspace, and time evolution takes place
        entirely inside that subspace.
        """
        if not 1 <= n_states <= self.n_states:
            raise ValueError(
                f"n_states must be between 1 and {self.n_states}, got {n_states}."
            )
        return replace(
            self,
            energies=self.energies[:n_states].copy(),
            states=np.ascontiguousarray(self.states[:, :n_states]),
        )

    # ------------------------------------------------------------------
    # Operators and states in the eigenbasis
    # ------------------------------------------------------------------

    def matrix(self, func: callable) -> np.ndarray:
        """
        Matrix of the multiplicative operator `f(x)` in the eigenbasis of H0.

        Entry (n, m) is the matrix element

            M[n, m] = <phi_n| f(x) |phi_m> = integral phi_n(x) f(x) phi_m(x) dx

        where phi_n is the n-th eigenstate kept in this spectrum.

        How it is computed: the representation gives the matrix of f(x) in
        its own raw basis, B_ij = <B_i| f |B_j>. An eigenstate has raw-basis
        coefficients `a_n = C @ states[:, n]`, so

            M = measure * A.T @ B @ A,     A = constraint @ states.

        `measure` is the quadrature weight of one raw basis function: `dr` for
        a collocation grid whose `braket` returns the bare collocation
        matrix, `1` for a basis whose `braket` already integrates (e.g.
        B-splines).

        The result is real and symmetric (states are real), shape
        (n_states, n_states).

        Important
        ---------
        After `truncate`, `M` is the operator *projected* on the kept
        subspace (P f P). For that reason ``matrix(lambda x: x**2)`` is NOT
        equal to ``matrix(lambda x: x) @ matrix(lambda x: x)``: the latter
        forgets every transition through the discarded states. Always build
        a function of x directly with `matrix`, never by multiplying matrices.

        Parameters
        ----------
        func : callable
            Vectorized function of position (must accept a numpy array).
        """
        B = self.representation.braket(func)
        A = self.constraint @ self.states
        return self.representation.measure * (A.T @ B @ A)

    def projector(self, states) -> np.ndarray:
        """
        Diagonal of the projector P = sum_{n in states} |phi_n><phi_n|.

        Since H0 is diagonal in its own eigenbasis, P is a 0/1 diagonal, so
        it is returned as a vector of length `n_states` (no n x n matrix
        needed). It can be passed to `Evolution.expectation`, which then
        returns the total population of those states.

        Parameters
        ----------
        states : boolean mask, integer indices or slice
            The states to project on, e.g. `spec.bound`, `~spec.bound`,
            `[0, 1]` or `slice(0, 3)`.
        """
        p = np.zeros(self.n_states)
        p[states] = 1.0
        return p

    def state(self, n: int) -> np.ndarray:
        """
        Coefficient vector of the n-th eigenstate in the eigenbasis, i.e. the
        unit vector e_n. This is the usual initial condition for a
        propagation.
        """
        c = np.zeros(self.n_states, dtype=np.complex128)
        c[n] = 1.0
        return c

    # ------------------------------------------------------------------
    # Back to position space
    # ------------------------------------------------------------------

    def reconstruct(self, coeffs: np.ndarray) -> callable:
        """
        Callable wavefunction for an arbitrary vector `coeffs` of expansion
        coefficients in the eigenbasis, psi(x) = sum_n coeffs[n] phi_n(x).
        """
        # eigenbasis -> constrained basis -> raw basis of the representation
        raw = self.constraint @ (self.states @ coeffs)
        return self.representation.reconstruct(raw)

    def wavefunction(self, n: int) -> callable:
        """Callable for the n-th eigenfunction of the quantum system."""
        # The constraint must be applied to the *normalized* coefficients:
        # `states` is a (n_constrained,) vector, while `reconstruct` expects
        # (n_functions,) coefficients in the raw basis of the representation.
        # Reversing the order, or normalizing after applying C, silently
        # rescales the wavefunction whenever boundary conditions removed
        # functions from the basis.
        return self.representation.reconstruct(self.constraint @ self.states[:, n])

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_representation(cls, representation: Representation,
                            potential: callable, left_bc=None, right_bc=None,
                            threshold: float = 0.0):
        """
        Builds and diagonalizes H0 in the given representation.

        Parameters
        ----------
        representation : `Representation`
            Basis in which to solve the problem;
        potential : callable
            Vectorized potential energy V(x);
        left_bc | right_bc : `str` or `(alpha, beta)`
            Boundary conditions, see `hamiltonian`;
        threshold : `float`
            Energy separating bound from continuum states.
        """

        H, S = hamiltonian(representation, potential, left_bc, right_bc)
        C = representation.constrained_basis(left_bc, right_bc)

        energies, states = solve_eigenproblem(H, S)

        return cls(representation=representation,
                   energies=energies,
                   states=states,
                   constraint=C,
                   threshold=threshold)
