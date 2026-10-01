from typing import Protocol
from warnings import warn

import numpy as np
from scipy.interpolate import BSpline, interp1d
from scipy.linalg import toeplitz

from emerald.numerics.quadrature import gauss_legendre_quadrature


class Representation(Protocol):
    number_functions: int
    normalization: float | None
    measure: float
    overlap: np.ndarray
    kinetic: np.ndarray

    def braket(self, func: callable) -> np.ndarray: ...
    def reconstruct(self, coeffs: np.ndarray) -> callable: ...
    def constrained_basis(self, left=None, right=None) -> np.ndarray: ...


class FourierGridBasis(Representation):

    def __init__(self, position_grid: np.ndarray):
        position_grid = np.asarray(position_grid)
        assert len(position_grid) % 2 == 1

        # `kinetic` and the derivative stencil of `constrained_basis` both
        # assume a *uniform* grid, so a non-uniform one has to be rejected
        # instead of silently producing a wrong operator.
        steps = np.diff(position_grid)
        if not np.allclose(steps, steps[0]):
            raise ValueError(
                "FourierGridBasis requires a uniformly spaced grid; got "
                f"spacing varying between {steps.min():.3e} and "
                f"{steps.max():.3e}. Resample the grid first."
            )

        self.position_grid    = position_grid
        self.dr               = steps[0]
        self.number_functions = len(position_grid)
        # The eigenvectors of the discretized problem are Euclidean-normalized
        # (sum c_i^2 = 1). Physical wavefunctions on the grid satisfy
        # sum c_i^2 * dr = 1, hence the 1 / sqrt(dr) prefactor.
        self.normalization    = 1 / np.sqrt(self.dr)
        # The raw basis functions are the delta-like samples
        # delta(x - x_i) / sqrt(dr), so every integral picks up a factor dr.
        # `braket` returns the bare collocation matrix and `Spectrum.matrix`
        # applies this weight.
        self.measure           = self.dr

    @property
    def overlap(self):
        return np.eye(self.number_functions)

    @property
    def kinetic(self):

        N = self.number_functions
        L = self.position_grid[-1] - self.position_grid[0]
        
        n = (N - 1) // 2

        # Compute the kinetic energy first row
        l = np.arange(n)
        g = 2 * (np.pi * l / L) ** 2  # g[0] = 0 naturally
        theta = 2 * np.pi / (N - 1)
        k = np.arange(N)
        C = np.cos(l[:, None] * k * theta)  # Shape: (n, N)
        t = (2 / (N - 1)) * np.dot(g, C)  # First row of T

        # Construct Hamiltonian
        K = toeplitz(t)  # Symmetric Toeplitz matrix

        return K

    def braket(self, func):
        return np.diag( func(self.position_grid) )

    def reconstruct(self, coeffs):
        return interp1d(self.position_grid, coeffs)

    def constrained_basis(self, left=None, right=None):
        """
        Only the periodic (wrap-around) case is supported, so this returns the
        identity and accepts no boundary conditions.

        Parameters
        ----------
        left | right : None
            Must be None.

        Raises
        ------
        NotImplementedError
            If either boundary condition is not None.

        Notes
        -----
        The Fourier grid is a *momentum* basis: the grid points are
        collocation points for a Fourier series,

            psi(x_j) = sum_l c_l exp(2 pi i l j / (N-1)),

        so no basis function is "located at" a boundary and there is nothing
        to clip out the way there is for the local B-spline functions.

        Imposing a boundary condition by constraining or deleting the
        boundary degrees of freedom does not work here. That trick is
        equivalent to the Dirichlet realization only for a *local* stencil
        (finite differences); the Fourier operator is nonlocal, so
        compressing it does not impose the condition in function space. The
        eigenvalues stay ~2x off the box spectrum and do not converge as the
        grid is refined.

        The correct way to impose a boundary condition in an FGH scheme is to
        choose the transform, which builds the condition into the basis:

        * periodic  -> DFT, circulant kinetic matrix;
        * dirichlet -> DST-I (sine basis, vanishes identically at both ends);
        * neumann   -> DCT (cosine basis).

        `kinetic` currently implements neither cleanly and carries spurious
        zero modes, so this class is not usable for a converged calculation.
        """
        if left is not None or right is not None:
            raise NotImplementedError(
                "FourierGridBasis does not support boundary conditions: the "
                "momentum basis has no boundary-localized functions to "
                "constrain. Use left=right=None (periodic), or switch to a "
                "representation that supports them, e.g. BSplineBasis."
            )

        return np.eye(self.number_functions)


class BSplineBasis(Representation):

    '''
    Basic class for generating a set of Bsplines given a particular breakpoint
    sequence. 
    '''

    def __init__(self, breakpoint_sequence: np.ndarray, order: int):
        '''
        Initialize the B-spline basis with a given breakpoint sequence and order.'''
        self.order                 = order
        self.degree                = order - 1
        self.number_points         = len(breakpoint_sequence)
        self.number_intervals      = self.number_points - 1
        self.breakpoint_sequence   = breakpoint_sequence
        self.multiplicity_sequence = self.multiplicity_from_breakpoints()
        self.continuity_sequence   = self.order - self.multiplicity_sequence
        self.knot_sequence         = self.knots_from_breakpoints()
        self.number_functions      = self.number_intervals + self.order - 1
        self.basis_functions       = self.generate_basis_functions()
        self.overlap               = self.overlap_matrix()
        self.kinetic               = self.kinetic_matrix()

        # `eigh(H, S)` already returns S-orthonormal states, i.e. physically
        # normalized wavefunctions: integral phi_n^2 dx = 1. Nothing to do.
        self.normalization         = None
        # The B-spline basis functions are genuine functions and `braket`
        # integrates them with Gauss-Legendre quadrature, so the integral
        # weight is already baked in.
        self.measure               = 1.0


    def multiplicity_from_breakpoints(self):
        """
        Calculates the knot multiplicity vector from a given breakpoints 
        sequence as: [k, 1, 1, ..., 1, k].
        """

        nu = np.array([self.order] + [1] * (self.number_points - 2) + [self.order])
        return nu

    def knots_from_breakpoints(self):
        """
        Generates the full knot sequence given the multiplicity of each
        breakpoint.
        """

        t = np.concatenate([[self.breakpoint_sequence[j]] * self.multiplicity_sequence[j] \
                            for j in range(self.number_points) ])
        return t

    def generate_basis_functions(self):
        """
        Generates the full basis functions using SciPy's `BSpline` object.
        """

        basis_functions = []
        for i in range(self.number_functions):
            coeff = np.eye(self.number_functions)[i]
            basis_function = BSpline(self.knot_sequence, coeff, self.degree,
                                        extrapolate=False)
            basis_functions.append(basis_function)
        return basis_functions

    def overlap_matrix(self, extra_integration_points: int = 0):
        '''
        Compute the overlap matrix for the B-spline basis functions.
        '''
        # Initialize the overlap matrix, N by N, where N is the number of 
        # basis functions
        overlap = np.zeros((self.number_functions, self.number_functions))

        # k points are enough to compute exactly a 2k - 1 degree polynomial
        integration_points = self.order + extra_integration_points

        # Iterate over each knot interval to apply the quadrature rule
        for m in range(self.number_intervals):
            # Integration limits for the current interval
            a = self.breakpoint_sequence[m]
            b = self.breakpoint_sequence[m+1]

            # Iterate over the k basis functions that are non-zero in the 
            # current interval
            for i in range(m, m+self.order):
                for j in range(i, m+self.order):
                    integrand = lambda x: \
                        self.basis_functions[i](x)*self.basis_functions[j](x)
                    # Use Gauss-Legendre quadrature to compute the integral 
                    # over [a, b] using k points to integrate the polynomial 
                    # of degree 2k-2 exactly
                    Sij = gauss_legendre_quadrature(integrand, a, b, integration_points)

                    overlap[i, j] += Sij

        # Symmetrize the overlap matrix
        S = overlap + overlap.T - np.diag(overlap.diagonal())
                    
        return S

    def kinetic_matrix(self, extra_integration_points: int = 0):
        '''
        Compute the kinetic matrix for the B-spline basis functions.
        -½∫ Bᵢ(x)B''ₖ(x) dx
        '''
        # k-1 points are enough to compute exactly a 2k - 3 degree polynomial
        integration_points = self.degree + extra_integration_points

        # Initialize the kinetic matrix, N by N, where N is the number of 
        # basis functions
        kinetic = np.zeros((self.number_functions, self.number_functions))

        # Iterate over each knot interval to apply the quadrature rule
        for m in range(self.number_intervals):

            # Integration limits for the current interval
            a = self.breakpoint_sequence[m]
            b = self.breakpoint_sequence[m+1]

            # Iterate over the basis functions that are non-zero in the 
            # current interval
            for i in range(m, m+self.order):
                for j in range(i, m+self.order):
                    integrand = lambda x: \
                        self.basis_functions[i](x)*self.basis_functions[j](x, nu=2)
                    # Use Gauss-Legendre quadrature to compute the integral 
                    # over [a, b] using k-1 points to integrate the polynomial 
                    # of degree 2k-4 exactly
                    Kij = gauss_legendre_quadrature(integrand, a, b, integration_points)

                    kinetic[i, j] += Kij

        # Symmetrize the kinetic matrix
        K = kinetic + kinetic.T - np.diag(kinetic.diagonal())
        
        return -(1/2)*K
    
    def braket(self, func: callable, extra_integration_points: int = 20):
        '''
        Compute the bra-ket <Bᵢ|f|Bₖ> in the B-spline basis.
        '''
        if extra_integration_points == 0:
            warn(f"No extra integration points specified per interval."
                 f"integration with {self.order} points will not be accurate.")
        integration_points = self.order + extra_integration_points

        # Initialize the braket matrix, N by N, where N is the number of 
        # basis functions
        braket = np.zeros((self.number_functions, self.number_functions))

        # Iterate over each breakpoint interval to apply the quadrature rule
        for m in range(self.number_intervals):

            # Integration limits for the current 
            # breakpoint interval
            a = self.breakpoint_sequence[m]
            b = self.breakpoint_sequence[m+1]

            # Iterate over the basis functions that are non-zero in the 
            # current interval
            for i in range(m, m+self.order):
                for j in range(i, m+self.order):
                    integrand = lambda x: \
                        self.basis_functions[i](x)*func(x)*self.basis_functions[j](x)
                    # Use Gauss-Legendre quadrature to compute the integral 
                    # over [a, b] using extra points to integrate the polynomial 
                    # of degree 2k-2 times the function, which may not be 
                    # polynomial, to machine accuracy
                    Bij = gauss_legendre_quadrature(integrand, a, b, integration_points)

                    braket[i, j] += Bij

        # Symmetrize the braket matrix

        B = braket + braket.T - np.diag(braket.diagonal())
        
        return B

    # ------------------------------------------------------------
    # Boundary-condition machinery
    # ------------------------------------------------------------

    def _boundary_operator(self, side, bc):
        """
        Construct the row vector L such that

            L @ c = 0

        represents the requested homogeneous boundary condition.

        Parameters
        ----------
        side : {"left", "right"}
            Boundary at the beginning or end of the interval.

        bc : None, "dirichlet", "neumann", or tuple
            Boundary condition.

            "dirichlet":
                f = 0

            "neumann":
                f' = 0

            (alpha, beta):
                alpha*f + beta*f' = 0

        Returns
        -------
        L : ndarray, shape (N,)
            Boundary-condition row.
        """

        N = self.number_functions

        if bc is None:
            return None

        # Convert convenient strings to (alpha, beta)
        if bc == "dirichlet":
            alpha, beta = 1.0, 0.0

        elif bc == "neumann":
            alpha, beta = 0.0, 1.0

        elif isinstance(bc, (tuple, list, np.ndarray)):
            if len(bc) != 2:
                raise ValueError(
                    "Robin BC must be given as (alpha, beta)."
                )

            alpha, beta = bc

        else:
            raise ValueError(
                "BC must be None, 'dirichlet', 'neumann', "
                "or (alpha, beta)."
            )

        # Determine endpoint
        if side == "left":
            x = self.breakpoint_sequence[0]

        elif side == "right":
            x = self.breakpoint_sequence[-1]

        else:
            raise ValueError("side must be 'left' or 'right'.")

        L = np.zeros(N)

        # Evaluate alpha*B_i(x) + beta*B_i'(x)
        for i, B in enumerate(self.basis_functions):

            value = B(x)

            derivative = B.derivative(1)(x)

            L[i] = alpha * value + beta * derivative

        return L

    def constrained_basis(self, left=None, right=None):
        """
        Build C such that c = C @ q maps boundary-adapted coefficients q
        back to the original B-spline coefficients c, satisfying the
        requested homogeneous BC(s) at each end.
    
        Exploits the fact that, for a clamped knot vector, a first-order
        BC at x=a only ever couples B_0 and B_1 (and symmetrically
        B_{N-1}, B_{N-2} at x=b) — so each BC is eliminated with a local
        2x2 combination instead of a dense nullspace computation. This
        keeps C sparse/near-identity, so it preserves the banded
        structure of S and K under C.T @ S @ C.
        """
        N = self.number_functions
        C = np.eye(N)
        keep = list(range(N))
    
        if left is not None:
            L = self._boundary_operator("left", left)
            l0, l1 = L[0], L[1]
            if np.isclose(l1, 0.0):
                # value-only constraint (e.g. Dirichlet): c_0 forced to 0
                keep.remove(0)
            else:
                # c_1 = -(l0/l1) c_0  -> fold into a single edge function
                C[1, 0] = -l0 / l1
                keep.remove(1)
    
        if right is not None:
            L = self._boundary_operator("right", right)
            lNm1, lNm2 = L[-1], L[-2]
            if np.isclose(lNm2, 0.0):
                keep.remove(N - 1)
            else:
                C[N - 2, N - 1] = -lNm1 / lNm2
                keep.remove(N - 2)
    
        return C[:, keep]

    def reconstruct(self, coeffs: np.ndarray):
        """
        Reconstructs a callable function given the basis coefficients.
        """
        return BSpline(self.knot_sequence, coeffs, self.degree, extrapolate=False)