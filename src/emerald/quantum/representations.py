from typing import Protocol
from warnings import warn

import numpy as np
from emerald.numerics.integration import gauss_legendre_quadrature
from scipy.interpolate import BSpline, interp1d
from scipy.linalg import toeplitz


class Representation(Protocol):
    number_functions: int
    overlap: np.ndarray
    kinetic: np.ndarray

    def braket(self, func: callable) -> np.ndarray: ...
    def reconstruct(self, coeffs: np.ndarray) -> callable: ...
    def constrained_basis(self, left, right) -> np.ndarray: ...


class FourierGridBasis(Representation):

    def __init__(self, position_grid: np.ndarray):
        assert len(position_grid) // 2 == 1
        self.position_grid    = np.asarray(position_grid)
        self.number_functions = len(position_grid)

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
        return interp1d(self.position_grid, coeffs*self.position_grid)

    def constrained_basis(self, left, right):
        # This method doesn't support different 
        # boundary conditions
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
        print("Hi")
        return BSpline(self.knot_sequence, coeffs, self.degree, extrapolate=False)