"""
Base module for potential implementation
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from ..numerics.quadrature import gauss_legendre_quadrature, simpson13


class Potential(ABC):
    """
    Base class for 1D potentials, responsible for calculating the potential 
    and its derivatives, as well as the classical turning points.
    """

    def __init__(self, *args, **kwargs):
        self.name    = kwargs.get('name') if 'name' in kwargs else None
        self.acronym = kwargs.get('acronym') if 'acronym' in kwargs else None
        self.symmetric = self._check_symmetry()

    def __call__(self, r):
        return self.value(r)

    @abstractmethod
    def value(self, position_array): ...

    @abstractmethod
    def first_derivative(self, position_array): ...

    @abstractmethod
    def second_derivative(self, position_array): ...

    def _check_symmetry(self):
        """
        Check if the potential is symmetric around the origin

        Returns
        -------
        bool
            True if the potential is symmetric, False otherwise
        """
        r = np.random.uniform(1e-5, 2, 100)  # Random small distance from the origin
        return np.allclose(self.value(r), self.value(-r))
    

    def turning_points(self, E=0.0, a=-20.0, b=20.0, npts=10000):
    
        f = lambda x: self.value(x) - E
    
        x = np.linspace(a, b, npts)
        y = f(x)
    
        roots = []
    
        for i in range(len(x)-1):
        
            # exact hit
            if abs(y[i]) < 1e-12:
                roots.append(x[i])
    
            # sign change
            elif y[i] * y[i+1] < 0:
                r = brentq(f, x[i], x[i+1])
                roots.append(r)
    
        return np.array(roots)


    def momentum(self, E: float, r): #np.ndarray | float):
        """
        Classical momentum of the particle in the potential at a given
        position and energy.

        Right at a classical turning point the radicand 2*(E - V(r)) may
        only be slightly negative because of floating-point roundoff.  Such
        values are clamped to zero so momentum returns 0 at the return
        points instead of NaN.  Radicands that are genuinely negative
        (classically forbidden region) are returned as NaN.
        """
        r = np.asarray(r)
        V = self.value(r)
        radicand = 2 * (E - V)

        # Roundoff noise in the radicand is of order eps times the energy
        # scale; the factor gives headroom for the whole evaluation chain
        # while staying far below any physical radicand.
        tol = 500 * np.finfo(float).eps * np.maximum(1.0, np.maximum(np.abs(E), np.abs(V)))

        with np.errstate(invalid='ignore'):
            radicand = np.where((radicand < 0) & (radicand > -tol), 0.0, radicand)
            radicand = np.where(radicand < 0, np.nan, radicand)

            return np.sqrt(radicand)


    def action(self, E: float, N: int = 500, dr: float | None = 1.e-6, method: str = "gauss"):
        """
        Classical action of the particle in the potential at a given energy

        Parameters
        ----------
        E : float
            Energy of the particle
        a : float, optional
            Lower limit for finding turning points, by default -10.0
        b : float, optional
            Upper limit for finding turning points, by default 10.0
        N : int, optional
            Number of quadrature points for numerical integration, 
            by default 500
        dr : float, optional
            Step size for numerical integration, by default 1.e-6
        method : str, optional
            Method for numerical integration, either "gauss" for Gauss-Legendre
            quadrature or "simpson" for Simpson's 1/3 rule, by default "gauss"
        """

        rm, rM = self.turning_points(E)
        if np.isnan(rm) or np.isnan(rM):
            raise ValueError(f"turning points not found for energy E={E}.")
        integrand = lambda r: self.momentum(E, r)
        if self.symmetric:
            if method == "gauss":
                action = gauss_legendre_quadrature(integrand, rm, rM, N) / np.pi
            else:
                action = simpson13(integrand, rm, rM, N, dr) / np.pi
            return action
        
        # action = gauss_legendre_quadrature(lambda r: self.momentum(E, r), r1, r2, N) / np.pi
        if method == "gauss":
            action = gauss_legendre_quadrature(integrand, rm, rM, N) / np.pi
        else:
            action = simpson13(integrand, rm, rM, N, dr) / np.pi
        return action

    def _inv_momentum_integrand(self, phi, E, rm, rM):
        L = rM - rm
        r = rm + L * np.sin(phi)**2
        dr_dphi = 2 * L * np.sin(phi) * np.cos(phi)

        return dr_dphi / self.momentum(E, r)


    def angular_frequency(self, E: float, N: int = 500):
        """
        Classical angular frequency of the particle in the potential at a given energy

        Parameters
        ----------
        E : float
            Energy of the particle
        N : int, optional
            Number of quadrature points for numerical integration, 
            by default 500
        """
        rm, rM = self.turning_points(E)
        if np.isnan(rm) or np.isnan(rM):
            raise ValueError(f"turning points not found for energy E={E}.")

        I = gauss_legendre_quadrature(
            lambda phi: self._inv_momentum_integrand(phi, E, rm, rM), 
            0, np.pi/2, N)

        return np.pi / I

    def period(self, E: float, N: int = 500):
        """
        Classical period of the particle in the potential at a given energy

        Parameters
        ----------
        E : float
            Energy of the particle
        N : int, optional
            Number of quadrature points for numerical integration, 
            by default 500
        """
        return 2 * np.pi / self.angular_frequency(E, N)

    def frequency(self, E: float, N: int = 500):
        """
        Classical frequency of the particle in the potential at a given energy

        Parameters
        ----------
        E : float
            Energy of the particle
        N : int, optional
            Number of quadrature points for numerical integration, 
            by default 500
        """
        return 1 / self.period(E, N)

    def angle(self, E: float, r, N: int = 500):
        """
        Calculates the canonical angle, the dynamical variable whose
        conjugate is the action.

        Parameters
        ----------
        E : float
            Energy of the particle
        r : float or np.ndarray
            Position(s) of the particle
        N : int, optional
            Number of quadrature points for numerical integration, 
            by default 500
        """

        r = np.asarray(r)

        rm, rM = self.turning_points(E)
        omg = self.angular_frequency(E, N)

        if r.size == 1:

            phi_r = np.arcsin(np.sqrt((r - rm) / (rM - rm)))

            if phi_r == 0:
                return 0.0

            return omg * gauss_legendre_quadrature(
                lambda phi: self._inv_momentum_integrand(phi, E, rm, rM), 
                0, phi_r, N)
        
        angles = np.zeros_like(r)
        for i in range(r.size):
            phi_r = np.arcsin(np.sqrt((r[i] - rm) / (rM - rm)))
            if phi_r == 0:
                angles[i] = 0.0
            else:
                angles[i] = omg * gauss_legendre_quadrature(
                    lambda phi: self._inv_momentum_integrand(phi, E, rm, rM), 
                    0, phi_r, N)
        
        return angles

    @staticmethod
    def _grid(lo: float, hi: float, N: int | None = None, dr: float | None = None) -> np.ndarray:
        """Build a spatial grid from either a point count or a resolution — never both."""
        if (N is None) == (dr is None):
            raise ValueError("Specify exactly one of N or dr.")
        if N is not None:
            return np.linspace(lo, hi, N)
        n = max(round((hi - lo) / dr) + 1, 2)
        return np.linspace(lo, hi, n)   # linspace, not arange, to hit `hi` exactly
        
    def phase_space(self, E: float, N: int | None = None, dr: float | None = None,
                     r1: float | None = None, r2: float | None = None, output: str = "rp"):
        """
        Phase space portrait of a particle of energy E under this potential.
    
        r1, r2 are only used as fallback plotting limits on a side where the
        potential is unbound at this energy (return_points gives None there) —
        they're ignored on any side that has a genuine return point.
        """
        rm, rM = self.turning_points(E)
    
        if not np.isnan(rm) and not np.isnan(rM):
            # bound: both edges are real turning points -> full closed orbit
            r = self._grid(rm, rM, N, dr)
            p = self.momentum(E, r)
            p[0], p[-1] = 0.0, 0.0                      # clamp: exact zero at true turning points
            r_out = np.concatenate([r, r[::-1]])
            p_out = np.concatenate([p, -p[::-1]])
    
        elif not np.isnan(rm) :                             # unbound above: only rm is real
            if r2 is None:
                raise ValueError("Unbound above at this energy — supply r2.")
            r = self._grid(rm, r2, N, dr)
            p = self.momentum(E, r)
            p[0] = 0.0
            # incoming branch first: puts the seam (rm, real) in the middle,
            # leaves the artificial edge r2 at the two loose ends
            r_out = np.concatenate([r[::-1], r])
            p_out = np.concatenate([-p[::-1], p])
    
        elif not np.isnan(rM):                             # unbound below: only rM is real
            if r1 is None:
                raise ValueError("Unbound below at this energy — supply r1.")
            r = self._grid(r1, rM, N, dr)
            p = self.momentum(E, r)
            p[-1] = 0.0
            r_out = np.concatenate([r, r[::-1]])
            p_out = np.concatenate([p, -p[::-1]])
    
        else:                                             # fully unbound: no fold at all
            if r1 is None or r2 is None:
                raise ValueError("No turning points at this energy — supply r1 and r2.")
            r_out = self._grid(r1, r2, N, dr)
            p_out = self.momentum(E, r_out)

        if output in "rp":
            return {"rp": (r_out, p_out), "r": r_out, "p": p_out}[output]


    def _batch_action(self, E, N: int = 500, M: int = 300):
        E = np.atleast_1d(np.asarray(E, dtype=float))
        bound = E < 0                      # only bound motion has a well-defined action
        result = np.full(E.shape, np.nan)
        if not np.any(bound):
            return result

        Eb = E[bound]
        E_grid = np.linspace(Eb.min(), Eb.max(), M)     # M evaluations, not T
        table = np.array([self.action(Ek, N=N) for Ek in E_grid])

        result[bound] = np.interp(Eb, E_grid, table)
        return result


    def _batch_angle(self, E, r, M: int = 300, N: int = 500):
        E, r = np.broadcast_arrays(
            np.asarray(E, dtype=float),
            np.asarray(r, dtype=float),
        )
    
        shape = E.shape
        Eall = E.ravel()
        rall = r.ravel()
    
        result = np.full(shape, np.nan)
    
        bound = Eall < 0
        if not np.any(bound):
            return result
    
        Eq = Eall[bound]
        rq = rall[bound]
    
        E_grid = np.linspace(Eq.min(), Eq.max(), M)
    
        rm = np.empty(M)
        rM = np.empty(M)
        omg = np.empty(M)
    
        for k, Ek in enumerate(E_grid):
            rm[k], rM[k] = self.turning_points(Ek)
            omg[k] = self.angular_frequency(Ek)
    
        phi_table = np.linspace(0.0, np.pi / 2, N)
    
        with np.errstate(divide="ignore", invalid="ignore"):
            integrand = self._inv_momentum_integrand(
                phi_table[None, :],
                E_grid[:, None],
                rm[:, None],
                rM[:, None],
            )
    
        integrand[:, 0] = integrand[:, 1]
    
        cum = cumulative_trapezoid(
            integrand,
            phi_table,
            axis=1,
            initial=0,
        )
    
        interp = RegularGridInterpolator(
            (E_grid, phi_table),
            cum,
            bounds_error=False,
            fill_value=np.nan,
        )
    
        rm_q = np.interp(Eq, E_grid, rm)
        rM_q = np.interp(Eq, E_grid, rM)
        omg_q = np.interp(Eq, E_grid, omg)
    
        phi_q = np.arcsin(
            np.sqrt(
                np.clip(
                    (rq - rm_q) / (rM_q - rm_q),
                    0,
                    1,
                )
            )
        )
    
        values = omg_q * interp(
            np.column_stack([Eq, phi_q])
        )
    
        result.flat[bound] = values
    
        return result