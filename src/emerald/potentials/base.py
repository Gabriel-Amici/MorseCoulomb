"""
Base module for potential implementation
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import brentq

from ..numerics.quadrature import gauss_legendre_quadrature, simpson13


class Potential(ABC):
    """
    Base class for 1D potentials, responsible for calculating the potential 
    and its derivatives, as well as the classical return points.
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
    
    # def return_points(self, E: float = 0.0, a: float = -10.0, b: float = 10.0):
    #     """
    #     Classical return points of the potential
        
    #     General implementation finds the return points numerically by solving V(r) = E for a given energy E.
    #     """

    #     fn = lambda r: self.value(r) - E

    #     # Find the return points by solving V(r) = E
    #     r1 = brentq(fn, a, 0)  # Left return
    #     r2 = brentq(fn, 0, b)   # Right return

    #     return r1, r2

    def return_points(self, E=0.0, a=-20.0, b=20.0, npts=10000):
    
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
        """
        r = np.asarray(r)
        radicand = 2 * (E - self.value(r))
        mask = (radicand < 0) & np.isclose(radicand, 0, atol=1e-8)
        radicand = np.where(mask, 0, radicand)

        return np.sqrt(radicand)

    def action(self, E: float, N: int = 2000, dr: float = 1.e-6):
        """
        Classical action of the particle in the potential at a given energy

        Parameters
        ----------
        E : float
            Energy of the particle
        a : float, optional
            Lower limit for finding return points, by default -10.0
        b : float, optional
            Upper limit for finding return points, by default 10.0
        N : int, optional
            Number of quadrature points for numerical integration, by default 2000
        """

        r1, r2 = self.return_points(E)
        if None in (r1, r2):
            raise ValueError(f"Return points not found for energy E={E}.")
        if self.symmetric:
            # action = 2 * gauss_legendre_quadrature(lambda r: self.momentum(E, r), 0, r2, N) / np.pi
            action = 2 * simpson13(lambda r: self.momentum(E, r), r1, r2, dr) / np.pi
            return action
        
        # action = gauss_legendre_quadrature(lambda r: self.momentum(E, r), r1, r2, N) / np.pi
        action = simpson13(lambda r: self.momentum(E, r), r1, r2, dr) / np.pi
        return action

    def angular_frequency(self, E: float, dE: float = 1e-5, N: int = 2000):
        """
        Classical angular frequency of the particle in the potential at a given energy

        Parameters
        ----------
        E : float
            Energy of the particle
        N : int, optional
            Number of quadrature points for numerical integration, by default 2000
        """
        return ( (1/(12*dE))*( self.action(E-2*dE, N)  
                                  - 8*self.action(E-dE, N ) 
                                  + 8*self.action(E+dE, N ) 
                                  - self.action(E+2*dE, N ) )
                                  )**(-1)

    def angle(self, E: float, r, dr: float = 1.e-5):
        """
        Calculates the canonycal angle, the dynamical variable whose
        conjugate is the action.
        """

        r = np.asarray(r)

        rm, _ = self.return_points(E)
        freq = self.angular_frequency(E)

        # integrand = lambda x: ( 2*(E - self.value(x)) )**(-1/2)

        if r.size == 1:
            # return freq*simpson13(integrand, rm, r, dr)
            return freq*gauss_legendre_quadrature(lambda x: 1/self.momentum(E, x), rm, r, 3000)

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
        rm, rM = self.return_points(E)
    
        if rm is not None and rM is not None:
            # bound: both edges are real turning points -> full closed orbit
            r = self._grid(rm, rM, N, dr)
            p = self.momentum(E, r)
            p[0], p[-1] = 0.0, 0.0                      # clamp: exact zero at true turning points
            r_out = np.concatenate([r, r[::-1]])
            p_out = np.concatenate([p, -p[::-1]])
    
        elif rm is not None:                             # unbound above: only rm is real
            if r2 is None:
                raise ValueError("Unbound above at this energy — supply r2.")
            r = self._grid(rm, r2, N, dr)
            p = self.momentum(E, r)
            p[0] = 0.0
            # incoming branch first: puts the seam (rm, real) in the middle,
            # leaves the artificial edge r2 at the two loose ends
            r_out = np.concatenate([r[::-1], r])
            p_out = np.concatenate([-p[::-1], p])
    
        elif rM is not None:                             # unbound below: only rM is real
            if r1 is None:
                raise ValueError("Unbound below at this energy — supply r1.")
            r = self._grid(r1, rM, N, dr)
            p = self.momentum(E, r)
            p[-1] = 0.0
            r_out = np.concatenate([r, r[::-1]])
            p_out = np.concatenate([p, -p[::-1]])
    
        else:                                             # fully unbound: no fold at all
            if r1 is None or r2 is None:
                raise ValueError("No return points at this energy — supply r1 and r2.")
            r_out = self._grid(r1, r2, N, dr)
            p_out = self.momentum(E, r_out)

        if output in "rp":
            return {"rp": (r_out, p_out), "r": r_out, "p": p_out}[output]