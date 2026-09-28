import numpy as np

from .base import Potential


class Coulomb(Potential):
    """
    1D Coulomb potential. The elementary potential describing the interaction
    between two point charges of opposite signs.
    """

    def __init__(self):
        self.name       = "Coulomb"
        self.acronym    = "C"
        self.well_depth = np.infty
        super().__init__(name=self.name, acronym=self.acronym)

    def value(self, position_array):
        r = np.asarray(position_array)
        return np.where( r > 0, -1 / r, np.infty)

    def first_derivative(self, position_array):
        r = np.asarray(position_array)
        return np.where(r > 0, 1/r**2, np.infty)

    def second_derivative(self, position_array):
        r = np.asarray(position_array)
        return np.where(r > 0, -2/r**3, np.infty)

    def turning_points(self, energy):

        shape = np.shape(energy)
        energy = np.asarray(energy)

        rM = - 1 / energy
        rm = 0

        # Bound states only exist within the well: -1/alpha < E < 0
        invalid = (energy >= 0) | (energy < -self.well_depth)
        rm = np.where(invalid, np.nan, rm)
        rM = np.where(invalid, np.nan, rM)

        return np.stack((rm, rM), axis=-1).reshape(shape + (2,))

    def action(self, E: float | np.ndarray):
        """
        Classical action of the Coulomb potential
        """

        return 1/np.sqrt( -2*E )

    def angular_frequency(self, E: float | np.ndarray):

        return 1/self.action(E)**3

    def frequency(self, E: float | np.ndarray):

        return self.angular_frequency(E)/(2*np.pi)
    
    def period(self, E: float | np.ndarray):

        return 1/self.frequency(E)

    def angle(self, E: float | np.ndarray, r: float | np.ndarray):

        theta =  2*( np.arcsin( np.sqrt( np.abs(E)*r ) ) \
                     - np.sqrt( np.abs(E)*r )*np.sqrt( 1 - np.abs(E)*r ) )

        return theta

    def extended_turning_points(self, E: float | np.ndarray):

        shape = np.shape(E)
        E = np.asarray(E)

        K = 4
        p2 = -E
    
        q1M = np.sqrt( K/(4*p2) )
        q1m = -q1M

        # Bound states only exist within the well: -1/alpha < E < 0
        invalid = (E >= 0) | (E < -self.well_depth)
        
        q1m = np.where(invalid, np.nan, q1m)
        q1M = np.where(invalid, np.nan, q1M)

        return np.stack((q1m, q1M), axis=-1).reshape(shape + (2,))

    def extended_momentum(self, E: float, q1: float | np.ndarray):

        q1 = np.asarray(q1)
        K = 4
        p2 = -E

        radicand = 2*(K-4*p2*q1**2)

        # Roundoff noise in the radicand is of order eps times the energy
        # scale; the factor gives headroom for the whole evaluation chain
        # while staying far below any physical radicand.
        tol = 500 * np.finfo(float).eps * np.maximum(1.0, np.abs(E))

        with np.errstate(invalid='ignore'):
            radicand = np.where((radicand < 0) & (radicand > -tol), 0.0, radicand)
            radicand = np.where(radicand < 0, np.nan, radicand)

            return np.sqrt(radicand)

    def extended_phase_space(self, E: float, N: int | None = None, dr: float | None = None,
                     q1_1: float | None = None, q1_2: float | None = None):

        q1m, q1M = self.extended_turning_points(E)
    
        if not np.isnan(q1m) and not np.isnan(q1M):
            # bound: both edges are real turning points -> full closed orbit
            q1 = self._grid(q1m, q1M, N, dr)
            p1 = self.extended_momentum(E, q1)
            # q1[0], p1[-1] = 0.0, 0.0                      # clamp: exact zero at true turning points
            q1_out = np.concatenate([q1, q1[::-1]])
            p1_out = np.concatenate([p1, -p1[::-1]])
    
        elif not np.isnan(q1m) :                             # unbound above: only rm is real
            if q1_2 is None:
                raise ValueError("Unbound above at this energy — supply q1_2.")
            q1 = self._grid(q1m, q1_2, N, dr)
            p1 = self.extended_momentum(E, q1)
            # p[0] = 0.0
            # incoming branch first: puts the seam (rm, real) in the middle,
            # leaves the artificial edge r2 at the two loose ends
            q1_out = np.concatenate([q1[::-1], q1])
            p1_out = np.concatenate([-p1[::-1], p1])
    
        elif not np.isnan(q1M):                             # unbound below: only rM is real
            if q1_1 is None:
                raise ValueError("Unbound below at this energy — supply q1_1.")
            q1 = self._grid(q1_1, q1M, N, dr)
            p1 = self.extended_momentum(E, q1)
            # p[-1] = 0.0
            q1_out = np.concatenate([q1, q1[::-1]])
            p1_out = np.concatenate([p1, -p1[::-1]])
    
        else:                                             # fully unbound: no fold at all
            if q1_1 is None or q1_2 is None:
                raise ValueError("No turning points at this energy — supply q1_1 and q1_2.")
            q1_out = self._grid(q1_1, q1_2, N, dr)
            p1_out = self.extended_momentum(E, q1_out)

        return (q1_out, p1_out)

    def phase_space(self, E: float, N: int | None = None, dr: float | None = None,
                     r1: float | None = None, r2: float | None = None):

        q1_1 = np.sqrt(r1) if r1 is not None else None
        q1_2 = np.sqrt(r2) if r2 is not None else None
        dq = dr**2
        N = int(N**2)
        
        extended_ps = self.extended_phase_space(E, N, dq, q1_1, q1_2)

        q1, p1 = extended_ps


        r_out = q1**2
        p_out = np.where(q1 != 0, p1/(2*q1), np.infty)

        return (r_out, p_out)