import numpy as np

from .base import Potential


class SoftCoulomb(Potential):
    """
    1D soft-Coulomb potential, a regularized Coulomb interaction. The
    parameter 'alpha' softens the singularity at the origin, so that the
    potential well has finite depth 1/alpha.
    """

    def __init__(self, alpha: float):
        self.alpha      = alpha
        self.well_depth = 1 / alpha
        self.name       = "soft-Coulomb"
        self.acronym    = "sC"
        super().__init__(name=self.name, acronym=self.acronym)

    def value(self, position_array):
        r = np.asarray(position_array)
        return -1 / np.sqrt(r**2 + self.alpha**2)

    def first_derivative(self, position_array):
        r = np.asarray(position_array)
        return r * (r**2 + self.alpha**2) ** (-3 / 2)

    def second_derivative(self, position_array):
        r = np.asarray(position_array)
        return (r**2 + self.alpha**2) ** (-3 / 2) - 3 * r**2 * (r**2 + self.alpha**2) ** (-5 / 2)

    def turning_points(self, energy):

        shape = np.shape(energy)
        energy = np.asarray(energy)

        rM = np.sqrt(1 / energy**2 - self.alpha**2)
        rm = -rM

        # Bound states only exist within the well: -1/alpha < E < 0
        invalid = (energy >= 0) | (energy < -self.well_depth)
        rm = np.where(invalid, np.nan, rm)
        rM = np.where(invalid, np.nan, rM)

        return np.stack((rm, rM), axis=-1).reshape(shape + (2,))