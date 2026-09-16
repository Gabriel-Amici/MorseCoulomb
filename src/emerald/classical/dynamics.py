"""
Base module for implementing Dynamics objects and Trajectory objects.

 - Dynamics: the general perturbed dynamics
 - StaticDynamics: the particular unperturbed case
 - Trajectory: a class to store the results of a dynamics propagation
"""

import numpy as np

from ..fields.base import Field
from ..potentials.base import Potential


class Dynamics:

    def __init__(self, potential: Potential,
                 field: Field = None):

        self.potential = potential
        self.field     = field

    def dstate(self, time: float = 0.0, state: np.ndarray = None):
        r, p = state[..., 0], state[..., 1]
        
        f1 = p
        f2 = -self.potential.first_derivative(r) - self.field(time)
        
        return np.stack([f1, f2], axis=-1)

class StaticDynamics(Dynamics):

    def __init__(self, potential):
        super().__init__(potential)

    def dstate(self, time: float = 0.0, state: np.ndarray = None):
        r, p = state[..., 0], state[..., 1]

        f1 = p
        f2 = -self.potential.first_derivative(r)
        
        return np.stack([f1, f2], axis=-1)

class Trajectory:
    def __init__(self, time, states, potential):
        self.time, self.states, self.potential = time, states, potential

    def for_particle(self, i) -> "Trajectory":
        return Trajectory(self.time, self.states[:, i], self.potential)

    @property
    def position(self):
        return self.states[..., 0]
    
    @property
    def momentum(self):
        return self.states[..., 1]
    
    @property
    def energy(self):
        r, p = self.position, self.momentum
        return p**2 / 2 + self.potential.value(r)   # vectorized, works for N=1 or N=many

    @property
    def action(self):
        E = self.energy
        act = np.vectorize(lambda E: self.potential.action(E))
        return np.where(E < 0, act(E), np.nan)