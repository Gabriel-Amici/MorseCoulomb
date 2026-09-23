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
    def __init__(self, time: np.ndarray, states: np.ndarray, 
                 potential: Potential):
        self.time, self.states, self.potential = time, states, potential

        self._angle  = None
        self._action = None
        self.N_trajectories = 1 if states.ndim != 3 else states.shape[1]
        self.T_times = states.shape[0]

    def for_particle(self, i) -> "Trajectory":
        if i < self.N_trajectories > 1:
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
        # E.shape = (T, N) or (T,)
        return p**2 / 2 + self.potential.value(r)   # vectorized, works for N=1 or N=many

    @property
    def action(self):
        if self._action is not None:
            return self._action
        self._action = self.potential._batch_action(self.energy)
        # J.shape = (T, N) or (T,)
        return self._action

    @property
    def angle(self):
        if self._angle is not None:
            return self._angle
        angs = self.potential._batch_angle(self.energy, self.position)
        self._angle  = np.sign(self.momentum)*angs
        # Θ.shape = (T, N) or (T,)
        return self._angle

    def ionization_probability(self, criteria='energy'):

        if criteria=='energy':
            ionized = np.sum(self.energy[-1,...] > 0)
            return ionized/self.N_trajectories