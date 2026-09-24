"""
Base module for implementing Dynamics objects and Trajectory objects.

 - Dynamics: the general perturbed dynamics
 - StaticDynamics: the particular unperturbed case
 - Trajectory: a class to store the results of a dynamics propagation
"""

import numpy as np
from scipy.interpolate import interp1d

from ..fields.base import Field
from ..integrators.base import Integrator
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

    def ionization_probability(self, criterion='energy'):

        if criterion=='energy':
            ionized = np.sum(self.energy[-1,...] > 0)
            return ionized/self.N_trajectories

        # later: distance criterium

class PoincareMap(Trajectory):

    def __init__(self, time, states, potential):
        super().__init__(time, states, potential)

class ClassicalSystem:

    def __init__(self, dynamics: Dynamics, states0: np.ndarray):

        self.dynamics  = dynamics
        self.states0   = states0
        self.potential = dynamics.potential
        self.field     = dynamics.field

    def propagate(self, integrator: Integrator, t_span, t_eval: np.ndarray) -> "Trajectory":

        return Trajectory(*integrator.propagate(self.dynamics, self.states0, t_span, t_eval),
                          self.potential)

    def poincare_map(self, integrator: Integrator, Nt: int = 100) -> "PoincareMap":
        """
        Compute the Poicaré Map of the system.

        Parameters
        ----------
        integrator: Integrator
            Integrator to be used.
        Nt: int
            How many field periods the particles will be propagated for.
        """
        if self.field is None:
            raise AssertionError("Field must be defined")
        t_eval = np.linspace(0, Nt*self.field.period, Nt)
        t_span = (t_eval[0], t_eval[-1])

        traj = self.propagate(integrator, t_span, t_eval)

        return PoincareMap(np.arange(Nt), traj.states, self.potential)

    # Factory methods
    @classmethod
    def from_energy(cls, dynamics: Dynamics, E: float, N: int,
                    momenta_signs: int = 0, position_signs: int = 0):

        rm, rM = dynamics.potential.turning_points(E)

        # Choose position interval
        position_intervals = {
            0:  (rm, rM),
            1:  (0,  rM),
            -1: (rm, 0),
        }

        try:
            r0, r1 = position_intervals[position_signs]
        except KeyError:
            raise ValueError(
                f"'position_signs' is {position_signs} but must be ±1 or 0"
            )

        if momenta_signs == 0:
            pos = np.linspace(r0, r1, N // 2)
            mom = dynamics.potential.momentum(E, pos)

            states_plus  = np.column_stack((pos,  mom))
            states_minus = np.column_stack((pos, -mom))
            states0 = np.vstack((states_minus, states_plus))

        elif abs(momenta_signs) == 1:
            pos = np.linspace(r0, r1, N)
            mom = dynamics.potential.momentum(E, pos)

            states0 = np.column_stack((pos, momenta_signs * mom))

        else:
            raise ValueError(
                f"'momenta_signs' is {momenta_signs} but must be ±1 or 0"
            )

        return cls(dynamics, states0)     

    @classmethod
    def from_angle(cls, dynamics: Dynamics, E: float, N: int,
                   momenta_signs: int = 0, position_signs: int = 0):

        rm, rM = dynamics.potential.turning_points(E)
        theta0 = dynamics.potential.angle(E, 0)

        pos_interval = np.linspace(rm, rM, 150)
        ang_interval = dynamics.potential.angle(E, pos_interval)
        interpolator = interp1d(ang_interval, 
                                pos_interval, 
                                kind='cubic', 
                                fill_value="extrapolate")
        
        # Choose position interval
        angle_intervals = {
            0:  (0, np.pi),
            1:  (theta0,  np.pi),
            -1: (0, theta0),
        }

        try:
            t0, t1 = angle_intervals[position_signs]
        except KeyError:
            raise ValueError(
                f"'position_signs' is {position_signs} but must be ±1 or 0"
            )


        if momenta_signs == 0:
            angs = np.linspace(t0, t1, N//2)

            pos = interpolator(angs)
            mom = dynamics.potential.momentum(E, pos)

            states_plus  = np.column_stack((pos,  mom))
            states_minus = np.column_stack((pos, -mom))
            states0 = np.vstack((states_minus, states_plus))

        elif abs(momenta_signs) == 1:
            angs = np.linspace(t0, t1, N)
            
            pos = interpolator(angs)
            mom = dynamics.potential.momentum(E, pos)

            states0 = np.column_stack((pos, momenta_signs * mom))

        else:
            raise ValueError(
                f"'momenta_signs' is {momenta_signs} but must be ±1 or 0"
            )

        return cls(dynamics, states0)