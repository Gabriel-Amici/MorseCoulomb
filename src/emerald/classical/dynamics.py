"""
Base module for implementing Dynamics objects and Trajectory objects.

 - Dynamics: the general perturbed dynamics
 - StaticDynamics: the particular unperturbed case
 - Trajectory: a class to store the results of a dynamics propagation
"""

import numpy as np
from scipy.interpolate import interp1d

from ..fields.base import Field
from ..potentials.base import Potential
from ..serialization import Serializable, SerializationError, deserialize_object
from .integrators import (
    _DEFAULT_INTEGRATOR,
    _DEFAULT_REGULARIZED_INTEGRATOR,
    Integrator,
)


class Dynamics(Serializable):

    _serialized_parameters = {"potential": "potential", "field": "field"}
    _format_version = 1

    def __init__(self, potential: Potential,
                 field: Field = None):

        self.potential = potential
        self.field     = field

    def dstate(self, time: float = 0.0, state: np.ndarray = None):
        r, p = state[..., 0], state[..., 1]
        
        f1 = p
        f2 = -self.potential.first_derivative(r) - self.field(time)
        
        return np.stack([f1, f2], axis=-1)

    @classmethod
    def from_dict(cls, data: dict):
        version = data.get("_format_version")
        if version != cls._format_version:
            raise SerializationError(
                f"Unsupported dynamics format version: {version}."
            )

        type_name = data.get("_type")
        if type_name is None:
            raise SerializationError("Dynamics data is missing '_type'.")

        from ..serialization import resolve

        concrete = resolve(type_name)
        if not issubclass(concrete, Dynamics):
            raise SerializationError(
                f"Serialisation type '{type_name}' is not a Dynamics subclass."
            )

        parameters = {}
        for parameter in concrete._serialized_parameters:
            if parameter not in data:
                raise SerializationError(
                    f"Missing parameter '{parameter}' for {type_name}."
                )
            parameters[parameter] = deserialize_object(data[parameter])

        if concrete is StaticDynamics:
            return concrete(parameters["potential"])
        return concrete(**parameters)

    def propagate(self, states0: np.ndarray, t_span, t_eval: np.ndarray, 
                  integrator: Integrator = _DEFAULT_INTEGRATOR) -> "Trajectory":

        return Trajectory(*integrator.propagate(self, states0, t_span, t_eval),
                          self.potential)
class StaticDynamics(Dynamics):

    _serialized_parameters = {"potential": "potential"}
    _format_version = 1

    def __init__(self, potential):
        super().__init__(potential, field=None)

    def dstate(self, time: float = 0.0, state: np.ndarray = None):
        r, p = state[..., 0], state[..., 1]

        f1 = p
        f2 = -self.potential.first_derivative(r)
        
        return np.stack([f1, f2], axis=-1)


class RegularizedDynamics(Dynamics):
    """
    Coulomb dynamics in KS-regularized extended phase space:
    Q = (q1, p1, q2, p2), q1 = sqrt(r), q2 = physical time.
    Integrated w.r.t. a fictitious parameter tau, not physical time.
    """

    _serialized_parameters = {"potential": "potential", "field": "field"}
    _format_version = 1
    time_index = 2    # which Q-component holds physical time

    def __init__(self, potential: Potential, field: Field = None):
        super().__init__(potential, field)

    def dstate(self, tau, Q):
        q1, p1, q2, p2 = Q[..., 0], Q[..., 1], Q[..., 2], Q[..., 3]
        F = self.field.value(q2)
        dFdt = self.field.derivative(q2)

        f1 = p1
        f2 = -8*q1*p2 - 16*q1**3 * F
        f3 = 4*q1**2
        f4 = -4*q1**4 * dFdt
        return np.stack([f1, f2, f3, f4], axis=-1)

    def to_regularized(self, state, t0=0.0):
        r0, p0 = state[..., 0], state[..., 1]
        E0 = p0**2/2 + self.potential.value(r0)
        q1 = np.sqrt(r0)
        return np.stack([q1, 2*q1*p0, np.full_like(r0, t0), -E0], axis=-1)

    def from_regularized(self, Q):
        q1, p1 = Q[..., 0], Q[..., 1]
        return np.stack([q1**2, p1/(2*q1)], axis=-1)

    def propagate(self, states0: np.ndarray, t_span, t_eval: np.ndarray, 
                  integrator: Integrator = _DEFAULT_REGULARIZED_INTEGRATOR) -> "Trajectory":

        return Trajectory(*integrator.propagate(self, states0, t_span, t_eval),
                          self.potential)

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

    def ionization_probabilities(self, criterion='energy', r_max: float = 100.):
        """
        Ionization probability as a time series
        """
        if criterion=='energy':
            if self.states.ndim != 3:
                ionized = self.energy > 0
            else:
                ionized = np.sum(self.energy > 0, axis=1)
            return ionized/self.N_trajectories

        if criterion=='distance':
            ionized = np.where((self.energy > 0) & \
                               (np.abs(self.position) > r_max), 1, 0)
            if self.states.ndim == 3:
                ionized = np.sum(ionized, axis=1)
            return ionized/self.N_trajectories

        # later: distance criterium

    def ionization_probability(self, criterion='energy'):
        """
        Ionization probability as at the end of the time series
        """

        return self.ionization_probabilities(criterion)[-1]

class PoincareMap(Trajectory):

    def __init__(self, time, states, potential):
        super().__init__(time, states, potential)

class ClassicalSystem:

    def __init__(self, dynamics: Dynamics, states0: np.ndarray):

        self.dynamics  = dynamics
        self.states0   = states0
        self.potential = dynamics.potential
        self.field     = dynamics.field

    def propagate(self, t_eval: np.ndarray, t_span: tuple | None = None, integrator: Integrator = _DEFAULT_INTEGRATOR) -> "Trajectory":

        if t_span is None:
            t_span = (t_eval[0], t_eval[-1])

        return Trajectory(*integrator.propagate(self.dynamics, self.states0, t_span, t_eval),
                          self.potential)

    def poincare_map(self, NT: int = 100, integrator: Integrator = _DEFAULT_INTEGRATOR) -> "PoincareMap":
        """
        Compute the Poicaré Map of the system.

        Parameters
        ----------
        integrator: Integrator
            Integrator to be used.
        NT: int
            How many field periods the particles will be propagated for.
        """
        if self.field is None:
            raise AssertionError("Field must be defined")
        t_eval = np.arange(NT)*self.field.period
        t_span = (t_eval[0], t_eval[-1])

        traj = self.propagate(t_eval, t_span, integrator)

        return PoincareMap(np.arange(NT), traj.states, self.potential)

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

            pos = pos[~np.isnan(mom)]
            mom = mom[~np.isnan(mom)]

            states_plus  = np.column_stack((pos,  mom))
            states_minus = np.column_stack((pos, -mom))
            states0 = np.vstack((states_minus, states_plus))

        elif abs(momenta_signs) == 1:
            pos = np.linspace(r0, r1, N)
            mom = dynamics.potential.momentum(E, pos)

            pos = pos[~np.isnan(mom)]
            mom = mom[~np.isnan(mom)]


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

            pos = pos[~np.isnan(mom)]
            mom = mom[~np.isnan(mom)]

            states_plus  = np.column_stack((pos,  mom))
            states_minus = np.column_stack((pos, -mom))
            states0 = np.vstack((states_minus, states_plus))

        elif abs(momenta_signs) == 1:
            angs = np.linspace(t0, t1, N)
            
            pos = interpolator(angs)
            mom = dynamics.potential.momentum(E, pos)

            pos = pos[~np.isnan(mom)]
            mom = mom[~np.isnan(mom)]


            states0 = np.column_stack((pos, momenta_signs * mom))

        else:
            raise ValueError(
                f"'momenta_signs' is {momenta_signs} but must be ±1 or 0"
            )

        return cls(dynamics, states0)