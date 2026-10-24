"""
Base module for implementing Dynamics objects and Trajectory objects.

 - Dynamics: the general perturbed dynamics
 - StaticDynamics: the particular unperturbed case
 - Trajectory: a class to store the results of a dynamics propagation
"""

import time

import numpy as np
from scipy.interpolate import interp1d

from ..fields.base import Field
from ..potentials.base import Potential
from ..serialization import (
    Serializable,
    SerializationError,
    array_to_dict,
    deserialize_object,
    resolve,
    to_jsonable,
)
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

    def propagate(self, states0: np.ndarray, t_eval: np.ndarray, t_span: tuple | None = None, 
                  integrator: Integrator = _DEFAULT_INTEGRATOR) -> "Trajectory":

        if t_span is None:
            t_span = (t_eval[0], t_eval[-1])

        t_start = time.perf_counter()
        time_vals, states = integrator.propagate(self, states0, t_span, t_eval)
        generation_time_s = time.perf_counter() - t_start
        metadata = {
            "generation_time_s": generation_time_s,
            "t_span": list(t_span) if hasattr(t_span, "__iter__") and not isinstance(t_span, str) else t_span,
        }
        if t_eval is not None:
            try:
                metadata["t_eval_len"] = int(len(t_eval))
            except Exception:
                pass
        return Trajectory(time_vals, states, dynamics=self, metadata=metadata)

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
        t_start = time.perf_counter()
        time_vals, states = integrator.propagate(self, states0, t_span, t_eval)
        generation_time_s = time.perf_counter() - t_start
        metadata = {
            "generation_time_s": generation_time_s,
            "t_span": list(t_span) if hasattr(t_span, "__iter__") and not isinstance(t_span, str) else t_span,
        }
        if t_eval is not None:
            try:
                metadata["t_eval_len"] = int(len(t_eval))
            except Exception:
                pass
        return Trajectory(time_vals, states, dynamics=self, metadata=metadata)

class Trajectory(Serializable):

    _format_version = 1
    _serialized_parameters = {}

    def __init__(self, time: np.ndarray, states: np.ndarray,
                 dynamics: Dynamics = None, metadata: dict | None = None):
        self.time = np.asarray(time)
        self.states = np.asarray(states)
        if dynamics is None:
            raise ValueError("Trajectory requires a dynamics object.")
        self.dynamics = dynamics
        self.potential = dynamics.potential
        self.field = getattr(dynamics, "field", None)
        self.metadata = dict(metadata) if metadata else {}

        self._angle = None
        self._action = None
        self.N_trajectories = 1 if self.states.ndim != 3 else self.states.shape[1]
        self.T_times = self.states.shape[0]

    def to_dict(self) -> dict:
        data = {
            "_type": type(self).__name__,
            "_format_version": self._format_version,
            "time": array_to_dict(self.time),
            "states": array_to_dict(self.states),
            "dynamics": self.dynamics.to_dict() if self.dynamics is not None else None,
            "metadata": to_jsonable(self.metadata),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict):
        version = data.get("_format_version")
        if version != cls._format_version:
            raise SerializationError(
                f"Unsupported trajectory format version: {version}."
            )

        type_name = data.get("_type")
        if type_name is None:
            raise SerializationError("Trajectory data is missing '_type'.")

        concrete = resolve(type_name)
        if not issubclass(concrete, Trajectory):
            raise SerializationError(
                f"Serialisation type '{type_name}' is not a Trajectory subclass."
            )

        time_vals = deserialize_object(data["time"])
        states = deserialize_object(data["states"])
        dynamics_data = data.get("dynamics")
        if dynamics_data is None:
            raise SerializationError("Trajectory data is missing 'dynamics'.")
        dynamics = deserialize_object(dynamics_data)
        metadata = data.get("metadata") or {}
        if isinstance(metadata, dict):
            pass
        else:
            metadata = {}

        return concrete(time_vals, states, dynamics=dynamics, metadata=metadata)

    def for_particle(self, i) -> "Trajectory":
        if i < self.N_trajectories > 1:
            return Trajectory(self.time, self.states[:, i], dynamics=self.dynamics)

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

    def __init__(self, time, states, dynamics: Dynamics = None, metadata: dict | None = None):
        super().__init__(time, states, dynamics=dynamics, metadata=metadata)

class ClassicalSystem(Serializable):
    """
    Classical system composed by a particle or an ensamble of particles, all
    subject to the same `Dynamics`.
    """

    _serialized_parameters = {"dynamics": "dynamics", "states0": "states0"}
    _format_version = 1

    def __init__(self, dynamics: Dynamics, states0: np.ndarray):

        self.dynamics  = dynamics
        self.states0   = np.asarray(states0)
        self.potential = dynamics.potential
        self.field     = getattr(dynamics, "field", None)

    def to_dict(self) -> dict:
        data = {
            "_type": type(self).__name__,
            "_format_version": self._format_version,
            "dynamics": self.dynamics.to_dict(),
            "states0": array_to_dict(self.states0),
        }
        return data

    @classmethod
    def from_dict(cls, data: dict):
        version = data.get("_format_version")
        if version != cls._format_version:
            raise SerializationError(
                f"Unsupported ClassicalSystem format version: {version}."
            )

        type_name = data.get("_type")
        if type_name is None:
            raise SerializationError("ClassicalSystem data is missing '_type'.")

        concrete = resolve(type_name)
        if not issubclass(concrete, ClassicalSystem):
            raise SerializationError(
                f"Serialisation type '{type_name}' is not a ClassicalSystem subclass."
            )

        dynamics = deserialize_object(data["dynamics"])
        states0 = deserialize_object(data["states0"])
        return concrete(dynamics, states0)

    def propagate(self, t_eval: np.ndarray, t_span: tuple | None = None, 
                  integrator: Integrator = _DEFAULT_INTEGRATOR) -> "Trajectory":

        if t_span is None:
            t_span = (t_eval[0], t_eval[-1])
        t_start = time.perf_counter()
        time_vals, states = integrator.propagate(self.dynamics, self.states0, t_span, t_eval)
        generation_time_s = time.perf_counter() - t_start
        metadata = {
            "generation_time_s": generation_time_s,
            "t_span": list(t_span) if hasattr(t_span, "__iter__") and not isinstance(t_span, str) else t_span,
        }
        try:
            metadata["t_eval_len"] = int(len(t_eval))
        except Exception:
            pass
        return Trajectory(time_vals, states, dynamics=self.dynamics, metadata=metadata)

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
        t_start = time.perf_counter()
        t_eval = np.arange(NT) * self.field.period
        t_span = (t_eval[0], t_eval[-1])

        traj = self.propagate(t_eval, t_span, integrator)
        generation_time_s = time.perf_counter() - t_start
        traj.metadata["type"] = "poincare"
        traj.metadata["NT"] = int(NT)
        try:
            traj.metadata["field_period"] = float(self.field.period)
        except Exception:
            pass
        traj.metadata["generation_time_s_total"] = generation_time_s
        return PoincareMap(traj.time, traj.states, dynamics=traj.dynamics, metadata=traj.metadata)

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

    @classmethod
    def merge_systems(cls, dynamics: Dynamics, syst_list: list['ClassicalSystem']):

        states0 = []

        for syst in syst_list:
            if not isinstance(syst, ClassicalSystem):
                raise TypeError(f"System mus be a ClassicalSystem, not {type(syst)}")

            states0.append(syst.states0)
        
        states = np.row_stack(tuple(states0))

        return cls(dynamics, states)