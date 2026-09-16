"""
Base module for implementing integrators.

 - Integrator: the general interface for integrating dynamics
 - FixedStepIntegrator: a base class for fixed-step integrators
 - RK4Integrator: the 4th-order Runge-Kutta integrator
 - ScipyIntegrator: an integrator that uses SciPy's solve_ivp function
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp

from ..classical.dynamics import Trajectory


class Integrator(ABC):

    @abstractmethod
    def propagate(self, state0, dynamics, t_span, t_eval): ...


class FixedStepIntegrator(Integrator):
    """Shared substep/output-grid logic. Subclasses only supply the one-step formula."""

    def __init__(self, dt: float):
        self.dt = dt

    def _step(self, dynamics, t, state, h):
        """
        Naivest integrator: forward Euler's method
        """

        return state + dynamics.dstate(t, state) * h


    def propagate(self, dynamics, state0, t_span, t_eval=None):
        t0, tf = t_span
        if t_eval is None:
            t_eval = np.arange(t0, tf, self.dt)      # library-wide default: dt's own grid

        state0 = np.asarray(state0, dtype=float)
        states = np.empty((len(t_eval), *state0.shape))
        t, state = t0, state0

        print(state)

        for j, t_target in enumerate(t_eval):
            while t < t_target:
                h = min(self.dt, t_target - t)         # only the last substep shrinks
                state = self._step(dynamics, t, state, h)
                t += h
            states[j] = state

        return Trajectory(np.asarray(t_eval), states, dynamics.potential)


class RK4Integrator(FixedStepIntegrator):

    def __init__(self, dt = 1.e-4):
        super().__init__(dt)

    def _step(self, dynamics, t, state, h):
        """
        Order 4 Runge-Kutta method for solving differencial equation systems
        """

        k1 = dynamics.dstate(t,         state)
        k2 = dynamics.dstate(t + 0.5*h, state + 0.5*h*k1)
        k3 = dynamics.dstate(t + 0.5*h, state + 0.5*h*k2)
        k4 = dynamics.dstate(t + h,     state + h*k3)
        return state + (h/6)*(k1 + 2*k2 + 2*k3 + k4)



class ScipyIntegrator(Integrator):
    def __init__(self, method: str = "RK45", **options):
        self.method, self.options = method, options

    def propagate(self, dynamics, state0, t_span, t_eval=None):
        state0 = np.asarray(state0, dtype=float)
        shape = state0.shape                       # (2,) or (N, 2)

        def fun(t, y):
            return np.ravel(dynamics.dstate(t, y.reshape(shape)))

        sol = solve_ivp(fun, t_span, np.ravel(state0),
                         method=self.method, t_eval=t_eval, **self.options)
        states = sol.y.T.reshape(len(sol.t), *shape)
        return Trajectory(sol.t, states, dynamics.potential)