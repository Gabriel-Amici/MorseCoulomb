"""
Integrators implementation

 - FixedStepIntegrator: a base class for fixed-step integrators
 - RK4Integrator: the 4th-order Runge-Kutta integrator
 - ScipyIntegrator: an integrator that uses SciPy's solve_ivp function
 - RegularizedScipyIntegrator: an integrator for regularized dynamics using SciPy's solve_ivp function
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


class Integrator(ABC):

    @abstractmethod
    def propagate(self, dynamics, states0, t_span, t_eval): ...

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

        for j, t_target in enumerate(t_eval):
            while t < t_target:
                h = min(self.dt, t_target - t)         # only the last substep shrinks
                state = self._step(dynamics, t, state, h)
                t += h
            states[j] = state

        return np.asarray(t_eval), states


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
        return sol.t, states

_DEFAULT_INTEGRATOR = ScipyIntegrator(method="RK45", rtol=1e-9, atol=1e-12)


class RegularizedScipyIntegrator(Integrator):
    def __init__(self, tau_max: float = 1e4, **options):
        self.tau_max, self.options = tau_max, options   # generous cap; the event cuts it short

    def propagate(self, dynamics, state0, t_span, t_eval):
        t0, tf = t_span
        state0 = np.atleast_2d(state0)
        N = state0.shape[0]
        out = np.empty((len(t_eval), N, 2))

        for i in range(N):                      # see note below on why per-particle
            Q0 = dynamics.to_regularized(state0[i], t0=t0)

            def reached_tf(tau, Q, tf=tf):
                return Q[dynamics.time_index] - tf
            reached_tf.terminal, reached_tf.direction = True, 1

            sol = solve_ivp(dynamics.dstate, (0, self.tau_max), Q0,
                             dense_output=True, events=reached_tf, **self.options)
            print(f"particle {i}: nfev={sol.nfev}  steps={len(sol.t)}  status={sol.status}")
            tau_grid = np.linspace(sol.t[0], sol.t[-1], 2000)
            t_grid   = sol.sol(tau_grid)[dynamics.time_index]     # q2(tau), monotonic
            tau_at_t = np.interp(t_eval, t_grid, tau_grid)        # invert it

            out[:, i, :] = dynamics.from_regularized(sol.sol(tau_at_t).T)

        return t_eval, out.squeeze()

_DEFAULT_REGULARIZED_INTEGRATOR = RegularizedScipyIntegrator(method="RK45", rtol=1e-9, atol=1e-12)