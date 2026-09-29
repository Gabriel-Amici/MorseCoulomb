import numpy as np
from scipy.integrate import solve_ivp

from .base import Integrator


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