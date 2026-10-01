"""
adaptive_sampling.py
=====================

A general-purpose implementation of 1D adaptive/monitor-function sampling.

Design goals
------------
1.  Decouple three concerns that were fused together in the original
    notebook procedure:
        (a) HOW to evaluate f, f', f''            -> `Differentiable` protocol
        (b) HOW to turn those into a density       -> `Monitor`
        (c) HOW to turn a density into samples      -> `AdaptiveSampler`
2.  Work equally well when you only have a fine grid + evaluated values
    (`GridFunction`, uses PCHIP-derivative finite differencing) and when
    you have closed-form f/f'/f'' (`AnalyticFunction`). Both satisfy the
    same `Differentiable` protocol, so every `Monitor` and the
    `AdaptiveSampler` itself are agnostic to which one you hand them.
3.  Monitors are composable: `Monitor` instances support `+`, matching the
    "sum of two monitors is a monitor" remark in the original notebook.

Two correctness notes relative to the original notebook (see README-ish
comments near `SlopeMonitor` and `AdaptiveSampler.sample`):
    - the slope monitor must use |f'(x)|, not f'(x) itself, or the density
      can go negative and break the monotonicity of the CDF;
    - "PCHIP" in the original was actually `interp1d(kind='cubic')`, a
      plain cubic-spline inverse (not shape-preserving, and used with
      `fill_value='extrapolate'` on the CDF axis, which is a real hazard
      because slight float overshoot of a random `u` above 1.0 would
      silently extrapolate a sample outside [a, b]). Genuine PCHIP is
      `scipy.interpolate.PchipInterpolator`, used here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator, interp1d

from .maths import Differentiable


class Monitor:
    """
    Base class for monitor functions M(x).

    A Monitor is just `(func: Differentiable, x: array) -> density array`.
    Subclass and implement `__call__`. Monitors compose with `+`.
    """

    def __call__(self, func: Differentiable, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __add__(self, other: Monitor) -> Monitor:
        return _SumMonitor(self, other)


class _SumMonitor(Monitor):
    def __init__(self, *parts: Monitor) -> None:
        # flatten nested sums so a chain of `+` doesn't build a deep tree
        flat: list[Monitor] = []

        for p in parts:
            flat.extend(p.parts if isinstance(p, _SumMonitor) else [p])

        self.parts = flat

    def __call__(self, func: Differentiable, x: np.ndarray) -> np.ndarray:
        return sum(p(func, x) for p in self.parts)


class ConstantMonitor(Monitor):
    """
    Flat baseline density, useful on its own, or added to another
    monitor if desired to keep the baseline explicit instead of folding
    it into e.g. SlopeMonitor's `baseline` term.
    """

    def __init__(self, level: float = 1.0) -> None:
        self.level = level

    def __call__(self, func: Differentiable, x: np.ndarray) -> np.ndarray:
        return np.full(np.shape(x), self.level, dtype=float)


class SlopeMonitor(Monitor):
    """
    M(x; lambda, p) = (baseline + lambda * |f'(x)|**p) ** (1/p).
    """

    def __init__(self, lamda: float = 1.0, p: float = 1.0, baseline: float = 1.0) -> None:
        self.lamda, self.p, self.baseline = lamda, p, baseline

    def __call__(self, func: Differentiable, x: np.ndarray) -> np.ndarray:
        dfx = func.derivative(x, order=1)
        return (self.baseline + self.lamda * np.abs(dfx) ** self.p) ** (1.0 / self.p)


class CurvatureMonitor(Monitor):
    """
    M(x; lambda) = baseline + lambda * kappa(x),
    
    kappa(x) = |f''(x)| / (1 + f'(x)**2)**1.5
    """

    def __init__(self, lamda: float = 1.0, baseline: float = 1.0) -> None:
        self.lamda, self.baseline = lamda, baseline

    def __call__(self, func: Differentiable, x: np.ndarray) -> np.ndarray:
        d1 = func.derivative(x, order=1)
        d2 = func.derivative(x, order=2)
        kappa = np.abs(d2) / (1.0 + d1 ** 2) ** 1.5
        return self.baseline + self.lamda * kappa

class SuperGaussianMonitor(Monitor):

    def __init__(self, lamda: float = 1.0,
                 sigma_R: float = 1.0, sigma_L: float = 1.0,
                 center: float = 0.0, p: float = 2) -> None:
        self.lamda = lamda
        self.sigma_R, self.sigma_L = sigma_R, sigma_L
        self.center = center
        self.p = p

    def __call__(self, func: Differentiable, x: np.ndarray) -> np.ndarray:
        exponent = np.where( x >= self.center,
                             -np.abs((x - self.center) / self.sigma_R) ** self.p,
                             -np.abs((x - self.center) / self.sigma_L) ** self.p)
        return self.lamda * np.exp(exponent)

class LogarithmicTailMonitor(Monitor):

    def __init__(self, lamda: float = 1.0, L: float = 1.0) -> None:
        self.lamda, self.L = lamda, L

    def __call__(self, func: Differentiable, x: np.ndarray) -> np.ndarray:
        return self.lamda*np.where( x >= 0, 1/(x+self.L), 1/self.L)

# --------------------------------------------------------------------------
# 3. The sampler: build the equidistribution CDF once, draw samples many times
# --------------------------------------------------------------------------

InterpMethod = Literal["linear", "pchip"]
SampleMethod = Literal["uniform", "random"]


class AdaptiveSampler:
    """
    Builds the monitor's CDF over [a, b] once; `sample()` can then be
    called repeatedly (uniform or random draws, any interpolation method)
    without recomputing the integral.
    """

    def __init__(
        self,
        func: Differentiable,
        a: float,
        b: float,
        monitor: Monitor,
        n_grid: int = 4001,
    ) -> None:
        self.func = func
        self.a, self.b = float(a), float(b)
        self.monitor = monitor
        self.x_grid = np.linspace(self.a, self.b, n_grid)
        self._build_cdf()

    def _build_cdf(self) -> None:
        g = np.asarray(self.monitor(self.func, self.x_grid), dtype=float)
        if np.any(g <= 0):
            raise ValueError(
                "monitor function must be strictly positive on [a, b] for "
                "the CDF to be invertible -- add a ConstantMonitor baseline "
                "or increase an existing monitor's baseline term"
            )
        G = cumulative_trapezoid(g, self.x_grid, initial=0.0)
        G /= G[-1]
        self.g_grid = g
        self.G_grid = G

    def sample(
        self,
        n: int,
        method: SampleMethod = "uniform",
        interp: InterpMethod = "pchip",
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if method == "uniform":
            u = np.linspace(0.0, 1.0, n)
        elif method == "random":
            rng = rng if rng is not None else np.random.default_rng()
            u = np.sort(rng.uniform(0.0, 1.0, n))
        else:
            raise ValueError(f"unknown sample method {method!r}")

        # G_grid spans exactly [0, 1] by construction and u in [0, 1] by
        # construction too, so no extrapolation should ever be needed --
        # clip defensively instead of extrapolating (unlike the original,
        # which used fill_value='extrapolate' on this axis).
        
        u = np.clip(u, self.G_grid[0], self.G_grid[-1])

        if interp == "linear":
            inv_cdf = interp1d(self.G_grid, self.x_grid, kind="linear")
        elif interp == "pchip":
            inv_cdf = PchipInterpolator(self.G_grid, self.x_grid)
        else:
            raise ValueError(f"unknown interpolation method {interp!r}")

        return np.asarray(inv_cdf(u))
