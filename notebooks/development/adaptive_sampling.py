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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol, runtime_checkable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import PchipInterpolator, interp1d

# --------------------------------------------------------------------------
# 1. The function abstraction
# --------------------------------------------------------------------------

@runtime_checkable
class Differentiable(Protocol):
    """Anything that can be evaluated and differentiated up to 2nd order.

    Monitors and the sampler only ever talk to this interface, so they
    don't care whether derivatives come from closed-form expressions or
    from finite-differencing a fine grid.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray: ...


@dataclass
class AnalyticFunction:
    """Wraps closed-form callables.

    Only the derivative orders you actually need for your chosen
    monitor(s) have to be supplied.
    """

    f: Callable[[np.ndarray], np.ndarray]
    df: Optional[Callable[[np.ndarray], np.ndarray]] = None
    d2f: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.f(np.asarray(x, dtype=float))

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if order == 1:
            if self.df is None:
                raise ValueError("no df supplied to this AnalyticFunction")
            return self.df(x)
        if order == 2:
            if self.d2f is None:
                raise ValueError("no d2f supplied to this AnalyticFunction")
            return self.d2f(x)
        raise NotImplementedError(f"order-{order} derivatives not supported")


@dataclass
class GridFunction:
    """Wraps a fine grid + evaluated values -- no analytic form required.

    A monotone PCHIP interpolant is built through (x_grid, y_grid); f and
    its derivatives are then evaluated as the interpolant and its exact
    derivative splines, so they can be evaluated at *any* x in
    range, not just at grid nodes. 
    This is both more robust and more convenient than raw `np.gradient` 
    finite differences, especially for the 2nd derivative, which amplifies 
    grid noise badly under naive differencing.
    """

    x_grid: np.ndarray
    y_grid: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.x_grid, dtype=float)
        y = np.asarray(self.y_grid, dtype=float)

        order = np.argsort(x)

        x, y = x[order], y[order]

        if np.any(np.diff(x) <= 0):
            raise ValueError("x_grid must have no duplicate points")

        self.x_grid, self.y_grid = x, y
        self._spline = PchipInterpolator(x, y)
        self._d1 = self._spline.derivative(1)
        self._d2 = self._spline.derivative(2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._spline(x)

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        if order == 1:
            return self._d1(x)
        if order == 2:
            return self._d2(x)
        raise NotImplementedError(f"order-{order} derivatives not supported")


# --------------------------------------------------------------------------
# 2. Monitor functions
# --------------------------------------------------------------------------

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
        rng: Optional[np.random.Generator] = None,
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
