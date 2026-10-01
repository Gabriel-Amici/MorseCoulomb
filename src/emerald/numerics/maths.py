from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import PchipInterpolator


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
    df: Callable[[np.ndarray], np.ndarray] | None = None
    d2f: Callable[[np.ndarray], np.ndarray] | None = None

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
