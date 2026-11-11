"""
Envelope fields: modulate a carrier field with a plateau-shaped window.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .base import Field

_BLACKMAN_A = 0.16

_RampFunction = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Ramp shapes
# ---------------------------------------------------------------------------

def _linear_ramp(s: np.ndarray) -> np.ndarray:
    return s


def _linear_ramp_d1(s: np.ndarray) -> np.ndarray:
    return np.ones_like(s, dtype=float)


def _linear_ramp_d2(s: np.ndarray) -> np.ndarray:
    return np.zeros_like(s, dtype=float)


def _sin2_ramp(s: np.ndarray) -> np.ndarray:
    return np.sin(0.5 * np.pi * s) ** 2


def _sin2_ramp_d1(s: np.ndarray) -> np.ndarray:
    return 0.5 * np.pi * np.sin(np.pi * s)


def _sin2_ramp_d2(s: np.ndarray) -> np.ndarray:
    return 0.5 * np.pi ** 2 * np.cos(np.pi * s)


def _blackman_ramp(s: np.ndarray, a: float = _BLACKMAN_A) -> np.ndarray:
    return 0.5 * (1.0 - a - np.cos(np.pi * s) + a * np.cos(2.0 * np.pi * s))


def _blackman_ramp_d1(s: np.ndarray, a: float = _BLACKMAN_A) -> np.ndarray:
    return 0.5 * (
        np.pi * np.sin(np.pi * s)
        - 2.0 * np.pi * a * np.sin(2.0 * np.pi * s)
    )


def _blackman_ramp_d2(s: np.ndarray, a: float = _BLACKMAN_A) -> np.ndarray:
    return 0.5 * (
        np.pi ** 2 * np.cos(np.pi * s)
        - 4.0 * np.pi ** 2 * a * np.cos(2.0 * np.pi * s)
    )


_SHAPES: dict[str, tuple[_RampFunction, _RampFunction, _RampFunction]] = {
    "linear": (_linear_ramp, _linear_ramp_d1, _linear_ramp_d2),
    "sin2": (_sin2_ramp, _sin2_ramp_d1, _sin2_ramp_d2),
    "blackman": (_blackman_ramp, _blackman_ramp_d1, _blackman_ramp_d2),
}

_SHAPE_ALIASES: dict[str, str] = {
    "linear": "linear",
    "sin2": "sin2",
    "sin²": "sin2",
    "sin_squared": "sin2",
    "blackman": "blackman",
}


def _canonical_shape(shape: str) -> str:
    if not isinstance(shape, str):
        raise TypeError("shape must be a string.")
    try:
        return _SHAPE_ALIASES[shape.lower()]
    except KeyError:
        valid = ", ".join(sorted(set(_SHAPE_ALIASES.values())))
        raise ValueError(
            f"Unknown envelope shape '{shape}'. Valid shapes: {valid}."
        ) from None


def _shape_values(shape: str, s: np.ndarray, order: int = 0) -> np.ndarray:
    if order not in (0, 1, 2):
        raise NotImplementedError(f"order-{order} derivatives not supported")
    return _SHAPES[shape][order](s)


# ---------------------------------------------------------------------------
# Envelope field
# ---------------------------------------------------------------------------

class PlateauEnvelope(Field):
    """
    A carrier field multiplied by a trapezoidal envelope.

    The envelope has three consecutive segments: a rising ramp, a plateau
    at unit amplitude, and a falling ramp.  Segment lengths are given in
    units of the carrier period::

        duration = (ramp_up + plateau + ramp_down) * carrier.period

    The pulse starts at ``t = 0`` and is zero outside
    ``[0, duration]``.  The carrier itself is evaluated at the absolute
    time, so its phase convention is preserved.

    Parameters
    ----------
    carrier
        Field to envelop.  It must provide a ``period`` attribute unless
        ``period`` is supplied explicitly.
    ramp_up, plateau, ramp_down
        Duration of each segment, in units of the carrier period.  All
        values must be non-negative, and their sum must be positive.
    shape
        Ramp shape.  Options are ``'linear'`` (default), ``'sin2'``
        (also accepted as ``'sin²'`` or ``'sin_squared'``), and
        ``'blackman'``.
    period
        Optional carrier period override.  Defaults to
        ``carrier.period``.
    """

    _serialized_parameters = {
        "carrier": "carrier",
        "ramp_up": "ramp_up",
        "plateau": "plateau",
        "ramp_down": "ramp_down",
        "shape": "shape",
        "period": "period",
    }
    _format_version = 1

    def __init__(
        self,
        carrier: Field,
        ramp_up: float,
        plateau: float,
        ramp_down: float,
        shape: str = "linear",
        period: float | None = None,
    ):
        super().__init__()

        self.carrier = carrier
        self.shape = _canonical_shape(shape)

        self.ramp_up = self._validate_non_negative("ramp_up", ramp_up)
        self.plateau = self._validate_non_negative("plateau", plateau)
        self.ramp_down = self._validate_non_negative("ramp_down", ramp_down)

        if period is None:
            period = getattr(carrier, "period", None)
        if period is None:
            raise ValueError(
                "The carrier has no 'period' attribute; pass period explicitly."
            )
        self.period = self._validate_positive("period", period)

        if self.total_periods <= 0:
            raise ValueError("The envelope duration must be greater than zero.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite, non-negative number.")
        return value

    @staticmethod
    def _validate_positive(name: str, value: float) -> float:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite, positive number.")
        return value

    # ------------------------------------------------------------------
    # Segment durations
    # ------------------------------------------------------------------

    @property
    def ramp_up_time(self) -> float:
        """Rising-ramp duration in physical time units."""
        return self.ramp_up * self.period

    @property
    def plateau_time(self) -> float:
        """Plateau duration in physical time units."""
        return self.plateau * self.period

    @property
    def ramp_down_time(self) -> float:
        """Falling-ramp duration in physical time units."""
        return self.ramp_down * self.period

    @property
    def total_periods(self) -> float:
        """Total envelope length in units of the carrier period."""
        return self.ramp_up + self.plateau + self.ramp_down

    @property
    def duration(self) -> float:
        """Total envelope length in physical time units."""
        return self.total_periods * self.period

    # ------------------------------------------------------------------
    # Carrier properties
    # ------------------------------------------------------------------

    @property
    def frequency(self) -> float:
        """Carrier frequency, or a value derived from the period."""
        carrier_frequency = getattr(self.carrier, "frequency", None)
        if carrier_frequency is not None:
            return carrier_frequency

        for attribute in ("ang_frequency", "angular_frequency"):
            value = getattr(self.carrier, attribute, None)
            if value is not None:
                return value / (2.0 * np.pi)

        return 1.0 / self.period

    @property
    def angular_frequency(self) -> float:
        """Carrier angular frequency, or a value derived from the period."""
        for attribute in ("ang_frequency", "angular_frequency"):
            value = getattr(self.carrier, attribute, None)
            if value is not None:
                return value

        carrier_frequency = getattr(self.carrier, "frequency", None)
        if carrier_frequency is not None:
            return 2.0 * np.pi * carrier_frequency

        return 2.0 * np.pi / self.period

    @property
    def ang_frequency(self) -> float:
        """Alias for :attr:`angular_frequency`."""
        return self.angular_frequency

    # ------------------------------------------------------------------
    # Envelope evaluation
    # ------------------------------------------------------------------

    def _envelope(self, times: np.ndarray, order: int = 0) -> np.ndarray:
        """Envelope value or derivative of the requested order."""
        t = np.asarray(times, dtype=float)
        result = np.zeros_like(t, dtype=float)

        ramp_up = self.ramp_up_time
        plateau = self.plateau_time
        ramp_down = self.ramp_down_time
        duration = self.duration

        if ramp_up > 0.0:
            mask = (t >= 0.0) & (t < ramp_up)
            s = t / ramp_up
            result = np.where(
                mask,
                _shape_values(self.shape, s, order) / ramp_up ** order,
                result,
            )

        if plateau > 0.0 and order == 0:
            mask = (t >= ramp_up) & (t < ramp_up + plateau)
            result = np.where(mask, 1.0, result)

        if ramp_down > 0.0:
            mask = (t >= ramp_up + plateau) & (t < duration)
            s = (duration - t) / ramp_down
            sign = (-1.0) ** order
            result = np.where(
                mask,
                sign * _shape_values(self.shape, s, order) / ramp_down ** order,
                result,
            )

        return result

    # ------------------------------------------------------------------
    # Field interface
    # ------------------------------------------------------------------

    def value(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        return self.carrier(t) * self._envelope(t, order=0)

    def first_derivative(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        envelope = self._envelope(t, order=0)
        envelope_d1 = self._envelope(t, order=1)
        carrier = self.carrier(t)
        carrier_d1 = self.carrier.derivative(t, order=1)
        return envelope_d1 * carrier + envelope * carrier_d1

    def second_derivative(self, times: np.ndarray) -> np.ndarray:
        t = np.asarray(times, dtype=float)
        envelope = self._envelope(t, order=0)
        envelope_d1 = self._envelope(t, order=1)
        envelope_d2 = self._envelope(t, order=2)
        carrier = self.carrier(t)
        carrier_d1 = self.carrier.derivative(t, order=1)
        carrier_d2 = self.carrier.derivative(t, order=2)
        return (
            envelope_d2 * carrier
            + 2.0 * envelope_d1 * carrier_d1
            + envelope * carrier_d2
        )
