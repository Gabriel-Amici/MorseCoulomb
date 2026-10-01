"""
Time evolution of wavepackets expanded in the eigenbasis of a `Spectrum`.

Overview
--------
The state of the system is a vector of complex coefficients ``c`` with

    |psi(t)> = sum_n c_n(t) |phi_n>,

where |phi_n> are the eigenstates kept in the `Spectrum` (H0 |phi_n> = E_n |phi_n>).
Everything in this module works on those coefficients:

* `StationaryPropagator`      exact evolution under H0 alone;
* `SplitOperatorPropagator`   evolution under H(t) = H0 + F(t) * coupling(x);
* `Evolution`                 the result: coefficients over time plus the
                              tools to extract populations, mean values,
                              ionization and wavefunctions.

Both propagators share the signature ``propagate(coeffs0, times, record_every)``
(the `Propagator` protocol), so a new scheme can be plugged in without changing
the analysis code.

Conventions
-----------
* Atomic units (hbar = 1); time, energy and field must be consistent.
* The whole evolution happens in the subspace spanned by the states kept in
  the spectrum. It is exactly unitary there: the norm is conserved to machine
  precision, which therefore says nothing about accuracy. Test convergence
  against the time step and the number of kept states instead.
* Initial coefficients may be a vector of shape (n_states,) or a matrix
  (n_states, K) holding K independent wavepackets that are propagated at
  once with the same field (the quantum analogue of an ensemble).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from tqdm import tqdm

from .dynamics import Spectrum


def _prepare_coefficients(coeffs0, n_states):
    """
    Copies the initial coefficients as complex128 and puts them in the
    internal (n_states, K) layout. Returns the array and whether the input
    was a single (1-D) wavepacket.
    """
    c = np.array(coeffs0, dtype=np.complex128)  # copy: never touch caller's data
    single = c.ndim == 1
    if single:
        c = c[:, None]
    if c.ndim != 2 or c.shape[0] != n_states:
        raise ValueError(
            f"Initial coefficients must have shape ({n_states},) or "
            f"({n_states}, K); got {np.shape(coeffs0)}."
        )
    return c, single


@dataclass
class Evolution:
    """
    Result of a propagation: the coefficients of the wavepacket in the
    eigenbasis of `spectrum` at the recorded times.

    This is the quantum counterpart of a `Trajectory`.

    Attributes
    ----------
    spectrum : `Spectrum`
        Defines the eigenbasis; also needed to turn coefficients back into
        wavefunctions and to know which states are bound.
    times : `np.ndarray`, shape (T,)
        Recorded times.
    coeffs : `np.ndarray`, shape (n_states, T) or (n_states, T, K)
        ``coeffs[n, t]`` is c_n(times[t]). The last axis exists only when K
        wavepackets were propagated together.
    """
    spectrum: Spectrum
    times: np.ndarray
    coeffs: np.ndarray

    # ------------------------------------------------------------------
    # Populations and norm
    # ------------------------------------------------------------------

    @property
    def populations(self) -> np.ndarray:
        """|cₙ(t)|²: probability of finding the system in eigenstate n.
        Same shape as `coeffs`."""
        return np.abs(self.coeffs) ** 2

    @property
    def norm(self) -> np.ndarray:
        """⟨ψ|ψ⟩ at every recorded time, shape (T,) or (T, K).
        Should stay equal to its initial value (1 for normalized states)."""
        return self.populations.sum(axis=0)

    # ------------------------------------------------------------------
    # Mean values
    # ------------------------------------------------------------------

    def expectation(self, operator: np.ndarray) -> np.ndarray:
        """
        Mean value of an observable at every recorded time,

            ⟨O⟩(t) = ⟨ψ(t)| O |ψ(t)⟩.

        The operator must be given in the eigenbasis of the spectrum (built
        with `Spectrum.matrix(func)` or `Spectrum.projector(states)`), and it
        must be Hermitian (only the real part is returned). The state is
        assumed normalized: the result is not divided by the norm.

        Two kinds of operator are accepted, which decides the formula used:

        * **full matrix**, shape (n_states, n_states):

              ⟨O⟩(t) = Σₙₘ conj(c_n(t)) O[n, m] c_m(t)

          i.e. the quadratic form ``c^dagger O c``. Cost O(n^2) per time.

        * **diagonal operator**, 1-D array of shape (n_states,) holding the
          diagonal ``O[n, n]`` (all other entries are zero):

              ⟨O⟩(t) = Σₙ O[n] |c_n(t)|^2

          i.e. a population-weighted average. Cost O(n) per time. Projectors
          and the energy operator H0 (``spectrum.energies``) are of this
          kind.

        Parameters
        ----------
        operator : `np.ndarray`
            Matrix (n, n) or diagonal vector (n,).

        Returns
        -------
        `np.ndarray` of shape (T,), or (T, K) for an ensemble.

        Examples
        --------
        >>> x = spec.matrix(lambda x: x)
        >>> ev.expectation(x)                      # ⟨x⟩(t)
        >>> ev.expectation(spec.energies)          # ⟨H0⟩(t)
        >>> ev.expectation(spec.projector([0]))    # population of the ground state
        """
        op = np.asarray(operator)
        n = self.spectrum.n_states
        if op.shape not in ((n,), (n, n)):
            raise ValueError(
                f"Operator must have shape ({n},) or ({n}, {n}); got {op.shape}."
            )

        if op.ndim == 1:
            # diagonal operator: contract the state index of O with the
            # state index (axis 0) of the populations
            return np.tensordot(op, self.populations, axes=(0, 0))

        # full matrix: (O c)[n, t] = sum_m O[n, m] c[m, t], then contract
        # with conj(c) over the state index
        Oc = np.tensordot(op, self.coeffs, axes=(1, 0))
        return (self.coeffs.conj() * Oc).sum(axis=0).real

    def population(self, states) -> np.ndarray:
        """
        Total population of a set of eigenstates at every recorded time,
        i.e. the mean value of the projector onto them,
        ⟨P⟩ = sum_{n in states} |cₙ|².

        Parameters
        ----------
        states : boolean mask, integer indices or slice
            See `Spectrum.projector`.
        """
        return self.expectation(self.spectrum.projector(states))

    def ionization_probability(self) -> np.ndarray:
        """
        Population outside the bound states, i.e. the mean value of the
        ionization operator  sum_{n : E_n >= threshold} |phi_n><phi_n|.

        Read it after the driving field has been switched off: while the
        field acts, the bound eigenstates of H0 are not the dressed states of
        the system. Within a truncated box it is a population of
        pseudo-continuum states.
        """
        return self.population(~self.spectrum.bound)

    # ------------------------------------------------------------------
    # Back to position space
    # ------------------------------------------------------------------

    def wavefunction(self, index: int, member: int | None = None) -> callable:
        """
        Callable wavefunction psi(x) at the ``index``-th recorded time.

        Parameters
        ----------
        index : int
            Position in `times` (not a time value).
        member : int, optional
            Which wavepacket to use for an ensemble (K > 1).
        """
        if self.coeffs.ndim == 3:
            if member is None:
                raise ValueError("`member` is required for an ensemble evolution.")
            c = self.coeffs[:, index, member]
        else:
            c = self.coeffs[:, index]
        return self.spectrum.reconstruct(c)


class Propagator(Protocol):
    """Anything able to evolve initial coefficients over a time grid."""

    def propagate(self, coeffs0: np.ndarray, times: np.ndarray,
                  record_every: int = 1) -> Evolution: ...


class StationaryPropagator:
    """
    Exact evolution under H0 alone. In the eigenbasis H0 is diagonal, so

        cₙ(t) = exp(-i Eₙ (t - t₀)) cₙ(t₀),      t₀ = times[0],

    which needs no time stepping (and no uniform time grid).
    """

    def __init__(self, spectrum: Spectrum):
        self.spectrum = spectrum

    def propagate(self, coeffs0, times, record_every: int = 1) -> Evolution:
        """
        Parameters
        ----------
        coeffs0 : `np.ndarray`, shape (n_states,) or (n_states, K)
            Coefficients at ``times[0]``.
        times : `np.ndarray`
            Time grid. The initial state is defined at ``times[0]``.
        record_every : int
            Keep one out of every ``record_every`` times (``times[::record_every]``).
        """
        c0, single = _prepare_coefficients(coeffs0, self.spectrum.n_states)
        times = np.asarray(times, dtype=float)[::record_every]

        # phases[n, t] = exp(-i E_n (t - t0))
        phases = np.exp(-1j * np.outer(self.spectrum.energies, times - times[0]))
        coeffs = phases[:, :, None] * c0[:, None, :]          # (n, T, K)

        return Evolution(self.spectrum, times, coeffs[..., 0] if single else coeffs)


class SplitOperatorPropagator:
    """
    Evolution under H(t) = H0 + F(t) * coupling(x) with the second-order
    (Strang) split-operator scheme in the eigenbasis of H0:

        U(t + dt, t) ~ exp(-i H0 dt/2) exp(-i F(t + dt/2) X dt) exp(-i H0 dt/2)

    * exp(-i H0 dt/2) is diagonal in the eigenbasis (just a phase per state).
    * X is the matrix of ``coupling(x)`` in the eigenbasis. It is diagonalized
      once, X = V diag(Ξ) V†, so exp(-i F X dt) = V diag(exp(-i F Ξ dt)) V†
      costs two matrix products per step, whatever the value of F.

    The error per unit time is O(dt^2). The scheme is exactly unitary.

    Parameters
    ----------
    spectrum : `Spectrum`
        Eigenbasis. Truncate it beforehand; the evolution is confined to the
        kept states, and X is the projection of the coupling on them.
    field : callable
        ``field(t) -> float``: value of F at time t (scalar in, scalar out).
    coupling : callable, optional
        Function of position multiplying the field. Defaults to ``x``
        (length gauge, dipole coupling). The sign convention lives here: use
        ``lambda x: -x`` for the opposite one.
    """

    def __init__(self, spectrum: Spectrum, field: Callable[[float], float],
                 coupling: Callable[[np.ndarray], np.ndarray] = lambda x: x):
        self.spectrum = spectrum
        self.field = field

        # Coupling in the eigenbasis of H0 and its diagonalization (done once)
        self.coupling_matrix = spectrum.matrix(coupling)
        self._xi, self._V = np.linalg.eigh(self.coupling_matrix)
        # V is unitary, so its inverse is its conjugate transpose: no need for
        # np.linalg.inv
        self._V_dag = self._V.conj().T

    def step(self, coeffs: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Advances the coefficients from ``t`` to ``t + dt``.

        Parameters
        ----------
        coeffs : `np.ndarray`, shape (n_states,) or (n_states, K)
        t, dt : float
            Start time and step. The field is sampled at the midpoint.

        Returns
        -------
        A new array with the same shape as ``coeffs`` (input is not modified).
        """
        # Reshape helper so the per-state factors broadcast over K columns
        shape = (-1,) + (1,) * (coeffs.ndim - 1)

        half = np.exp(-0.5j * self.spectrum.energies * dt).reshape(shape)
        F = float(self.field(t + 0.5 * dt))
        kick = np.exp(-1j * F * self._xi * dt).reshape(shape)

        c = half * coeffs                       # half step under H0
        c = self._V_dag @ c                     # to the eigenbasis of X
        c = kick * c                            # full step under the coupling
        c = self._V @ c                         # back to the eigenbasis of H0
        return half * c                         # half step under H0

    def propagate(self, coeffs0, times, record_every: int = 1,
                  progress: bool = True) -> Evolution:
        """
        Propagates ``coeffs0`` (defined at ``times[0]``) along a uniform grid.

        Parameters
        ----------
        coeffs0 : `np.ndarray`, shape (n_states,) or (n_states, K)
            Initial coefficients, e.g. ``spectrum.state(0)``.
        times : `np.ndarray`
            Uniform time grid; its spacing is the time step.
        record_every : int
            The state is stored at ``times[::record_every]`` only. The time
            step is unaffected: use it to save memory in long runs.
        progress : bool
            Show a progress bar.

        Returns
        -------
        `Evolution` with ``times[::record_every]`` as recorded times.
        """
        times = np.asarray(times, dtype=float)
        if len(times) < 2:
            raise ValueError("`times` needs at least two points.")
        dt = times[1] - times[0]
        if not np.allclose(np.diff(times), dt):
            raise ValueError("The split-operator scheme needs a uniform time grid.")

        c, single = _prepare_coefficients(coeffs0, self.spectrum.n_states)

        n_states, K = c.shape
        n_recorded = (len(times) - 1) // record_every + 1
        history = np.empty((n_states, n_recorded, K), dtype=np.complex128)
        history[:, 0] = c

        slot = 1
        for k in tqdm(range(len(times) - 1), disable=not progress):
            c = self.step(c, times[k], dt)
            if (k + 1) % record_every == 0:
                history[:, slot] = c
                slot += 1

        return Evolution(self.spectrum, times[::record_every],
                         history[..., 0] if single else history)
