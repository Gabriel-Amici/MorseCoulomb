"""Krotov's method for quantum optimal control.

Custom implementation (no external krotov package) using the split-operator
method in the energy eigenbasis.

Supports:
- Single objective (state-to-state) via :class:`KrotovOptimizer`
- Multiple objectives (simultaneous transitions) via :class:`KrotovMultiOptimizer`
- Ionization / bound-to-continuum via :class:`IonizationOptimizer`

Single objective usage::

    from emerald.quantum.krotov import KrotovOptimizer

    optimizer = KrotovOptimizer(
        energies=energies,
        Xi_vals=Xi_vals, Xi_vecs=Xi_vecs, Xi_conjT=Xi_conjT,
        time_grid=time_grid,
        psi_0=psi_0,
        psi_target=psi_target,
        initial_field=initial_guess_field,
        lambda_a=70.0,
        t_on=10.0,
        t_off=10.0,
        max_iterations=300,
        JT_threshold=1e-4,
    )

    result = optimizer.run()
    optimized_field = result.optimized_field

Multi-objective usage::

    from emerald.quantum.krotov import Objective, KrotovMultiOptimizer

    objectives = [
        Objective(psi_0=gs, psi_target=exc1, weight=0.5),
        Objective(psi_0=gs, psi_target=exc2, weight=0.5),
    ]
    optimizer = KrotovMultiOptimizer(
        objectives=objectives,
        energies=energies, Xi_vals=Xi_vals, Xi_vecs=Xi_vecs, Xi_conjT=Xi_conjT,
        time_grid=time_grid, initial_field=initial_guess_field,
        lambda_a=70, t_on=10, t_off=10,
    )
    result = optimizer.run()

Ionization usage (minimize bound-state population)::

    from emerald.quantum.krotov import IonizationOptimizer

    bound_mask = energies < 0
    optimizer = IonizationOptimizer(
        psi_0=ground_state,
        bound_mask=bound_mask,
        energies=energies, Xi_vals=Xi_vals, Xi_vecs=Xi_vecs, Xi_conjT=Xi_conjT,
        time_grid=time_grid, initial_field=initial_guess_field,
        lambda_a=70, t_on=10, t_off=10,
    )
    result = optimizer.run()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Any

import numpy as np


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------

def blackman_shape(t: np.ndarray | float, t0: float, t1: float, a: float = 0.16) -> np.ndarray | float:
    """Blackman-like shape B(t; t0, t1)."""
    t = np.asarray(t)
    x = (t - t0) / (t1 - t0)
    return 0.5 * (1 - a - np.cos(2 * np.pi * x) + a * np.cos(4 * np.pi * x))


def S_l(t: np.ndarray | float, T: float, t_on: float, t_off: float) -> np.ndarray:
    """Flattop pulse with Blackman ramps.

    Returns 0 at the boundaries t=0 and t=T.
    """
    t = np.asarray(t, dtype=float)
    result = np.zeros_like(t, dtype=float)

    mask1 = (t > 0) & (t < t_on)
    result[mask1] = blackman_shape(t[mask1], 0, 2 * t_on)

    mask2 = (t >= t_on) & (t <= T - t_off)
    result[mask2] = 1.0

    mask3 = (t > T - t_off) & (t < T)
    result[mask3] = blackman_shape(t[mask3], T - 2 * t_off, T)

    return result


# ---------------------------------------------------------------------------
# Split-operator evolution (single step)
# ---------------------------------------------------------------------------

def single_step_evolution(
    state: np.ndarray,
    external_field_val: complex | float,
    exp_H0: np.ndarray,
    Xi_vals: np.ndarray,
    Xi_vecs: np.ndarray,
    Xi_conjT: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Forward split-operator step.

    U(δ) = e^{-iH₀δ/2} · V · e^{-iεΞδ} · V† · e^{-iH₀δ/2}
    """
    tmp = exp_H0 * state
    tmp = Xi_conjT @ tmp
    tmp = np.exp(-1j * external_field_val * Xi_vals * delta_t) * tmp
    tmp = Xi_vecs @ tmp
    return exp_H0 * tmp


def single_step_inverse_evolution(
    state: np.ndarray,
    external_field_val: complex | float,
    exp_H0: np.ndarray,
    Xi_vals: np.ndarray,
    Xi_vecs: np.ndarray,
    Xi_conjT: np.ndarray,
    delta_t: float,
) -> np.ndarray:
    """Backward (inverse) split-operator step.

    U†(δ) = e^{+iH₀δ/2} · V · e^{+iεΞδ} · V† · e^{+iH₀δ/2}
    """
    tmp = exp_H0 * state
    tmp = Xi_conjT @ tmp
    tmp = np.exp(1j * external_field_val * Xi_vals * delta_t) * tmp
    tmp = Xi_vecs @ tmp
    return exp_H0 * tmp


# ---------------------------------------------------------------------------
# Functionals and matrix elements
# ---------------------------------------------------------------------------

def final_time_target_functional(psi_T: np.ndarray, psi_target: np.ndarray) -> float:
    """J_T = 1 - |<ψ(T)|ψ_target>|²."""
    overlap = np.vdot(psi_T, psi_target)
    return float(1 - np.abs(overlap) ** 2)


def co_state_final_condition(
    psi_T: np.ndarray,
    psi_target: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """χ(T) = w · <ψ(T)|ψ_target> · ψ_target.

    For the weighted multi-objective functional
    J_T = (1/Σwₖ) Σₖ wₖ (1 - |<φₖ(T)|φₖᵗᵍᵗ>|²),
    the boundary condition is
    |χₖ(T)⟩ = (wₖ/Σwⱼ) <φₖᵗᵍᵗ|φₖ(T)> |φₖᵗᵍᵗ⟩.

    The normalization by total weight is applied externally;
    here ``weight`` is just wₖ.
    """
    overlap = np.vdot(psi_T, psi_target)
    return weight * overlap * psi_target


def ionization_JT(psi_T: np.ndarray, bound_mask: np.ndarray) -> float:
    """J_T = Σ_{n ∈ bound} |ψ_T[n]|²  (bound-state population).

    Minimizing this functional drives population into the continuum
    (E > 0).  ``psi_T`` is assumed to be in the energy eigenbasis,
    so the bound population is simply the sum of |cₙ|² over bound indices.

    Parameters
    ----------
    psi_T
        Final state in the energy eigenbasis.
    bound_mask
        Boolean array: True for bound-state indices (E < 0).
    """
    return float(np.sum(np.abs(psi_T[bound_mask]) ** 2))


def ionization_co_state_final(
    psi_T: np.ndarray,
    bound_mask: np.ndarray,
) -> np.ndarray:
    """χ(T) = -∂J_T/∂<ψ(T)| = -P_bound |ψ(T)⟩.

    In the energy eigenbasis this is simply:
    χ[n] = -ψ_T[n] for bound states, 0 otherwise.
    """
    chi = np.zeros_like(psi_T)
    chi[bound_mask] = -psi_T[bound_mask]
    return chi


def R_braket(
    chi: np.ndarray,
    phi: np.ndarray,
    Xi_vals: np.ndarray,
    Xi_vecs: np.ndarray,
    Xi_conjT: np.ndarray,
) -> complex:
    """<χ|Ξ|φ> using the eigen-decomposition Ξ = V D V†."""
    Xi_phi = Xi_vecs @ (Xi_vals * (Xi_conjT @ phi))
    return np.vdot(chi, Xi_phi)


# ---------------------------------------------------------------------------
# Objective dataclass
# ---------------------------------------------------------------------------

@dataclass
class Objective:
    """One state-to-state transition objective.

    Parameters
    ----------
    psi_0
        Initial state in the energy eigenbasis.
    psi_target
        Target state in the energy eigenbasis.
    weight
        Relative weight in the multi-objective functional.
    """
    psi_0: np.ndarray
    psi_target: np.ndarray
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class KrotovResult:
    """Container for the output of a Krotov optimization run."""

    optimized_field: np.ndarray
    JT_history: np.ndarray
    n_iterations: int
    converged: bool
    final_overlap: float
    controls: dict[int, np.ndarray] | None = None
    phi_history: np.ndarray | None = None
    chi_history: np.ndarray | None = None
    elapsed_seconds: float = 0.0
    n_objectives: int = 1

    # Per-objective diagnostics (only for multi-objective runs)
    JT_per_objective: np.ndarray | None = None
    overlap_per_objective: np.ndarray | None = None

    @property
    def best_iteration(self) -> int:
        return int(np.argmin(self.JT_history))

    @property
    def best_JT(self) -> float:
        return float(np.min(self.JT_history))

    def __repr__(self) -> str:
        status = "converged" if self.converged else "max iterations"
        return (
            f"KrotovResult({status}, "
            f"iterations={self.n_iterations}, "
            f"best_JT={self.best_JT:.6e}, "
            f"final_overlap={self.final_overlap:.6f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialisable dictionary for saving to JSON.

        Contains all optimisation parameters and results.  The user can
        append additional metadata before calling ``json.dump``::

            d = result.to_dict()
            d['notes'] = 'run with lambda_a=70'
            d['alpha'] = 1.0
            with open('result.json', 'w') as f:
                json.dump(d, f, indent=2)
        """
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_objectives': self.n_objectives,
                'n_iterations': self.n_iterations,
                'converged': self.converged,
                'elapsed_seconds': round(self.elapsed_seconds, 2),
            },
            'results': {
                'best_JT': self.best_JT,
                'best_iteration': self.best_iteration,
                'final_overlap': self.final_overlap,
                'JT_history': self.JT_history.tolist(),
                'optimized_field': self.optimized_field.tolist(),
            },
            'per_objective': (
                {
                    'JT': self.JT_per_objective.tolist() if self.JT_per_objective is not None else None,
                    'overlap': self.overlap_per_objective.tolist() if self.overlap_per_objective is not None else None,
                }
                if self.n_objectives > 1
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_header(n_obj: int = 1) -> None:
    if n_obj <= 1:
        print(f"{'iter.':>5}  {'J_T':>12}  {'|<psi|tgt>|':>10}  {'delta J_T':>12}  {'secs':>6}")
    else:
        print(f"{'iter.':>5}  {'J_T':>12}  {'delta J_T':>12}  {'secs':>6}")
    print("-" * 55)


def _print_row(
    iteration: int,
    JT: float,
    elapsed: float,
    delta_JT: float | None = None,
    overlap: float | None = None,
) -> None:
    dJ = f"{delta_JT:+.3e}" if delta_JT is not None else "n/a"
    if overlap is not None:
        print(f"{iteration:>5}  {JT:>12.6e}  {overlap:>10.6f}  {dJ:>12}  {elapsed:>6.1f}")
    else:
        print(f"{iteration:>5}  {JT:>12.6e}  {dJ:>12}  {elapsed:>6.1f}")


# ===========================================================================
# KrotovOptimizer  (single objective)
# ===========================================================================

class KrotovOptimizer:
    """Run Krotov's method for a single state-to-state transition.

    All pre-computed spectral data (energies, interaction matrix
    eigen-decomposition, half-step phase factors) must be supplied at
    construction time.

    Parameters
    ----------
    energies
        Eigenvalues of H₀ (1-D, length N).
    Xi_vals, Xi_vecs, Xi_conjT
        Eigen-decomposition of the interaction matrix Ξ = R̂ in the energy basis.
        ``Xi_vecs`` has eigenvectors as columns; ``Xi_conjT`` is its conjugate
        transpose.
    time_grid
        1-D array of time points [t₀, t₁, …, t_{N_T-1}].
    psi_0
        Initial state in the energy eigenbasis (1-D, length N).
    psi_target
        Target state in the energy eigenbasis (1-D, length N).
    initial_field
        Guess control field, 1-D array with same length as ``time_grid``.
    lambda_a
        Krotov step-size parameter (inverse step weight).  Larger values give
        smaller, more conservative updates.
    t_on, t_off
        Rise / fall times for the shape function S_l(t).
    max_iterations
        Upper bound on Krotov iterations.
    JT_threshold
        Stop when J_T falls below this value.
    store_histories
        If True, store the full field, state, and co-state history for every
        iteration (uses O(iterations × N × N_T) memory).
    """

    def __init__(
        self,
        *,
        energies: np.ndarray,
        Xi_vals: np.ndarray,
        Xi_vecs: np.ndarray,
        Xi_conjT: np.ndarray,
        time_grid: np.ndarray,
        psi_0: np.ndarray,
        psi_target: np.ndarray,
        initial_field: np.ndarray,
        lambda_a: float = 70.0,
        t_on: float = 10.0,
        t_off: float = 10.0,
        max_iterations: int = 300,
        JT_threshold: float = 1e-4,
        store_histories: bool = False,
    ) -> None:
        self.Xi_vals = Xi_vals
        self.Xi_vecs = Xi_vecs
        self.Xi_conjT = Xi_conjT
        self.time_grid = np.asarray(time_grid)
        self.delta_t = float(self.time_grid[1] - self.time_grid[0])
        self.N_T = len(self.time_grid)
        self.psi_0 = np.asarray(psi_0, dtype=np.complex128).copy()
        self.psi_target = np.asarray(psi_target, dtype=np.complex128).copy()
        self.lambda_a = lambda_a
        self.t_on = t_on
        self.t_off = t_off
        self.max_iterations = max_iterations
        self.JT_threshold = JT_threshold
        self.store_histories = store_histories

        T = float(self.time_grid[-1])
        self.shape_values = S_l(self.time_grid, T, t_on, t_off)

        self.exp_H0_fwd = np.exp(-1j * energies * self.delta_t / 2)
        self.exp_H0_bwd = np.exp(1j * energies * self.delta_t / 2)

        self.current_field = np.asarray(initial_field, dtype=float).copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _forward_propagate(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate |φ⟩ forward with *field*. Returns (phi_history, final_state)."""
        base_size = len(self.psi_0)
        phi_hist = np.zeros((base_size, self.N_T), dtype=np.complex128)
        state = self.psi_0.copy()
        for tau in range(self.N_T):
            phi_hist[:, tau] = state
            state = single_step_evolution(
                state, field[tau],
                self.exp_H0_fwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                self.delta_t,
            )
        return phi_hist, state

    def _backward_propagate(
        self, field: np.ndarray, chi_final: np.ndarray
    ) -> np.ndarray:
        """Propagate |χ⟩ backward from t=T. Returns chi_history."""
        base_size = len(chi_final)
        chi_hist = np.zeros((base_size, self.N_T), dtype=np.complex128)
        chi_hist[:, -1] = chi_final
        for tau in reversed(range(self.N_T - 1)):
            chi_hist[:, tau] = single_step_inverse_evolution(
                chi_hist[:, tau + 1], field[tau + 1],
                self.exp_H0_bwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                self.delta_t,
            )
        return chi_hist

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        callback: Callable[[int, float, float], None] | None = None,
        verbose: bool = True,
    ) -> KrotovResult:
        """Execute the Krotov optimization loop.

        Parameters
        ----------
        callback
            Optional function called after each iteration with
            ``(iteration, JT_value, overlap)``.
        verbose
            Print a progress table to stdout.

        Returns
        -------
        KrotovResult
        """
        t0 = time.time()
        JT_history = np.zeros(self.max_iterations + 1)

        controls: dict[int, np.ndarray] | None = {} if self.store_histories else None
        phi_opt: dict[int, np.ndarray] | None = {} if self.store_histories else None
        chi_opt: dict[int, np.ndarray] | None = {} if self.store_histories else None

        # --- Iteration 0: forward + backward with the guess field ---
        phi_hist, final_state = self._forward_propagate(self.current_field)
        JT_history[0] = final_time_target_functional(final_state, self.psi_target)
        overlap_0 = np.abs(np.vdot(final_state, self.psi_target))

        if verbose:
            _print_header()
            _print_row(0, JT_history[0], elapsed=0.0, delta_JT=None, overlap=overlap_0)

        chi_final = co_state_final_condition(final_state, self.psi_target)
        chi_hist = self._backward_propagate(self.current_field, chi_final)

        if self.store_histories:
            controls[0] = self.current_field.copy()
            phi_opt[0] = phi_hist.copy()
            chi_opt[0] = chi_hist.copy()

        converged = False
        overlap = overlap_0

        # --- Krotov iterations ---
        for i in range(1, self.max_iterations + 1):
            iter_start = time.time()

            prev_field = controls[i - 1] if self.store_histories else self.current_field.copy()

            phi_hist = np.zeros((len(self.psi_0), self.N_T), dtype=np.complex128)
            phi_hist[:, 0] = self.psi_0

            for n in range(1, self.N_T):
                delta_eps = (
                    self.shape_values[n - 1]
                    / self.lambda_a
                    * np.imag(
                        R_braket(
                            chi_hist[:, n - 1],
                            phi_hist[:, n - 1],
                            self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                        )
                    )
                )
                self.current_field[n] = prev_field[n] + delta_eps

                phi_hist[:, n] = single_step_evolution(
                    phi_hist[:, n - 1], self.current_field[n],
                    self.exp_H0_fwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                    self.delta_t,
                )

            final_state = phi_hist[:, -1]
            JT = final_time_target_functional(final_state, self.psi_target)
            JT_history[i] = JT
            overlap = np.abs(np.vdot(final_state, self.psi_target))

            # Backward propagation for the next iteration
            chi_final = co_state_final_condition(final_state, self.psi_target)
            chi_hist = self._backward_propagate(self.current_field, chi_final)

            elapsed = time.time() - iter_start
            delta_JT = JT - JT_history[i - 1]
            if verbose:
                _print_row(i, JT, elapsed=elapsed, delta_JT=delta_JT, overlap=overlap)

            if self.store_histories:
                controls[i] = self.current_field.copy()
                phi_opt[i] = phi_hist.copy()
                chi_opt[i] = chi_hist.copy()

            if callback is not None:
                callback(i, JT, overlap)

            if JT < self.JT_threshold:
                converged = True
                if verbose:
                    print(f"\nConverged: J_T = {JT:.6e} < threshold {self.JT_threshold:.6e} at iteration {i}")
                break

        total_elapsed = time.time() - t0
        last = i

        return KrotovResult(
            optimized_field=self.current_field.copy(),
            JT_history=JT_history[:last + 1],
            n_iterations=last,
            converged=converged,
            final_overlap=float(overlap),
            controls=controls,
            phi_history=phi_opt.get(last) if phi_opt is not None else None,
            chi_history=chi_opt.get(last) if chi_opt is not None else None,
            elapsed_seconds=total_elapsed,
            n_objectives=1,
        )


# ===========================================================================
# KrotovMultiOptimizer  (N objectives)
# ===========================================================================

class KrotovMultiOptimizer:
    """Run Krotov's method for N simultaneous state-to-state transitions.

    The multi-objective functional is::

        J_T = (1 / Σₖ wₖ) Σₖ wₖ (1 - |<φₖ(T)|φₖᵗᵍᵗ>|²)

    The control update sums over all objectives::

        Δε(t) = (S(t)/λₐ) · Im[ Σₖ wₖ · <χₖ(t)|Ξ|φₖ(t)> ]

    Parameters
    ----------
    objectives
        List of :class:`Objective` instances.
    energies, Xi_vals, Xi_vecs, Xi_conjT
        Spectral data (same as :class:`KrotovOptimizer`).
    time_grid
        1-D array of time points.
    initial_field
        Guess control field.
    lambda_a
        Krotov step-size parameter.
    t_on, t_off
        Rise / fall times for the shape function.
    max_iterations
        Upper bound on iterations.
    JT_threshold
        Stop when the total J_T falls below this value.
    store_histories
        If True, store per-iteration histories (memory-intensive for N > 1).
    """

    def __init__(
        self,
        *,
        objectives: list[Objective],
        energies: np.ndarray,
        Xi_vals: np.ndarray,
        Xi_vecs: np.ndarray,
        Xi_conjT: np.ndarray,
        time_grid: np.ndarray,
        initial_field: np.ndarray,
        lambda_a: float = 70.0,
        t_on: float = 10.0,
        t_off: float = 10.0,
        max_iterations: int = 300,
        JT_threshold: float = 1e-4,
        store_histories: bool = False,
    ) -> None:
        if not objectives:
            raise ValueError("At least one objective is required")

        self.objectives = objectives
        self.N_obj = len(objectives)
        self.total_weight = sum(obj.weight for obj in objectives)

        # Validate that all objectives share the same Hilbert space dimension
        base_size = len(objectives[0].psi_0)
        for k, obj in enumerate(objectives):
            assert len(obj.psi_0) == base_size, f"Objective {k} has wrong state dimension"
            assert len(obj.psi_target) == base_size, f"Objective {k} has wrong target dimension"

        self.base_size = base_size
        self.Xi_vals = Xi_vals
        self.Xi_vecs = Xi_vecs
        self.Xi_conjT = Xi_conjT
        self.time_grid = np.asarray(time_grid)
        self.delta_t = float(self.time_grid[1] - self.time_grid[0])
        self.N_T = len(self.time_grid)
        self.lambda_a = lambda_a
        self.t_on = t_on
        self.t_off = t_off
        self.max_iterations = max_iterations
        self.JT_threshold = JT_threshold
        self.store_histories = store_histories

        T = float(self.time_grid[-1])
        self.shape_values = S_l(self.time_grid, T, t_on, t_off)

        self.exp_H0_fwd = np.exp(-1j * energies * self.delta_t / 2)
        self.exp_H0_bwd = np.exp(1j * energies * self.delta_t / 2)

        self.current_field = np.asarray(initial_field, dtype=float).copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _forward_propagate_all(
        self, field: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """Forward propagate all objectives.

        Returns (phi_hist[N_obj, base_size, N_T], final_states[N_obj, base_size], overlaps).
        """
        phi_hist = np.zeros((self.N_obj, self.base_size, self.N_T), dtype=np.complex128)
        final_states = []
        overlaps = []

        for k, obj in enumerate(self.objectives):
            state = obj.psi_0.copy()
            for tau in range(self.N_T):
                phi_hist[k, :, tau] = state
                state = single_step_evolution(
                    state, field[tau],
                    self.exp_H0_fwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                    self.delta_t,
                )
            final_states.append(state)
            overlaps.append(float(np.abs(np.vdot(state, obj.psi_target))))

        return phi_hist, np.array(final_states), overlaps

    def _backward_propagate_all(
        self, field: np.ndarray, chi_finals: np.ndarray
    ) -> np.ndarray:
        """Backward propagate all co-states.

        Returns chi_hist[N_obj, base_size, N_T].
        """
        chi_hist = np.zeros((self.N_obj, self.base_size, self.N_T), dtype=np.complex128)

        for k in range(self.N_obj):
            chi_hist[k, :, -1] = chi_finals[k]
            for tau in reversed(range(self.N_T - 1)):
                chi_hist[k, :, tau] = single_step_inverse_evolution(
                    chi_hist[k, :, tau + 1], field[tau + 1],
                    self.exp_H0_bwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                    self.delta_t,
                )

        return chi_hist

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        callback: Callable[[int, float, np.ndarray], None] | None = None,
        verbose: bool = True,
    ) -> KrotovResult:
        """Execute the multi-objective Krotov loop.

        Parameters
        ----------
        callback
            Called after each iteration with ``(iteration, JT_total, overlaps_array)``.
        verbose
            Print a progress table.

        Returns
        -------
        KrotovResult
        """
        t0 = time.time()
        JT_history = np.zeros(self.max_iterations + 1)
        JT_per_obj_history: list[np.ndarray] = []
        overlap_per_obj_history: list[np.ndarray] = []

        controls: dict[int, np.ndarray] | None = {} if self.store_histories else None

        # --- Iteration 0: forward + backward with the guess field ---
        phi_hist, final_states, overlaps = self._forward_propagate_all(self.current_field)

        # J_T = (1/Σw) Σ w (1 - |overlap|²)
        JT_0 = sum(
            obj.weight * (1 - ov ** 2)
            for obj, ov in zip(self.objectives, overlaps)
        ) / self.total_weight
        JT_history[0] = JT_0
        JT_per_obj_history.append(np.array([1 - ov ** 2 for ov in overlaps]))
        overlap_per_obj_history.append(np.array(overlaps))

        if verbose:
            _print_header(self.N_obj)
            _print_row(0, JT_0, elapsed=0.0, delta_JT=None)

        # Co-state final conditions
        chi_finals = np.zeros((self.N_obj, self.base_size), dtype=np.complex128)
        for k, obj in enumerate(self.objectives):
            chi_finals[k] = co_state_final_condition(final_states[k], obj.psi_target, weight=obj.weight)

        chi_hist = self._backward_propagate_all(self.current_field, chi_finals)

        if self.store_histories:
            controls[0] = self.current_field.copy()

        converged = False

        # --- Krotov iterations ---
        for i in range(1, self.max_iterations + 1):
            iter_start = time.time()

            prev_field = controls[i - 1] if self.store_histories else self.current_field.copy()

            phi_hist = np.zeros((self.N_obj, self.base_size, self.N_T), dtype=np.complex128)
            for k, obj in enumerate(self.objectives):
                phi_hist[k, :, 0] = obj.psi_0

            for n in range(1, self.N_T):
                # Sum over objectives: Σₖ wₖ <χₖ|Ξ|φₖ>
                sum_braket: complex = 0j
                for k, obj in enumerate(self.objectives):
                    sum_braket += obj.weight * R_braket(
                        chi_hist[k, :, n - 1],
                        phi_hist[k, :, n - 1],
                        self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                    )

                delta_eps = (self.shape_values[n - 1] / self.lambda_a) * np.imag(sum_braket)
                self.current_field[n] = prev_field[n] + delta_eps

                # Forward propagate each objective with updated field
                for k in range(self.N_obj):
                    phi_hist[k, :, n] = single_step_evolution(
                        phi_hist[k, :, n - 1], self.current_field[n],
                        self.exp_H0_fwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                        self.delta_t,
                    )

            # Evaluate J_T and per-objective metrics
            overlaps = []
            JT_individual = []
            for k, obj in enumerate(self.objectives):
                fs = phi_hist[k, :, -1]
                ov = float(np.abs(np.vdot(fs, obj.psi_target)))
                overlaps.append(ov)
                JT_individual.append(1 - ov ** 2)

            JT_total = sum(w * j for w, j in zip(
                [obj.weight for obj in self.objectives], JT_individual
            )) / self.total_weight
            JT_history[i] = JT_total
            JT_per_obj_history.append(np.array(JT_individual))
            overlap_per_obj_history.append(np.array(overlaps))

            # Backward propagation for the next iteration
            chi_finals = np.zeros((self.N_obj, self.base_size), dtype=np.complex128)
            for k, obj in enumerate(self.objectives):
                chi_finals[k] = co_state_final_condition(
                    phi_hist[k, :, -1], obj.psi_target, weight=obj.weight
                )
            chi_hist = self._backward_propagate_all(self.current_field, chi_finals)

            elapsed = time.time() - iter_start
            delta_JT = JT_total - JT_history[i - 1]
            if verbose:
                _print_row(i, JT_total, elapsed=elapsed, delta_JT=delta_JT)

            if self.store_histories:
                controls[i] = self.current_field.copy()

            if callback is not None:
                callback(i, JT_total, np.array(overlaps))

            if JT_total < self.JT_threshold:
                converged = True
                if verbose:
                    print(f"\nConverged: J_T = {JT_total:.6e} < threshold {self.JT_threshold:.6e} at iteration {i}")
                break

        total_elapsed = time.time() - t0
        last = i

        # Build per-objective history arrays: shape (n_iters, n_objectives)
        JT_per_obj_arr = np.array(JT_per_obj_history)
        overlap_per_obj_arr = np.array(overlap_per_obj_history)

        # Final overlap is the weighted average for compatibility
        final_overlap_avg = float(np.mean(overlap_per_obj_arr[-1]))

        return KrotovResult(
            optimized_field=self.current_field.copy(),
            JT_history=JT_history[:last + 1],
            n_iterations=last,
            converged=converged,
            final_overlap=final_overlap_avg,
            controls=controls,
            elapsed_seconds=total_elapsed,
            n_objectives=self.N_obj,
            JT_per_objective=JT_per_obj_arr,
            overlap_per_objective=overlap_per_obj_arr,
        )


# ===========================================================================
# IonizationOptimizer  (bound-to-continuum via projector)
# ===========================================================================

class IonizationOptimizer:
    """Krotov optimization for ionization (bound → continuum).

    Instead of targeting a specific state, this minimizes the total bound-state
    population at the final time::

        J_T = Σ_{n ∈ bound} |ψₙ(T)|²

    In the energy eigenbasis this is simply the sum of probabilities over
    bound-state indices.  The co-state boundary condition is the negative
    projection onto the bound subspace::

        |χ(T)⟩ = -P_bound |ψ(T)⟩

    Parameters
    ----------
    psi_0
        Initial state (typically the ground state) in the energy eigenbasis.
    bound_mask
        Boolean array of length ``base_size``: True for indices where the
        energy is below the bound/continuum threshold (typically E < 0).
    energies, Xi_vals, Xi_vecs, Xi_conjT
        Spectral data (same as :class:`KrotovOptimizer`).
    time_grid
        1-D array of time points.
    initial_field
        Guess control field.
    lambda_a
        Krotov step-size parameter.
    t_on, t_off
        Rise / fall times for the shape function.
    max_iterations
        Upper bound on iterations.
    JT_threshold
        Stop when bound population falls below this value.
    store_histories
        If True, store per-iteration histories.
    """

    def __init__(
        self,
        *,
        psi_0: np.ndarray,
        bound_mask: np.ndarray,
        energies: np.ndarray,
        Xi_vals: np.ndarray,
        Xi_vecs: np.ndarray,
        Xi_conjT: np.ndarray,
        time_grid: np.ndarray,
        initial_field: np.ndarray,
        lambda_a: float = 70.0,
        t_on: float = 10.0,
        t_off: float = 10.0,
        max_iterations: int = 300,
        JT_threshold: float = 0.01,
        store_histories: bool = False,
    ) -> None:
        self.psi_0 = np.asarray(psi_0, dtype=np.complex128).copy()
        self.bound_mask = np.asarray(bound_mask, dtype=bool)
        self.n_bound = int(np.sum(self.bound_mask))
        self.base_size = len(psi_0)
        self.Xi_vals = Xi_vals
        self.Xi_vecs = Xi_vecs
        self.Xi_conjT = Xi_conjT
        self.time_grid = np.asarray(time_grid)
        self.delta_t = float(self.time_grid[1] - self.time_grid[0])
        self.N_T = len(self.time_grid)
        self.lambda_a = lambda_a
        self.t_on = t_on
        self.t_off = t_off
        self.max_iterations = max_iterations
        self.JT_threshold = JT_threshold
        self.store_histories = store_histories

        T = float(self.time_grid[-1])
        self.shape_values = S_l(self.time_grid, T, t_on, t_off)

        self.exp_H0_fwd = np.exp(-1j * energies * self.delta_t / 2)
        self.exp_H0_bwd = np.exp(1j * energies * self.delta_t / 2)

        self.current_field = np.asarray(initial_field, dtype=float).copy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _forward_propagate(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate |φ⟩ forward. Returns (phi_history, final_state)."""
        phi_hist = np.zeros((self.base_size, self.N_T), dtype=np.complex128)
        state = self.psi_0.copy()
        for tau in range(self.N_T):
            phi_hist[:, tau] = state
            state = single_step_evolution(
                state, field[tau],
                self.exp_H0_fwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                self.delta_t,
            )
        return phi_hist, state

    def _backward_propagate(self, field: np.ndarray, chi_final: np.ndarray) -> np.ndarray:
        """Propagate |χ⟩ backward from t=T."""
        chi_hist = np.zeros((self.base_size, self.N_T), dtype=np.complex128)
        chi_hist[:, -1] = chi_final
        for tau in reversed(range(self.N_T - 1)):
            chi_hist[:, tau] = single_step_inverse_evolution(
                chi_hist[:, tau + 1], field[tau + 1],
                self.exp_H0_bwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                self.delta_t,
            )
        return chi_hist

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        callback: Callable[[int, float, float], None] | None = None,
        verbose: bool = True,
    ) -> KrotovResult:
        """Execute the ionization Krotov loop.

        Parameters
        ----------
        callback
            Called after each iteration with ``(iteration, JT, unbound_pop)``.
        verbose
            Print a progress table.

        Returns
        -------
        KrotovResult
        """
        t0 = time.time()
        JT_history = np.zeros(self.max_iterations + 1)

        controls: dict[int, np.ndarray] | None = {} if self.store_histories else None

        # --- Iteration 0: forward + backward with the guess field ---
        phi_hist, final_state = self._forward_propagate(self.current_field)
        JT_history[0] = ionization_JT(final_state, self.bound_mask)
        bound_pop_0 = JT_history[0]
        unbound_pop_0 = 1.0 - bound_pop_0

        if verbose:
            print(f"{'iter.':>5}  {'J_T (bound)':>14}  {'unbound pop':>12}  {'delta J_T':>12}  {'secs':>6}")
            print("-" * 61)
            _print_row(0, JT_history[0], elapsed=0.0, delta_JT=None, overlap=unbound_pop_0)

        chi_final = ionization_co_state_final(final_state, self.bound_mask)
        chi_hist = self._backward_propagate(self.current_field, chi_final)

        if self.store_histories:
            controls[0] = self.current_field.copy()

        converged = False

        # --- Krotov iterations ---
        for i in range(1, self.max_iterations + 1):
            iter_start = time.time()

            prev_field = controls[i - 1] if self.store_histories else self.current_field.copy()

            phi_hist = np.zeros((self.base_size, self.N_T), dtype=np.complex128)
            phi_hist[:, 0] = self.psi_0

            for n in range(1, self.N_T):
                delta_eps = (
                    self.shape_values[n - 1]
                    / self.lambda_a
                    * np.imag(
                        R_braket(
                            chi_hist[:, n - 1],
                            phi_hist[:, n - 1],
                            self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                        )
                    )
                )
                self.current_field[n] = prev_field[n] + delta_eps

                phi_hist[:, n] = single_step_evolution(
                    phi_hist[:, n - 1], self.current_field[n],
                    self.exp_H0_fwd, self.Xi_vals, self.Xi_vecs, self.Xi_conjT,
                    self.delta_t,
                )

            final_state = phi_hist[:, -1]
            JT = ionization_JT(final_state, self.bound_mask)
            JT_history[i] = JT
            unbound_pop = 1.0 - JT

            chi_final = ionization_co_state_final(final_state, self.bound_mask)
            chi_hist = self._backward_propagate(self.current_field, chi_final)

            elapsed = time.time() - iter_start
            delta_JT = JT - JT_history[i - 1]
            if verbose:
                _print_row(i, JT, elapsed=elapsed, delta_JT=delta_JT, overlap=unbound_pop)

            if self.store_histories:
                controls[i] = self.current_field.copy()

            if callback is not None:
                callback(i, JT, unbound_pop)

            if JT < self.JT_threshold:
                converged = True
                if verbose:
                    print(f"\nConverged: bound pop = {JT:.6e} < threshold {self.JT_threshold:.6e} at iteration {i}")
                break

        total_elapsed = time.time() - t0
        last = i

        return KrotovResult(
            optimized_field=self.current_field.copy(),
            JT_history=JT_history[:last + 1],
            n_iterations=last,
            converged=converged,
            final_overlap=float(1.0 - JT),
            controls=controls,
            elapsed_seconds=total_elapsed,
            n_objectives=1,
        )
