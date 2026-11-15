"""
Shared helpers for the quantum ionization scans.

The scripts in this directory load stationary spectra that were computed and
saved separately (see ``notebooks/plotting/tcc/data/spectrum/quantum_pectra``)
and propagate an initial eigenstate under

    H(t) = H0 + F(t) r,

which is the quantum counterpart of the classical equation used by
``scripts/ionization/classical``:

    r'' = -V'(r) - F(t).

The spectra only describe a finite box, so the population of states with
``E >= spectrum.threshold`` is a *pseudo-continuum* ionization estimate.  It is
not a physical continuum yield until convergence with respect to the box,
basis and retained states has been checked.

The MsC and sC spectra are parametrised by ``alpha``; the Coulomb case is
selected with the acronym ``"C"`` and always uses ``alpha = 0``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from emerald.quantum.dynamics import Spectrum
from emerald.serialization import to_jsonable

# Repository layout.  Resolving from ``__file__`` keeps the scripts usable from
# any working directory, unlike the classical scripts which resolve their input
# data relative to the CWD.
REPO_ROOT = Path(__file__).resolve().parents[3]
SPECTRA_DIR = (
    REPO_ROOT / "notebooks" / "plotting" / "tcc" / "data" / "spectrum" / "quantum_pectra"
)

# The Coulomb (alpha = 0) spectrum in this directory was built from a
# one-sided B-spline basis with a free right boundary:
#
#   bps    = np.logspace(0, np.log10(5001), 1000) - 1
#   basis  = BSplineBasis(bps, 9)
#   spec   = Spectrum.from_representation(basis, Coulomb(), "dirichlet", None)
#
# It has 1006 states (63 bound, 549 below E = 4.0), matching the density of the
# saved MsC/sC spectra.  Propagation itself only needs the saved eigenbasis.

# The two sampling controls requested for the quantum scans:
#
# * SAMPLES_PER_PERIOD sets the integration step, dt = T / SAMPLES_PER_PERIOD,
#   so the field changes little between consecutive steps.
# * SAVE_EVERY is passed as ``record_every`` to the propagator, so only one
#   state per period is kept in the ``Evolution`` history.
SAMPLES_PER_PERIOD = 35
SAVE_EVERY = 35

DEFAULT_ENERGY_CUTOFF = 4.0

POTENTIAL_NAMES = {
    "MsC": "Morse-soft-Coulomb",
    "sC": "soft-Coulomb",
    "C": "Coulomb",
}

POTENTIAL_METADATA = {
    "MsC": {"_type": "MorseSoftCoulomb", "_format_version": 1},
    "sC": {"_type": "SoftCoulomb", "_format_version": 1},
    "C": {"_type": "Coulomb", "_format_version": 1},
}

# Coulomb has no alpha parameter.  Its saved spectrum keeps the conventional
# ``a--0`` tag so that it fits the same filename pattern as the MsC and sC
# spectra, while the metadata stays identical to the classical Coulomb output.
COULOMB_ALPHA = 0.0
COULOMB_SPECTRUM_NAME = "C-Spectrum-a-0.json"


# ---------------------------------------------------------------------------
# Spectrum lookup
# ---------------------------------------------------------------------------

def alpha_tag(alpha: float) -> str:
    """Filename tag used by the saved spectra (``sqrt2`` is special-cased)."""
    alpha = float(alpha)
    if np.isclose(alpha, np.sqrt(2.0)):
        return "sqrt2"
    return f"{alpha:g}"


def spectrum_path(acronym: str, alpha: float) -> Path:
    """Path of the saved spectrum for a potential acronym and alpha."""
    if acronym == "C":
        return SPECTRA_DIR / COULOMB_SPECTRUM_NAME
    return SPECTRA_DIR / f"{acronym}-Spectrum-a-{alpha_tag(alpha)}.json"


def resolve_alpha(acronym: str, alpha: float) -> float:
    """Return the alpha actually used by a potential.

    Coulomb is not parametrised by alpha, so it always resolves to zero; this
    keeps ``--potential C`` usable without requiring ``--alpha 0``.
    """
    return COULOMB_ALPHA if acronym == "C" else float(alpha)


def available_spectra(acronym: str) -> list[str]:
    """Names of the saved spectra available for one potential."""
    return sorted(path.name for path in SPECTRA_DIR.glob(f"{acronym}-Spectrum-a-*.json"))


def load_spectrum(
    acronym: str,
    alpha: float,
    energy_cutoff: float | None = DEFAULT_ENERGY_CUTOFF,
    path: str | Path | None = None,
) -> tuple[Spectrum, Path]:
    """
    Load a saved spectrum, optionally truncating it at an energy cutoff.

    Parameters
    ----------
    acronym : {"MsC", "sC", "C"}
        Potential acronym, used to find the default file and for metadata.
    alpha : float
        Potential parameter.  It must correspond to one of the saved spectra;
        it is ignored for ``"C"`` (Coulomb has no alpha).
    energy_cutoff : float or None
        Keep only states with ``E < energy_cutoff``.  ``None`` keeps the full
        spectrum.
    path : str or Path, optional
        Explicit spectrum file.  Overrides the ``acronym``/``alpha`` lookup.

    Returns
    -------
    spectrum, path
    """
    if acronym not in POTENTIAL_NAMES:
        raise ValueError(
            f"Unknown potential acronym '{acronym}'. "
            f"Choose one of {sorted(POTENTIAL_NAMES)}."
        )

    spectrum_file = Path(path) if path is not None else spectrum_path(acronym, alpha)
    if not spectrum_file.is_file():
        available = ", ".join(available_spectra(acronym))
        raise FileNotFoundError(
            f"No saved spectrum at {spectrum_file}.\n"
            f"Available {acronym} spectra: {available}"
        )

    with spectrum_file.open() as handle:
        spectrum = Spectrum.from_dict(json.load(handle))

    if energy_cutoff is not None:
        n_states = int(
            np.searchsorted(spectrum.energies, float(energy_cutoff), side="right")
        )
        n_states = max(1, min(n_states, spectrum.n_states))
        if n_states < spectrum.n_states:
            spectrum = spectrum.truncate(n_states)

    return spectrum, spectrum_file


def potential_metadata(acronym: str, alpha: float) -> dict:
    """Serialisable metadata for the potential (without constructing it)."""
    try:
        metadata = dict(POTENTIAL_METADATA[acronym])
    except KeyError:
        raise ValueError(
            f"Unknown potential acronym '{acronym}'. "
            f"Choose one of {sorted(POTENTIAL_NAMES)}."
        ) from None
    if acronym != "C":
        metadata["alpha"] = float(alpha)
    return to_jsonable(metadata)


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------

def build_time_grid(
    total_time: float,
    ang_frequency: float,
    samples_per_period: int = SAMPLES_PER_PERIOD,
    save_every: int = SAVE_EVERY,
) -> tuple[np.ndarray, float]:
    """
    Uniform time grid with a conservative field sampling.

    The target step is ``T / samples_per_period`` with ``T = 2*pi/omega``.  The
    number of steps is rounded up to a multiple of ``save_every`` so that the
    last saved point is exactly ``total_time`` and the recorded grid is
    ``times[::save_every]`` (one point per period when the two controls are
    equal).

    Returns
    -------
    times : np.ndarray
        Uniform time grid from 0 to ``total_time``.
    dt : float
        Actual step of the grid (``times[1] - times[0]``).
    """
    if ang_frequency <= 0.0:
        raise ValueError("ang_frequency must be positive.")
    if total_time <= 0.0:
        raise ValueError("total_time must be positive.")
    if samples_per_period < 1 or save_every < 1:
        raise ValueError("samples_per_period and save_every must be positive integers.")

    period = 2.0 * np.pi / ang_frequency
    dt_target = period / samples_per_period
    stride = dt_target * save_every
    n_steps = save_every * int(np.ceil(total_time / stride))
    n_steps = max(n_steps, 1)

    times = np.linspace(0.0, float(total_time), n_steps + 1)
    return times, float(times[1] - times[0])


# ---------------------------------------------------------------------------
# Measurements and output
# ---------------------------------------------------------------------------

def summarize_ionization(evolution) -> dict:
    """
    Summary of the pseudo-continuum population along one propagation.

    Returns the final, mean, final-mean and maximum values together with the
    full history and the recorded times.  ``final_mean`` averages the last 5 %
    of the recorded points, which reduces the dependence on the exact field
    phase at the final time.
    """
    ionization = np.asarray(evolution.ionization_probability(), dtype=float)
    tail = max(1, int(np.ceil(0.05 * ionization.size)))
    norm = np.asarray(evolution.norm, dtype=float)
    norm_drift = float(np.max(np.abs(norm - norm[0]))) if norm.size else 0.0

    return {
        "final": float(ionization[-1]),
        "mean": float(ionization.mean()),
        "final_mean": float(ionization[-tail:].mean()),
        "max": float(ionization.max()),
        "history": ionization.tolist(),
        "times": np.asarray(evolution.times, dtype=float).tolist(),
        "norm_drift": norm_drift,
    }


def timestamp() -> str:
    """Timestamp used in output filenames, matching the classical scripts."""
    return datetime.now().strftime(r"%Y-%m-%d--%H-%M")


def make_output_path(output_dir: str | Path, filename: str) -> Path:
    """Create the output directory if needed and return the target path."""
    directory = Path(output_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename


def write_json(path: str | Path, data: dict) -> None:
    """Write ``data`` as JSON, converting numpy objects first."""
    with Path(path).open("w") as handle:
        json.dump(to_jsonable(data), handle, indent=2)
