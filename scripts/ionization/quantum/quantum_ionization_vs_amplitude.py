"""
Quantum counterpart of ``classical_ionization_vs_amplitude.py``.

Propagates the ground state of a saved MsC (or sC) spectrum under a sinusoidal
field with fixed angular frequency and increasing amplitude,

    H(t) = H0 + F(t) r,      F(t) = F0 sin(omega t),

and reports the population of the pseudo-continuum states (``E >= 0``) at the
end of each run.

The sC potential is intentionally left out of the default amplitude scan; add
``--potential sC`` (or run with a saved sC spectrum via ``--spectrum``) if it is
needed.  sC is mainly used by the stabilization script.  ``--potential C``
selects the Coulomb spectrum (the alpha = 0 case).

PARAMETERS

--alpha                     potential parameter (default 0.001; ignored for C)
--potential {MsC,sC,C}      which saved spectrum to use (default MsC)
--spectrum PATH             explicit spectrum file, overrides --potential/--alpha
--omega                     field angular frequency (default 1.0)
--amplitude-min/--amplitude-max/--n-amplitudes
                            amplitude scan (default linspace(0, 0.1, 21))
--total-time                physical duration of each run (default 2000)
--initial-state             eigenstate used as initial condition (default 0)
--energy-cutoff             keep states with E < cutoff (default 4.0)
--samples-per-period        integration steps per field period (default 35)
--save-every                keep one of every N integration steps (default 35)

OUTPUT

One JSON file containing the spectrum metadata, the field parameters, the
amplitude set, the final/mean/max ionization probabilities and the full
ionization history at the saved times.
"""

import argparse
import time

import numpy as np

try:  # executed as a script
    from _common import (
        DEFAULT_ENERGY_CUTOFF,
        POTENTIAL_NAMES,
        SAMPLES_PER_PERIOD,
        SAVE_EVERY,
        alpha_tag,
        build_time_grid,
        load_spectrum,
        make_output_path,
        potential_metadata,
        resolve_alpha,
        summarize_ionization,
        timestamp,
        write_json,
    )
except ImportError:  # imported as part of the scripts package
    from ._common import (
        DEFAULT_ENERGY_CUTOFF,
        POTENTIAL_NAMES,
        SAMPLES_PER_PERIOD,
        SAVE_EVERY,
        alpha_tag,
        build_time_grid,
        load_spectrum,
        make_output_path,
        potential_metadata,
        resolve_alpha,
        summarize_ionization,
        timestamp,
        write_json,
    )

from emerald.fields import SinusoidField
from emerald.quantum.evolution import SplitOperatorPropagator
from emerald.serialization import to_jsonable

'''
PARAMETERS
'''

field_amplitude_min = 0.0
field_amplitude_max = 0.1
n_amplitudes        = 21

field_frequency = 1.0

total_time = 2000.0

alpha = 0.001

initial_state = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum ionization probability versus field amplitude."
    )
    parser.add_argument(
        "--alpha", type=float, default=alpha,
        help=f"Potential parameter selecting the saved spectrum "
             f"(default: {alpha}; ignored for C).",
    )
    parser.add_argument(
        "--potential", choices=("MsC", "sC", "C"), default="MsC",
        help="Potential acronym selecting the saved spectrum "
             "(default: MsC; C is the alpha = 0 Coulomb case).",
    )
    parser.add_argument(
        "--spectrum", default=None,
        help="Explicit spectrum JSON file; overrides --potential/--alpha lookup.",
    )
    parser.add_argument(
        "--omega", type=float, default=field_frequency,
        help=f"Field angular frequency (default: {field_frequency}).",
    )
    parser.add_argument(
        "--amplitude-min", type=float, default=field_amplitude_min,
        help=f"Lowest field amplitude (default: {field_amplitude_min}).",
    )
    parser.add_argument(
        "--amplitude-max", type=float, default=field_amplitude_max,
        help=f"Highest field amplitude (default: {field_amplitude_max}).",
    )
    parser.add_argument(
        "--n-amplitudes", type=int, default=n_amplitudes,
        help=f"Number of amplitude points (default: {n_amplitudes}).",
    )
    parser.add_argument(
        "--total-time", type=float, default=total_time,
        help=f"Duration of each propagation (default: {total_time}).",
    )
    parser.add_argument(
        "--initial-state", type=int, default=initial_state,
        help=f"Initial eigenstate index (default: {initial_state}).",
    )
    parser.add_argument(
        "--energy-cutoff", type=float, default=DEFAULT_ENERGY_CUTOFF,
        help=f"Keep states with E < cutoff (default: {DEFAULT_ENERGY_CUTOFF}).",
    )
    parser.add_argument(
        "--full-spectrum", action="store_true",
        help="Keep every saved state, ignoring --energy-cutoff.",
    )
    parser.add_argument(
        "--samples-per-period", type=int, default=SAMPLES_PER_PERIOD,
        help=f"Integration steps per field period (default: {SAMPLES_PER_PERIOD}).",
    )
    parser.add_argument(
        "--save-every", type=int, default=SAVE_EVERY,
        help=f"Record one of every N integration steps (default: {SAVE_EVERY}).",
    )
    parser.add_argument(
        "--output-dir", default=".",
        help="Directory for the output JSON (default: current directory).",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable per-run progress bars.",
    )
    args = parser.parse_args()

    if args.omega <= 0.0:
        parser.error("--omega must be positive.")
    if args.total_time <= 0.0:
        parser.error("--total-time must be positive.")
    if args.n_amplitudes < 1:
        parser.error("--n-amplitudes must be at least 1.")
    if args.samples_per_period < 1 or args.save_every < 1:
        parser.error("--samples-per-period and --save-every must be positive.")

    # Coulomb has no alpha parameter; --potential C implies alpha = 0.
    args.alpha = resolve_alpha(args.potential, args.alpha)

    return args


def main():
    args = parse_args()

    energy_cutoff = None if args.full_spectrum else args.energy_cutoff
    amplitudes = np.linspace(
        args.amplitude_min, args.amplitude_max, args.n_amplitudes
    )

    spectrum, spectrum_file = load_spectrum(
        args.potential, args.alpha, energy_cutoff=energy_cutoff, path=args.spectrum
    )
    if not 0 <= args.initial_state < spectrum.n_states:
        raise SystemExit(
            f"--initial-state must be between 0 and {spectrum.n_states - 1}."
        )

    times, dt = build_time_grid(
        args.total_time,
        args.omega,
        samples_per_period=args.samples_per_period,
        save_every=args.save_every,
    )

    coeff0 = spectrum.state(args.initial_state)

    # The coupling matrix and its eigendecomposition are field-independent, so
    # the propagator is built once and only its ``field`` is swapped per point.
    propagator = SplitOperatorPropagator(
        spectrum, SinusoidField(float(amplitudes[0]), args.omega, phase=0)
    )

    histories = []
    generation_time = 0.0

    for index, amplitude in enumerate(amplitudes):
        propagator.field = SinusoidField(float(amplitude), args.omega, phase=0)

        start = time.perf_counter()
        evolution = propagator.propagate(
            coeff0,
            times=times,
            record_every=args.save_every,
            progress=not args.no_progress,
        )
        elapsed = time.perf_counter() - start
        generation_time += elapsed

        summary = summarize_ionization(evolution)
        histories.append(summary["history"])

        print(
            f"[{index + 1}/{len(amplitudes)}] "
            f"F0 = {amplitude:.4f}  "
            f"P_ion = {summary['final']:.6f}  "
            f"norm drift = {summary['norm_drift']:.2e}  "
            f"({elapsed:.1f} s)"
        )

    now = timestamp()
    filename = (
        f"{now}--{args.potential}-quantum-ionization-amplitude"
        f"--a--{alpha_tag(args.alpha)}--Omg--{round(args.omega, 3)}.json"
    )
    output_path = make_output_path(args.output_dir, filename)

    output_data = {
        "parameters": {
            "potential": POTENTIAL_NAMES[args.potential],
            "potential_parameters": potential_metadata(args.potential, args.alpha),
            "potential_acronym": args.potential,
            "alpha": args.alpha,
            "spectrum_file": str(spectrum_file),
            "n_states": spectrum.n_states,
            "n_bound": spectrum.n_bound,
            "energy_cutoff": energy_cutoff,
            "initial_state": args.initial_state,
            "initial_energy": float(spectrum.energies[args.initial_state]),
            "field_frequency": args.omega,
            "total_time": args.total_time,
            "samples_per_period": args.samples_per_period,
            "save_every": args.save_every,
            "dt": dt,
            "time_grid_length": int(len(times)),
        },
        "results": {
            "field_amplitude_set": to_jsonable(amplitudes),
            "ionization_probabilities_energy": [
                float(history[-1]) for history in histories
            ],
            "ionization_probabilities_final": [
                float(history[-1]) for history in histories
            ],
            "ionization_probabilities_mean": [
                float(np.mean(history)) for history in histories
            ],
            "ionization_probabilities_final_mean": [
                float(np.mean(history[-max(1, int(np.ceil(0.05 * len(history)))):]))
                for history in histories
            ],
            "ionization_probabilities_max": [
                float(np.max(history)) for history in histories
            ],
            "ionization_times": to_jsonable(times[::args.save_every]),
            "ionization_histories": histories,
            "generation_time_s": generation_time,
        },
    }

    write_json(output_path, output_data)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
