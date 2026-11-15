"""
Quantum counterpart of ``classical_ionization_vs_frequency.py``.

Propagates the ground state of a saved MsC (or sC) spectrum under a sinusoidal
field of fixed amplitude and increasing angular frequency,

    H(t) = H0 + F(t) r,      F(t) = F0 sin(omega t),

and reports the population of the pseudo-continuum states (``E >= 0``) at the
end of each run.

Because the time step is defined as ``T/--samples-per-period`` and ``T`` depends
on ``omega``, every frequency is propagated on its own uniform time grid.  The
propagator (field-independent coupling matrix and its eigendecomposition) is
built only once per spectrum and reused by swapping its field.

The sC potential is intentionally left out of the default frequency scan; add
``--potential sC`` if it is needed.  sC is mainly used by the stabilization
script.  ``--potential C`` selects the Coulomb spectrum (the alpha = 0 case).

PARAMETERS

--alpha                     potential parameter (default 0.001; ignored for C)
--potential {MsC,sC,C}      which saved spectrum to use (default MsC)
--spectrum PATH             explicit spectrum file, overrides --potential/--alpha
--field-amplitude           fixed field amplitude (default 0.04)
--frequency-min/--frequency-max/--n-frequencies
                            angular-frequency scan (default linspace(0.5, 2, 21))
--total-time                physical duration of each run (default 2000)
--initial-state             eigenstate used as initial condition (default 0)
--energy-cutoff             keep states with E < cutoff (default 4.0)
--samples-per-period        integration steps per field period (default 35)
--save-every                keep one of every N integration steps (default 35)

OUTPUT

One JSON file containing the spectrum metadata, the field amplitude, the
frequency set, the final/mean/max ionization probabilities, the per-frequency
recorded times and the full ionization histories.
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

field_frequency_min = 0.5
field_frequency_max = 2.0
n_frequencies       = 21

field_amplitude = 0.04

total_time = 2000.0

alpha = 0.001

initial_state = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum ionization probability versus field frequency."
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
        "--field-amplitude", type=float, default=field_amplitude,
        help=f"Field amplitude (default: {field_amplitude}).",
    )
    parser.add_argument(
        "--frequency-min", type=float, default=field_frequency_min,
        help=f"Lowest angular frequency (default: {field_frequency_min}).",
    )
    parser.add_argument(
        "--frequency-max", type=float, default=field_frequency_max,
        help=f"Highest angular frequency (default: {field_frequency_max}).",
    )
    parser.add_argument(
        "--n-frequencies", type=int, default=n_frequencies,
        help=f"Number of frequency points (default: {n_frequencies}).",
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

    if args.field_amplitude < 0.0:
        parser.error("--field-amplitude must be non-negative.")
    if args.frequency_min <= 0.0 or args.frequency_max <= 0.0:
        parser.error("--frequency-min and --frequency-max must be positive.")
    if args.total_time <= 0.0:
        parser.error("--total-time must be positive.")
    if args.n_frequencies < 1:
        parser.error("--n-frequencies must be at least 1.")
    if args.samples_per_period < 1 or args.save_every < 1:
        parser.error("--samples-per-period and --save-every must be positive.")

    # Coulomb has no alpha parameter; --potential C implies alpha = 0.
    args.alpha = resolve_alpha(args.potential, args.alpha)

    return args


def main():
    args = parse_args()

    energy_cutoff = None if args.full_spectrum else args.energy_cutoff
    frequencies = np.linspace(
        args.frequency_min, args.frequency_max, args.n_frequencies
    )

    spectrum, spectrum_file = load_spectrum(
        args.potential, args.alpha, energy_cutoff=energy_cutoff, path=args.spectrum
    )
    if not 0 <= args.initial_state < spectrum.n_states:
        raise SystemExit(
            f"--initial-state must be between 0 and {spectrum.n_states - 1}."
        )

    coeff0 = spectrum.state(args.initial_state)

    # Built once: the coupling matrix does not depend on the field, only the
    # time grid (set by omega) and the field itself change between points.
    propagator = SplitOperatorPropagator(
        spectrum,
        SinusoidField(args.field_amplitude, float(frequencies[0]), phase=0),
    )

    histories = []
    recorded_times = []
    steps = []
    generation_time = 0.0

    for index, omega in enumerate(frequencies):
        times, dt = build_time_grid(
            args.total_time,
            float(omega),
            samples_per_period=args.samples_per_period,
            save_every=args.save_every,
        )
        propagator.field = SinusoidField(
            args.field_amplitude, float(omega), phase=0
        )

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
        recorded_times.append(summary["times"])
        steps.append(int(len(times) - 1))

        print(
            f"[{index + 1}/{len(frequencies)}] "
            f"omega = {omega:.4f}  "
            f"P_ion = {summary['final']:.6f}  "
            f"dt = {dt:.4f}  "
            f"norm drift = {summary['norm_drift']:.2e}  "
            f"({elapsed:.1f} s)"
        )

    now = timestamp()
    filename = (
        f"{now}--{args.potential}-quantum-ionization-frequency"
        f"--a--{alpha_tag(args.alpha)}"
        f"--F0--{round(args.field_amplitude, 3)}.json"
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
            "field_amplitude": args.field_amplitude,
            "total_time": args.total_time,
            "samples_per_period": args.samples_per_period,
            "save_every": args.save_every,
        },
        "results": {
            "field_frequency_set": to_jsonable(frequencies),
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
            "ionization_times": recorded_times,
            "ionization_histories": histories,
            "time_steps": steps,
            "generation_time_s": generation_time,
        },
    }

    write_json(output_path, output_data)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
