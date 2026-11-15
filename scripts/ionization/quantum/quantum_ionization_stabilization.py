"""
Quantum counterpart of ``classical_ionization_stabilization.py``.

Propagates the ground state of a saved MsC or sC spectrum under a trapezoidal
pulse,

    H(t) = H0 + F(t) r,
    F(t) = A sin(omega t) * envelope(t),

where the envelope rises linearly for ``--ramp-up`` carrier periods, stays at
unit amplitude for ``--plateau`` periods and falls for ``--ramp-down`` periods.
The population of the pseudo-continuum states (``E >= 0``) is recorded at the
end of each pulse, after the field has been switched off.

Unlike the amplitude and frequency scans, this script evaluates both the MsC
and the sC spectra by default (``--potential both``), since the comparison
between the two is the point of the stabilization study.  ``--potential C``
runs the alpha = 0 Coulomb case; ``--potential both`` never includes it.

PARAMETERS

--alpha                     potential parameter (default 1.0; ignored for C)
--potential {MsC,sC,C,both} spectra to propagate (default both = MsC + sC)
--spectrum PATH             explicit spectrum file (requires a single potential)
--omega                     carrier angular frequency (default 0.8)
--amplitude-min/--amplitude-max/--n-amplitudes
                            amplitude scan (default linspace(0, 1, 21))
--ramp-up/--plateau/--ramp-down
                            envelope segment lengths in carrier periods
                            (defaults 6/44/1, i.e. 51 periods in total)
--initial-state             eigenstate used as initial condition (default 0)
--energy-cutoff             keep states with E < cutoff (default 4.0)
--samples-per-period        integration steps per field period (default 35)
--save-every                keep one of every N integration steps (default 35)

OUTPUT

One JSON file per potential, containing the spectrum metadata, the envelope
parameters, the amplitude set, the final/mean/max ionization probabilities and
the full ionization history at the saved times.
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

from emerald.fields import PlateauEnvelope, SinusoidField
from emerald.quantum.evolution import SplitOperatorPropagator
from emerald.serialization import to_jsonable

'''
PARAMETERS
'''

field_amplitude_min = 0.0
field_amplitude_max = 1.0
n_amplitudes        = 21

field_frequency = 0.8

ramp_up   = 6
plateau   = 44
ramp_down = 1

alpha = 1.0

initial_state = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum ionization stabilization versus field amplitude."
    )
    parser.add_argument(
        "--alpha", type=float, default=alpha,
        help=f"Potential parameter selecting the saved spectra "
             f"(default: {alpha}; ignored for C).",
    )
    parser.add_argument(
        "--potential", choices=("MsC", "sC", "C", "both"), default="both",
        help="Potential spectra to propagate (default: both = MsC + sC).",
    )
    parser.add_argument(
        "--spectrum", default=None,
        help="Explicit spectrum JSON file; requires a single --potential.",
    )
    parser.add_argument(
        "--omega", type=float, default=field_frequency,
        help=f"Carrier angular frequency (default: {field_frequency}).",
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
        "--ramp-up", type=float, default=ramp_up,
        help=f"Rising-ramp length in carrier periods (default: {ramp_up}).",
    )
    parser.add_argument(
        "--plateau", type=float, default=plateau,
        help=f"Plateau length in carrier periods (default: {plateau}).",
    )
    parser.add_argument(
        "--ramp-down", type=float, default=ramp_down,
        help=f"Falling-ramp length in carrier periods (default: {ramp_down}).",
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
    if args.n_amplitudes < 1:
        parser.error("--n-amplitudes must be at least 1.")
    if args.ramp_up < 0.0 or args.plateau < 0.0 or args.ramp_down < 0.0:
        parser.error("--ramp-up, --plateau and --ramp-down must be non-negative.")
    if args.ramp_up + args.plateau + args.ramp_down <= 0.0:
        parser.error("The envelope duration must be positive.")
    if args.samples_per_period < 1 or args.save_every < 1:
        parser.error("--samples-per-period and --save-every must be positive.")
    if args.spectrum is not None and args.potential == "both":
        parser.error("--spectrum requires a single --potential (MsC, sC or C).")

    # Coulomb has no alpha parameter; --potential C implies alpha = 0.
    args.alpha = resolve_alpha(args.potential, args.alpha)

    return args


def main():
    args = parse_args()

    energy_cutoff = None if args.full_spectrum else args.energy_cutoff
    amplitudes = np.linspace(
        args.amplitude_min, args.amplitude_max, args.n_amplitudes
    )

    acronyms = ("MsC", "sC") if args.potential == "both" else (args.potential,)

    period = 2.0 * np.pi / args.omega
    total_time = (args.ramp_up + args.plateau + args.ramp_down) * period

    times, dt = build_time_grid(
        total_time,
        args.omega,
        samples_per_period=args.samples_per_period,
        save_every=args.save_every,
    )

    now = timestamp()

    for acronym in acronyms:
        spectrum, spectrum_file = load_spectrum(
            acronym, args.alpha, energy_cutoff=energy_cutoff, path=args.spectrum
        )
        if not 0 <= args.initial_state < spectrum.n_states:
            raise SystemExit(
                f"--initial-state must be between 0 and {spectrum.n_states - 1}."
            )

        coeff0 = spectrum.state(args.initial_state)

        # Built once per spectrum; only the field is replaced per amplitude.
        first_carrier = SinusoidField(float(amplitudes[0]), args.omega, phase=0)
        propagator = SplitOperatorPropagator(
            spectrum,
            PlateauEnvelope(first_carrier, args.ramp_up, args.plateau, args.ramp_down),
        )

        histories = []
        generation_time = 0.0

        print(f"\n=== {acronym} (alpha = {args.alpha:g}) ===")

        for index, amplitude in enumerate(amplitudes):
            carrier = SinusoidField(float(amplitude), args.omega, phase=0)
            propagator.field = PlateauEnvelope(
                carrier, args.ramp_up, args.plateau, args.ramp_down
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

            print(
                f"[{index + 1}/{len(amplitudes)}] "
                f"A = {amplitude:.4f}  "
                f"P_ion = {summary['final']:.6f}  "
                f"norm drift = {summary['norm_drift']:.2e}  "
                f"({elapsed:.1f} s)"
            )

        filename = (
            f"{now}--{acronym}-quantum-ionization-stabilization"
            f"--a--{alpha_tag(args.alpha)}--Omg--{round(args.omega, 3)}.json"
        )
        output_path = make_output_path(args.output_dir, filename)

        output_data = {
            "parameters": {
                "potential": POTENTIAL_NAMES[acronym],
                "potential_parameters": potential_metadata(acronym, args.alpha),
                "potential_acronym": acronym,
                "alpha": args.alpha,
                "spectrum_file": str(spectrum_file),
                "n_states": spectrum.n_states,
                "n_bound": spectrum.n_bound,
                "energy_cutoff": energy_cutoff,
                "initial_state": args.initial_state,
                "initial_energy": float(spectrum.energies[args.initial_state]),
                "field_frequency": args.omega,
                "envelope_shape": "linear",
                "ramp_up": args.ramp_up,
                "plateau": args.plateau,
                "ramp_down": args.ramp_down,
                "total_periods": args.ramp_up + args.plateau + args.ramp_down,
                "total_time": total_time,
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
                    float(
                        np.mean(history[-max(1, int(np.ceil(0.05 * len(history)))):])
                    )
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
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
