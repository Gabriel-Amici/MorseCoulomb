'''
25/10/2026

Program to calculate the classical ionization probability vs field
frequency values for an ensemble of particles under MsC, sC and Coulomb
potentials.

PARAMETERS

--alpha
field_amplitude
field_frequency_set
initial_energy
initial_conditions
total_time

OUTPUT

One JSON file per potential, containing the simulation parameters, the field
amplitude, the field frequency set and the ionization probability lists for
the energy and distance criteria.
'''

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from emerald.classical.dynamics import (
    _DEFAULT_INTEGRATOR,
    _DEFAULT_REGULARIZED_INTEGRATOR,
    ClassicalSystem,
    Dynamics,
    RegularizedDynamics,
)
from emerald.fields import SinusoidField
from emerald.potentials import Coulomb, MorseSoftCoulomb, SoftCoulomb
from emerald.serialization import to_jsonable

'''
PARAMETERS
'''

field_amplitude     = 0.04
field_frequency_set = np.linspace(0.5, 2, 21)

total_time = 2000

initial_energy = -0.5

initial_conditions = 300

distance_criterion_r_max = 100.0

DEFAULT_WORKERS = 10


def simulate_frequency(field_frequency, potential_configs, field_amplitude,
                       t_eval, initial_energy, initial_conditions,
                       distance_criterion_r_max):
    field = SinusoidField(field_amplitude, field_frequency, phase=0)
    point = {}

    for acronym, potential, dynamics_class, integrator in potential_configs:

        dynamics = dynamics_class(potential, field)

        system = ClassicalSystem.from_angle(
            dynamics,
            E=initial_energy,
            N=initial_conditions,
            position_signs=1,
        )

        trajectory = system.propagate(t_eval, integrator=integrator)

        point[acronym] = {
            "n_initial": len(system.states0),
            "energy": float(trajectory.ionization_probability('energy')),
            "distance": float(
                trajectory.ionization_probabilities(
                    'distance', r_max=distance_criterion_r_max
                )[-1]
            ),
        }

    return point


def main():
    parser = argparse.ArgumentParser(
        description="Classical ionization probability versus field frequency."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Soft-Coulomb / Morse-soft-Coulomb alpha parameter. "
             "Use 0 to run only the Coulomb potential.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of worker processes (default: {DEFAULT_WORKERS}).",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least 1.")

    workers = min(args.workers, os.cpu_count() or 1)
    alpha = args.alpha

    if alpha == 0:
        potential_configs = [
            ("C", Coulomb(), RegularizedDynamics, _DEFAULT_REGULARIZED_INTEGRATOR),
        ]
    else:
        potential_configs = [
            ("MsC", MorseSoftCoulomb(alpha=alpha), Dynamics, _DEFAULT_INTEGRATOR),
            # ("sC", SoftCoulomb(alpha=alpha), Dynamics, _DEFAULT_INTEGRATOR),
        ]

    results = {
        acronym: {
            "potential": potential,
            "initial_conditions": None,
            "ionization_probabilities_energy": [],
            "ionization_probabilities_distance": [],
        }
        for acronym, potential, _, _ in potential_configs
    }

    t_eval = np.linspace(0, total_time, 10)
    ordered = [None] * len(field_frequency_set)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                simulate_frequency,
                float(field_frequency),
                potential_configs,
                field_amplitude,
                t_eval,
                initial_energy,
                initial_conditions,
                distance_criterion_r_max,
            ): i
            for i, field_frequency in enumerate(field_frequency_set)
        }

        for future in as_completed(futures):
            i = futures[future]
            point = future.result()
            ordered[i] = point

            field_frequency = field_frequency_set[i]
            print(
                f"Frequency {field_frequency:.3f} Done: "
                + " | ".join(
                    f"{acronym} = {point[acronym]['energy']:.2f}"
                    for acronym, _, _, _ in potential_configs
                )
            )

    for acronym, potential, _, _ in potential_configs:

        potential_data = results[acronym]

        potential_data["initial_conditions"] = next(
            point[acronym]["n_initial"] for point in ordered
        )
        potential_data["ionization_probabilities_energy"] = [
            point[acronym]["energy"] for point in ordered
        ]
        potential_data["ionization_probabilities_distance"] = [
            point[acronym]["distance"] for point in ordered
        ]

    now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

    for acronym, potential, _, _ in potential_configs:

        potential_data = results[acronym]

        output_data = {
            "parameters": {
                "potential": potential.name,
                "potential_parameters": to_jsonable(potential.to_dict()),
                "alpha": alpha,
                "initial_energy": initial_energy,
                "initial_conditions": potential_data["initial_conditions"],
                "field_amplitude": field_amplitude,
                "total_time": total_time,
                "t_eval": to_jsonable(t_eval),
                "distance_criterion_r_max": distance_criterion_r_max,
            },
            "results": {
                "field_frequency_set": to_jsonable(field_frequency_set),
                "ionization_probabilities_energy":
                    potential_data["ionization_probabilities_energy"],
                "ionization_probabilities_distance":
                    potential_data["ionization_probabilities_distance"],
            },
        }

        filename = (
            f"{now}--{acronym}-classical-ionization-frequency"
            f"--a--{round(alpha, 3):g}--F0--{round(field_amplitude, 3)}.json"
        )

        with open(filename, "w") as filehandle:
            json.dump(output_data, filehandle, indent=2)


if __name__ == "__main__":
    main()
