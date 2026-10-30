'''
25/10/2026

Program to calculate the classical ionization probability vs alpha values
for an ensemble of particles under MsC, sC and Coulomb potentials.

PARAMETERS

alpha_set
field_amplitude
field_frequency
initial_energy
initial_conditions
total_time

OUTPUT

One JSON file per potential, containing the simulation parameters, the field
parameters, the alpha set and the ionization probability lists for the energy
and distance criteria.
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

alpha_set = np.linspace(0, 1.5, 21)

field_amplitude = 0.04
field_frequency = 1.0

total_time = 2000

initial_energy = -0.5

initial_conditions = 300

distance_criterion_r_max = 100.0

DEFAULT_WORKERS = 10


def simulate_alpha(alpha, potential_configs, field_amplitude, field_frequency,
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


def potential_configs_for_alpha(alpha):
    if alpha == 0:
        return [
            ("C", Coulomb(), RegularizedDynamics, _DEFAULT_REGULARIZED_INTEGRATOR),
        ]

    return [
        ("MsC", MorseSoftCoulomb(alpha=alpha), Dynamics, _DEFAULT_INTEGRATOR),
        # ("sC", SoftCoulomb(alpha=alpha), Dynamics, _DEFAULT_INTEGRATOR),
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Classical ionization probability versus alpha."
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

    results = {}

    t_eval = np.linspace(0, total_time, 10)

    tasks = []

    for alpha in alpha_set:

        alpha = float(alpha)

        if alpha != 0 and alpha >= -1 / initial_energy:
            print(
                f"Skipping alpha = {alpha:g}: "
                f"no bound orbit at E = {initial_energy:g}."
            )
            continue

        potential_configs = potential_configs_for_alpha(alpha)

        for acronym, potential, _, _ in potential_configs:
            if acronym not in results:
                results[acronym] = {
                    "potential": potential,
                    "alpha_set": [],
                    "initial_conditions": None,
                    "ionization_probabilities_energy": [],
                    "ionization_probabilities_distance": [],
                }

        tasks.append((alpha, potential_configs))

    ordered = [None] * len(tasks)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                simulate_alpha,
                alpha,
                potential_configs,
                field_amplitude,
                field_frequency,
                t_eval,
                initial_energy,
                initial_conditions,
                distance_criterion_r_max,
            ): i
            for i, (alpha, potential_configs) in enumerate(tasks)
        }

        for future in as_completed(futures):
            i = futures[future]
            point = future.result()
            ordered[i] = point

            alpha, potential_configs = tasks[i]
            print(
                f"Alpha {alpha:.3f} Done: "
                + " | ".join(
                    f"{acronym} = {point[acronym]['energy']:.2f}"
                    for acronym, _, _, _ in potential_configs
                )
            )

    for i, (alpha, potential_configs) in enumerate(tasks):

        point = ordered[i]

        for acronym, _, _, _ in potential_configs:

            potential_data = results[acronym]

            if potential_data["initial_conditions"] is None:
                potential_data["initial_conditions"] = point[acronym]["n_initial"]

            potential_data["alpha_set"].append(alpha)
            potential_data["ionization_probabilities_energy"].append(
                point[acronym]["energy"]
            )
            potential_data["ionization_probabilities_distance"].append(
                point[acronym]["distance"]
            )

    now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

    for acronym, potential_data in results.items():

        potential = potential_data["potential"]

        potential_parameters = to_jsonable(potential.to_dict())
        potential_parameters.pop("alpha", None)

        output_data = {
            "parameters": {
                "potential": potential.name,
                "potential_parameters": potential_parameters,
                "initial_energy": initial_energy,
                "initial_conditions": potential_data["initial_conditions"],
                "field_amplitude": field_amplitude,
                "field_frequency": field_frequency,
                "total_time": total_time,
                "t_eval": to_jsonable(t_eval),
                "distance_criterion_r_max": distance_criterion_r_max,
            },
            "results": {
                "alpha_set": potential_data["alpha_set"],
                "ionization_probabilities_energy":
                    potential_data["ionization_probabilities_energy"],
                "ionization_probabilities_distance":
                    potential_data["ionization_probabilities_distance"],
            },
        }

        alpha_values = potential_data["alpha_set"]
        alpha_tag = f"{round(alpha_values[0], 3):g}"
        if len(alpha_values) > 1:
            alpha_tag += f"-{round(alpha_values[-1], 3):g}"

        filename = (
            f"{now}--{acronym}-classical-ionization-alpha"
            f"--a--{alpha_tag}"
            f"--F0--{round(field_amplitude, 3)}"
            f"--Omg--{round(field_frequency, 3)}.json"
        )

        with open(filename, "w") as filehandle:
            json.dump(output_data, filehandle, indent=2)


if __name__ == "__main__":
    main()
