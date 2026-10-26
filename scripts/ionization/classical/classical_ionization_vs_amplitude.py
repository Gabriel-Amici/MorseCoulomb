'''
25/10/2026

Program to calculate the classical ionization probability vs field
amplitude values for an ensemble of particles under MsC, sC and Coulomb
potentials.

PARAMETERS

--alpha
field_amplitude_set
field_frequency
initial_energy
initial_conditions
total_time

OUTPUT

One JSON file per potential, containing the simulation parameters, the field
frequency, the field amplitude set and the ionization probability lists for
the energy and distance criteria.
'''

import argparse
import json
from datetime import datetime

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

field_amplitude_set = np.linspace(0, 0.1, 21)
field_frequency     = 1.0

total_time = 2000

initial_energy = -0.5

initial_conditions = 300

distance_criterion_r_max = 100.0

parser = argparse.ArgumentParser(
    description="Classical ionization probability versus field amplitude."
)
parser.add_argument(
    "--alpha",
    type=float,
    required=True,
    help="Soft-Coulomb / Morse-soft-Coulomb alpha parameter. "
         "Use 0 to run only the Coulomb potential.",
)
alpha = parser.parse_args().alpha

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

for amplitude in field_amplitude_set:

    field = SinusoidField(amplitude, field_frequency, phase=0)

    for acronym, potential, dynamics_class, integrator in potential_configs:

        dynamics = dynamics_class(potential, field)

        system = ClassicalSystem.from_angle(
            dynamics,
            E=initial_energy,
            N=initial_conditions,
            position_signs=1,
        )

        if results[acronym]["initial_conditions"] is None:
            results[acronym]["initial_conditions"] = len(system.states0)

        trajectory = system.propagate(t_eval, integrator=integrator)

        Pion_energy = trajectory.ionization_probability('energy')
        Pion_distance = trajectory.ionization_probabilities(
            'distance', r_max=distance_criterion_r_max
        )[-1]

        results[acronym]["ionization_probabilities_energy"].append(float(Pion_energy))
        results[acronym]["ionization_probabilities_distance"].append(float(Pion_distance))

    print(
        f"Amplitude {amplitude:.3f} Done: "
        + " | ".join(
            f"{acronym} = {results[acronym]['ionization_probabilities_energy'][-1]:.2f}"
            for acronym, _, _, _ in potential_configs
        )
    )

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
            "field_frequency": field_frequency,
            "total_time": total_time,
            "t_eval": to_jsonable(t_eval),
            "distance_criterion_r_max": distance_criterion_r_max,
        },
        "results": {
            "field_amplitude_set": to_jsonable(field_amplitude_set),
            "ionization_probabilities_energy":
                potential_data["ionization_probabilities_energy"],
            "ionization_probabilities_distance":
                potential_data["ionization_probabilities_distance"],
        },
    }

    filename = (
        f"{now}--{acronym}-classical-ionization-amplitude"
        f"--a--{round(alpha, 3):g}--Omg--{round(field_frequency, 3)}.json"
    )

    with open(filename, "w") as filehandle:
        json.dump(output_data, filehandle, indent=2)
