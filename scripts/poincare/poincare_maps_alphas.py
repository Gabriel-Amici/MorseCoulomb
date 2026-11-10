'''
03/11/2026

Program to generate Poincare maps vs alpha values
for an ensemble of particles under MsC, ~sC~ and Coulomb potentials.

PARAMETERS

alpha_set
field_amplitude
field_frequency
initial_energy_set
initial_conditions
total_time

OUTPUT

One JSON file per potential, containing the simulation parameters, the field
parameters, the alpha set and the poincaré map that potential
'''

import json
from datetime import datetime

import numpy as np

from emerald.classical.dynamics import (
    _DEFAULT_INTEGRATOR,
    _DEFAULT_REGULARIZED_INTEGRATOR,
    ClassicalSystem,
    Dynamics,
    PoincareMap,
    RegularizedDynamics,
)
from emerald.fields import SinusoidField
from emerald.potentials import Coulomb, MorseSoftCoulomb, SoftCoulomb
from emerald.serialization import to_jsonable

'''
PARAMETERS
'''

alpha_set = [0.01, 0.5, 1.]

field_amplitude = 0.025
field_frequency = 1.0

total_periods = 300

initial_energy_set = np.linspace(-0.6, -0.4, 31)

initial_conditions = 40



def simulate_alpha(alpha, potential_configs, field_amplitude, field_frequency,
                   total_periods, initial_energy_set, initial_conditions):
    
    field = SinusoidField(field_amplitude, field_frequency, phase=0)
    point = {}

    for acronym, potential, dynamics_class, integrator in potential_configs:

        dynamics = dynamics_class(potential, field)

        subsystems = [ClassicalSystem.from_angle(
            dynamics,
            E=_E,
            N=initial_conditions,
            position_signs=1,
        ) for _E in initial_energy_set]

        system = ClassicalSystem.merge_systems(dynamics, subsystems)

        pmap = system.poincare_map(total_periods, integrator=integrator)

    return pmap.to_dict()


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
    for alpha in alpha_set:

        alpha = float(alpha)

        potential_configs = potential_configs_for_alpha(alpha)

        pmap_dict = simulate_alpha(alpha,
                                   potential_configs,
                                   field_amplitude,
                                   field_frequency,
                                   total_periods,
                                   initial_energy_set,
                                   initial_conditions,
                                   )


        
        print(
            f"Alpha {alpha:.3f} Done: elapsed time: {pmap_dict['metadata']['generation_time_s']/60:.2f} minutes"
        )

        now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

        filename = (
            f"{now}--{potential_configs[0][0]}-Poincare-map-alpha"
            f"--a--{alpha:.3f}"
            f"--F0--{round(field_amplitude, 3)}"
            f"--Omg--{round(field_frequency, 3)}.json"
        )

        with open(filename, "w") as filehandle:
            json.dump(pmap_dict, filehandle)


if __name__ == "__main__":
    main()
