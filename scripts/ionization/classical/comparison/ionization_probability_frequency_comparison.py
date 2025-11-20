'''
03/07/2025

Program to calculate the the classical ionization probability for various field frequency values for an ensamble of particles under various potentials.

PARAMETERS

alpha
field_amplitude
field_frequency_set
initial_energy
initial_conditions
delta_t
total_time

OUTPUT

data -- object containing simulation parameters and results (history of expansion coefficients, ionization probability)
'''

import numpy as np
import json
from datetime import datetime


'''
PARAMETERS
'''

field_amplitude = 0.05
field_frequency_set = np.arange( 0.2, 1.55, 0.1 )
delta_t = 1.e-4
total_time = 2000

alpha = 0.5
initial_energy = -0.5

initial_conditions = 150

'''MsC potential'''

from emerald.classical.msc_ionization import MsC_ionization_frequency

for alpha in [0.2, np.sqrt(2)]:

    print(f"Calculating MsC ionization frequency for alpha = {alpha}, field amplitude = {field_amplitude}")
    
    Pis = MsC_ionization_frequency(alpha, initial_energy, field_amplitude, field_frequency_set, initial_conditions, total_time, 80, delta_t)

    ionization_probabilities = np.array(Pis)


    output_data  = {
        "parameters" : {
            "potential": "Morse-soft-Coulomb",
            "alpha" : alpha,
            "initial_energy" : initial_energy,
            "initial_conditions" : int( (initial_conditions-1)*2 ),
            "field_amplitude" : field_amplitude,
            "field_frequency_set" : list(field_frequency_set),
            "time_grid": {
                "time_grid_starts" : list(np.pi/(2*field_frequency_set)),
                "total_time" : total_time,
                "delta_t" : delta_t
            }
        },
        "results" : {
            "ionization_probabilities" : list(ionization_probabilities)
        }
    }

    now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")


    filename = f"{now}--MsC-ionization-frequency--a--{round(alpha, 3)}--F0--{field_amplitude}.json"
    with open(filename, "w") as filehandle:
        json.dump(output_data, filehandle)


'''sC potential'''

from emerald.classical.sc_ionization import sC_ionization_frequency

for alpha in [0.2, np.sqrt(2)]:

    print(f"Calculating sC ionization frequency for alpha = {alpha}, field amplitude = {field_amplitude}")
    Pis = sC_ionization_frequency(alpha, initial_energy, field_amplitude, field_frequency_set, initial_conditions, total_time, 80, delta_t)

    ionization_probabilities = np.array(Pis)


    output_data  = {
        "parameters" : {
            "potential": "soft-Coulomb",
            "alpha" : alpha,
            "initial_energy" : initial_energy,
            "initial_conditions" : int( (initial_conditions-1)*2 ),
            "field_amplitude" : field_amplitude,
            "field_frequency_set" : list(field_frequency_set),
            "time_grid": {
                "time_grid_starts" : list(np.pi/(2*field_frequency_set)),
                "total_time" : total_time,
                "delta_t" : delta_t
            }
        },
        "results" : {
            "ionization_probabilities" : list(ionization_probabilities)
        }
    }

    now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")


    filename = f"{now}--sC-ionization-frequency--a--{round(alpha, 3)}--F0--{field_amplitude}.json"
    with open(filename, "w") as filehandle:
        json.dump(output_data, filehandle)


'''Coulomb potential'''

from emerald.classical.coulomb_ionization import C_ionization_frequency

Pis = C_ionization_frequency(initial_energy, field_amplitude, field_frequency_set, initial_conditions, total_time, delta_t)

ionization_probabilities = np.array(Pis)


output_data  = {
    "parameters" : {
        "potential": "Coulomb",
        "alpha" : 0,
        "initial_energy" : initial_energy,
        "initial_conditions" : int( (initial_conditions-1)*2 ),
        "field_amplitude" : field_amplitude,
        "field_frequency_set" : list(field_frequency_set),
        "time_grid": {
            "time_grid_starts" : list(np.pi/(2*field_frequency_set)),
            "total_time" : total_time,
            "delta_t" : delta_t
        }
    },
    "results" : {
        "ionization_probabilities" : list(ionization_probabilities)
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")


filename = f"{now}--C-ionization-frequency--a--{0}--F0--{field_amplitude}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)
