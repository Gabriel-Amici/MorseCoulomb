'''
13/04/2025

Program to calculate the the classical ionization probability for various field frequency values for an ensamble of particles under the Morse-soft-Coulomb.

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

from emerald.classical.msc_ionization import MsC_ionization_frequency_criteria

'''
PARAMETERS
'''

field_amplitude = 0.005
field_frequency_set = np.arange( 0.01, .52, 0.02 )
delta_t = 0.001
total_time = 100

alpha = 0.1
initial_energy = -0.186280312943416

initial_conditions = 150

ionization_probabilities_8rM = []
ionization_probabilities_4rM = []
ionization_probabilities_comp = []



Pis = MsC_ionization_frequency_criteria(alpha, initial_energy, field_amplitude, field_frequency_set, initial_conditions, total_time, 80, delta_t)

Pis = np.array(Pis)

ionization_probabilities_8rM = Pis[:, 0]
ionization_probabilities_4rM = Pis[:, 1]
ionization_probabilities_comp = Pis[:, 2]

output_data  = {
    "parameters" : {
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
        "ionization_probabilities_8rM" : list(ionization_probabilities_8rM),
        "ionization_probabilities_4rM" : list(ionization_probabilities_4rM),
        "ionization_probabilities_comp" : list(ionization_probabilities_comp),
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

'''
import matplotlib.pyplot as plt
plt.ylim(0, 1)
plt.plot(output_data["parameters"]["field_frequency_set"], output_data["results"]["ionization_probabilities_8rM"], label="$r>8r_M$")
plt.plot(output_data["parameters"]["field_frequency_set"], output_data["results"]["ionization_probabilities_4rM"], label="$r>4r_M$")
plt.plot(output_data["parameters"]["field_frequency_set"], output_data["results"]["ionization_probabilities_comp"], label="compensate")
plt.legend()
plt.show()
'''

filename = f"{now}--classical-ionization-frequency--a--{alpha}--F0--{field_amplitude}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)

