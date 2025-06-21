'''
12/04/2025

Program to calculate the the classical ionization probability for various field amplitude values for an ensamble of particles under the Morse-soft-Coulomb.

PARAMETERS

alpha
field_amplitude_set
field_frequency
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

from emerald.classical.msc_ionization import MsC_ionization_amplitude_criteria

'''
PARAMETERS
'''

field_amplitude_set = np.arange(0, 0.075, 0.005)/2#np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])/5#np.arange(0, 0.11, 0.01)
field_frequency = 0.25
field_period = 2*np.pi/field_frequency

delta_t = 0.001
total_time = 50*field_period
time_grid_start = np.pi/(2*field_frequency)

alpha = 0.1
initial_energy = -0.186280312943416

initial_conditions = 150

ionization_probabilities_8rM = []
ionization_probabilities_4rM = []
ionization_probabilities_comp = []



Pis = MsC_ionization_amplitude_criteria(alpha, initial_energy, field_amplitude_set, field_frequency, initial_conditions, time_grid_start, total_time, 100, delta_t)

Pis = np.array(Pis)

ionization_probabilities_8rM = Pis[:, 0]
ionization_probabilities_4rM = Pis[:, 1]
ionization_probabilities_comp = Pis[:, 2]

output_data  = {
    "parameters" : {
        "alpha" : alpha,
        "initial_energy" : initial_energy,
        "initial_conditions" : int( (initial_conditions-2)*2 ),
        "field_amplitude_set" : list(field_amplitude_set),
        "field_frequency" : field_frequency,
        "time_grid": {
            "time_grid_start" : time_grid_start,
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

import matplotlib.pyplot as plt
plt.ylim(0, 1)
plt.plot(output_data["parameters"]["field_amplitude_set"], output_data["results"]["ionization_probabilities_8rM"], label="$r>8r_M$")
plt.plot(output_data["parameters"]["field_amplitude_set"], output_data["results"]["ionization_probabilities_4rM"], label="$r>4r_M$")
plt.plot(output_data["parameters"]["field_amplitude_set"], output_data["results"]["ionization_probabilities_comp"], label="compensate")
plt.legend()
plt.show()

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--classical-ionization-amplitude--a--{alpha}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)

