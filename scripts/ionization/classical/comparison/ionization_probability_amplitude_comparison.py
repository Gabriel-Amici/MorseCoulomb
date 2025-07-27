'''
03/07/2025

Program to calculate the the classical ionization probability for various field amplitude values for an ensamble of particles under various potentials.

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


'''
PARAMETERS
'''

field_amplitude_set = np.arange(0, 0.11, 0.01)
field_frequency = 1
field_period = 2*np.pi/field_frequency

delta_t = 1.e-4
total_time = 2000*field_period
time_grid_start = np.pi/(2*field_frequency)

alpha = 0.1
initial_energy = -0.5

initial_conditions = 150

'''MsC potential'''

from emerald.classical.msc_ionization import MsC_ionization_amplitude

ionization_probabilities = []
Pis = MsC_ionization_amplitude(alpha, initial_energy, field_amplitude_set, field_frequency, initial_conditions, time_grid_start, total_time, 100, delta_t)

ionization_probabilities = np.array(Pis)


output_data  = {
    "parameters" : {
        "potential": "Morse-soft-Coulomb",
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
        "ionization_probabilities" : list(ionization_probabilities)
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--MsC-classical-ionization-amplitude--a--{alpha}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)


'''sC potential'''

from emerald.classical.sc_ionization import sC_ionization_amplitude

ionization_probabilities = []
Pis = sC_ionization_amplitude(alpha, initial_energy, field_amplitude_set, field_frequency, initial_conditions, time_grid_start, total_time, 100, delta_t)

ionization_probabilities = np.array(Pis)


output_data  = {
    "parameters" : {
        "potential": "soft-Coulomb",
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
        "ionization_probabilities" : list(ionization_probabilities)
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--sC-classical-ionization-amplitude--a--{alpha}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)


'''normalized MsC potential'''

from emerald.classical.normalized_msc_ionization import nMsC_ionization_amplitude

ionization_probabilities = []
Pis = nMsC_ionization_amplitude(alpha, initial_energy, field_amplitude_set, field_frequency, initial_conditions, time_grid_start, total_time, 100, delta_t)

ionization_probabilities = np.array(Pis)


output_data  = {
    "parameters" : {
        "potential": "normalized Morse-soft-Coulomb",
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
        "ionization_probabilities" : list(ionization_probabilities)
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--nMsC-classical-ionization-amplitude--a--{alpha}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)


'''Coulomb potential'''

from emerald.classical.coulomb_ionization import C_ionization_amplitude

ionization_probabilities = []
Pis = C_ionization_amplitude(initial_energy, field_amplitude_set, field_frequency, initial_conditions, time_grid_start, total_time, delta_t)

ionization_probabilities = np.array(Pis)


output_data  = {
    "parameters" : {
        "potential": "Coulomb",
        "alpha" : 0,
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
        "ionization_probabilities" : list(ionization_probabilities)
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--C-classical-ionization-amplitude--a--{0}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)



