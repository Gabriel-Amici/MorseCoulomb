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

field_amplitude_set = np.arange(0, 0.11, 0.01) #[0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ]

initial_energy = -0.5

initial_conditions = 150

'''MsC potential'''

from emerald.classical.msc_ionization import MsC_ionization_amplitude
from emerald.classical.msc_unperturbed import MsC_angular_frequency

for alpha in [0.2, np.sqrt(2)]:

    field_frequency = MsC_angular_frequency(alpha, initial_energy, 1.e-7)
    field_period = 2*np.pi/field_frequency

    delta_t = 1.e-4
    total_time = 2000*field_period
    time_grid_start = np.pi/(2*field_frequency)

    print(f"Calculating MsC ionization amplitude for alpha = {alpha}, field frequency = {field_frequency}")
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

    filename = f"{now}--MsC-classical-ionization-amplitude--a--{round(alpha, 3)}--Omg--{round(field_frequency, 3)}.json"
    with open(filename, "w") as filehandle:
        json.dump(output_data, filehandle)


'''sC potential'''

from emerald.classical.sc_ionization import sC_ionization_amplitude
from emerald.classical.sc_unperturbed import sC_angular_frequency

for alpha in [0.2, np.sqrt(2)]:

    field_frequency = sC_angular_frequency(alpha, initial_energy, 1.e-7)
    field_period = 2*np.pi/field_frequency

    delta_t = 1.e-4
    total_time = 2000*field_period
    time_grid_start = np.pi/(2*field_frequency)

    print(f"Calculating sC ionization amplitude for alpha = {alpha}, field frequency = {field_frequency}")

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

    filename = f"{now}--sC-classical-ionization-amplitude--a--{round(alpha, 3)}--Omg--{round(field_frequency, 3)}.json"
    with open(filename, "w") as filehandle:
        json.dump(output_data, filehandle)


'''Coulomb potential'''

from emerald.classical.coulomb_ionization import C_ionization_amplitude
from emerald.classical.coulomb_unperturbed import C_angular_frequency

for alpha in [0.2, np.sqrt(2)]:

    field_frequency = C_angular_frequency(initial_energy)
    field_period = 2*np.pi/field_frequency

    delta_t = 1.e-4
    total_time = 2000*field_period
    time_grid_start = np.pi/(2*field_frequency)

    print(f"Calculating Coulomb ionization amplitude field frequency = {field_frequency}")

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

    filename = f"{now}--C-classical-ionization-amplitude--a--{round(0, 3)}--Omg--{round(field_frequency, 3)}.json"
    with open(filename, "w") as filehandle:
        json.dump(output_data, filehandle)
