'''
29/07/2025

Program to calculate the the poincare map of an ensamble of particles under various potentials.

PARAMETERS

alpha
field_amplitude
field_frequency
initial_energy_set
section_points
num_trajectories
time_grid_start
delta_t
total_time

OUTPUT

data -- object containing simulation parameters and results (dots on (r, p) phase space)
'''

import numpy as np
import json
from datetime import datetime


'''
PARAMETERS
'''

initial_energy_set = np.arange(-0.65, -0.3, 0.05)

field_amplitude = 0.025
field_frequency = 1
field_period = 2*np.pi/field_frequency

delta_t = 1.e-4
section_points = 200 # number of points in the Poincare section
num_trajectories = 80
time_grid_start = np.pi/(2*field_frequency)
total_time = time_grid_start + section_points*field_period

alpha = 0.1


'''MsC potential'''

from emerald.classical.msc_poincare import MsC_poincare_energies

poincare_map = MsC_poincare_energies(alpha, initial_energy_set, field_amplitude, field_frequency, section_points, num_trajectories, time_grid_start, delta_t)

output_data  = {
    "parameters" : {
        "potential": "Morse-soft-Coulomb",
        "alpha" : alpha,
        "initial_energy_set" : list(initial_energy_set),
        "initial_conditions" : int( (num_trajectories-2)*2 ),
        "section_points" : section_points,
        "field_amplitude" : field_amplitude,
        "field_frequency" : field_frequency,
        "time_grid": {
            "time_grid_start" : time_grid_start,
            "total_time" : total_time,
            "delta_t" : delta_t
        }
    },
    "results" : {
        "poincare_map" : [list(x) for x in poincare_map]
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--MsC-poincare-energies--a--{alpha}--F0--{field_amplitude}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)

from sys import exit ; exit(0) # Exit after MsC potential to avoid running the other potentials
'''sC potential'''

from emerald.classical.sc_poincare import sC_poincare_energies

poincare_map = sC_poincare_energies(alpha, initial_energy_set, field_amplitude, field_frequency, section_points, num_trajectories, time_grid_start, delta_t)

output_data  = {
    "parameters" : {
        "potential": "soft-Coulomb",
        "alpha" : alpha,
        "initial_energy_set" : list(initial_energy_set),
        "initial_conditions" : int( (num_trajectories-2)*2 ),
        "section_points" : section_points,
        "field_amplitude" : field_amplitude,
        "field_frequency" : field_frequency,
        "time_grid": {
            "time_grid_start" : time_grid_start,
            "total_time" : total_time,
            "delta_t" : delta_t
        }
    },
    "results" : {
        "poincare_map" : [list(x) for x in poincare_map]
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--sC-poincare-energies--a--{alpha}--F0--{field_amplitude}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)


'''nMsC potential'''

from emerald.classical.normalized_msc_poincare import nMsC_poincare_energies

poincare_map = nMsC_poincare_energies(alpha, initial_energy_set, field_amplitude, field_frequency, section_points, num_trajectories, time_grid_start, delta_t)

output_data  = {
    "parameters" : {
        "potential": "normalized Morse-soft-Coulomb",
        "alpha" : alpha,
        "initial_energy_set" : list(initial_energy_set),
        "initial_conditions" : int( (num_trajectories-2)*2 ),
        "section_points" : section_points,
        "field_amplitude" : field_amplitude,
        "field_frequency" : field_frequency,
        "time_grid": {
            "time_grid_start" : time_grid_start,
            "total_time" : total_time,
            "delta_t" : delta_t
        }
    },
    "results" : {
        "poincare_map" : [list(x) for x in poincare_map]
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--nMsC-poincare-energies--a--{alpha}--F0--{field_amplitude}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)


'''Coulomb potential'''

from emerald.classical.coulomb_poincare import C_poincare_energies

poincare_map = C_poincare_energies(initial_energy_set, field_amplitude, field_frequency, section_points, num_trajectories, time_grid_start, delta_t)

output_data  = {
    "parameters" : {
        "potential": "Coulomb",
        "alpha" : 0,
        "initial_energy_set" : list(initial_energy_set),
        "initial_conditions" : int( (num_trajectories-2)*2 ),
        "section_points" : section_points,
        "field_amplitude" : field_amplitude,
        "field_frequency" : field_frequency,
        "time_grid": {
            "time_grid_start" : time_grid_start,
            "total_time" : total_time,
            "delta_t" : delta_t
        }
    },
    "results" : {
        "poincare_map" : [list(x) for x in poincare_map]
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--C-poincare-energies--a--{0}--F0--{field_amplitude}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)

