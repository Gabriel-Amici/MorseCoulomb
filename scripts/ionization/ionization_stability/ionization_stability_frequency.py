from emerald.classical.msc_ionization import MsC_ionization_frequency_criteria
from emerald.classical.sc_ionization import sC_ionization_frequency_criteria
from emerald.utils import FieldParams

import numpy as np
import json
from datetime import datetime
import time

#field parameters
field_amplitude = 0.4
field_frequency_set = np.arange(0.1, 1.3, .1) # Ω up to 1.2 -- 12 points

# Given in terms of field period, so that the number of cycles
#  is the same for all frequencies
operation_time = 50
rampup_time = 6
rampdown_time = 6

field_params = FieldParams(
    amplitude=field_amplitude,
    frequency=1,      # is changed in the loop, so this value is just a placeholder
    operation_time=operation_time,
    rampup_time=rampup_time,
    rampdown_time=rampdown_time,
    form='sin',
    envelope='linear'
)

dt = 1.e-4                  #seems sufficient, 1.e-5 produces same trajectory 
initial_conditions = 502    #1000 initial conditions

#potential parameters for MsC
alpha = 1.
initial_energy = -0.555510603954473    # Morse soft-Coulomb ground state

simulation_start_time = time.time()

PIs = np.array(MsC_ionization_frequency_criteria(
    alpha=alpha,
    E_0=initial_energy,
    frequencies=field_frequency_set,
    field_params=field_params, 
    total_time=operation_time,
    Num_trajectories=initial_conditions,
    poly_degree=100,
    dt=dt
))

simulation_end_time = time.time()
simulation_duration_hours = round((simulation_end_time - simulation_start_time) / 3600, 2)

output_data  = {
        "parameters" : {
            "potential": "Morse-soft-Coulomb",
            "alpha" : alpha,
            "initial_energy" : initial_energy,
            "initial_conditions" : int( (initial_conditions-2)*2 ),
            "field_frequency_set" : list(field_frequency_set),
            "field_parameters" : field_params._asdict()
        },
        "results" : {
            "ionization_probabilities_distance_criterion" : list(PIs[:, 0]),
            "ionization_probabilities_energy_criterion" : list(PIs[:, 1]),
            "simulation_duration_hours" : simulation_duration_hours
        }
    }

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--MsC-classical-ionization-frequency--a--{alpha:.3f}--F0--{field_amplitude:.3f}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)


#potential parameters for soft-Coulomb
alpha = 1.
initial_energy = -0.6689    #soft-Coulomb ground state

simulation_start_time = time.time()

PIs = np.array(sC_ionization_frequency_criteria(
    alpha=alpha,
    E_0=initial_energy,
    frequencies=field_frequency_set,
    field_params=field_params, 
    total_time=operation_time,
    Num_trajectories=initial_conditions,
    poly_degree=100,
    dt=dt
))

simulation_end_time = time.time()
simulation_duration_hours = round((simulation_end_time - simulation_start_time) / 3600, 2)

output_data  = {
        "parameters" : {
            "potential": "soft-Coulomb",
            "alpha" : alpha,
            "initial_energy" : initial_energy,
            "initial_conditions" : int( (initial_conditions-2)*2 ),
            "field_frequency_set" : list(field_frequency_set),
            "field_parameters" : field_params._asdict()
        },
        "results" : {
            "ionization_probabilities_distance_criterion" : list(PIs[:, 0]),
            "ionization_probabilities_energy_criterion" : list(PIs[:, 1]),
            "simulation_duration_hours" : simulation_duration_hours
        }
    }

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--sC-classical-ionization-frequency--a--{alpha:.3f}--F0--{field_amplitude:.3f}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)
