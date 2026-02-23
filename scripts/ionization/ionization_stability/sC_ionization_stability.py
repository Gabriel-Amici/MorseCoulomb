from emerald.classical.sc_ionization import sC_ionization_amplitude_criteria
from emerald.utils import FieldParams

import numpy as np
import json
from datetime import datetime
import time

#potential parameters
alpha = 1.
initial_energy = -0.6689    #soft-Coulomb ground state

#field parameters
field_frequency = 0.8
field_amplitude_set = np.arange(0., 13.2, .4) # F₀ up to 13.2 ≡ α₀ up to 20 -- 33 points

field_period = 2*np.pi/field_frequency
operation_time = 50*field_period
rampup_time = 6*field_period
rampdown_time = 6*field_period

field_params = FieldParams(
    amplitude=0.5,    #is changed in the loop, so this value is just a placeholder
    frequency=field_frequency,  
    operation_time=operation_time,
    rampup_time=rampup_time,
    rampdown_time=rampdown_time,
    form='sin',
    envelope='linear'
)

dt = 1.e-4                  #seems sufficient, 1.e-5 produces same trajectory 
initial_conditions = 502    #1000 initial conditions

simulation_start_time = time.time()

PIs = np.array(sC_ionization_amplitude_criteria(
    alpha=alpha,
    E_0=initial_energy,
    amplitudes=field_amplitude_set,
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
            "field_amplitude_set" : list(field_amplitude_set),
            "field_parameters" : field_params._asdict()
        },
        "results" : {
            "ionization_probabilities_distance_criterion" : list(PIs[:, 0]),
            "ionization_probabilities_energy_criterion" : list(PIs[:, 1]),
            "simulation_duration_hours" : simulation_duration_hours
        }
    }

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--sC-classical-ionization-amplitude--a--{round(alpha, 3)}--Omg--{round(field_frequency, 3)}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)

