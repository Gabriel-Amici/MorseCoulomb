'''
29/03/20256

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

data -- object containing simulation parameters and results (dots on (θ, E) phase space)
'''
from emerald.utils import FieldParams
from emerald.classical.msc_poincare import MsC_poincare_energies
from emerald.classical.sc_poincare import sC_poincare_energies

import numpy as np
import json
from datetime import datetime


'''
PARAMETERS
'''

# System parameters
alpha = np.sqrt(2)
# initial_energy_set = np.arange(-0.65, -0.3, 0.05)
initial_energy_set = np.arange(-0.6, -0.4, 0.05)

# Field parameters
field_amplitude    = 0.8
field_frequency    = 0.8
field_period       = 2*np.pi/field_frequency
delta_t            = 1.e-3

# Section parameters
section_points     = 50 
num_trajectories   = 125
# num_trajectories   = 50


field_params = FieldParams(
    amplitude      = field_amplitude,
    frequency      = field_frequency,
    form           = 'sin',
    envelope       = 'linear',
    rampup_time    = 6*field_period,
    rampdown_time  = 6*field_period,
    operation_time = section_points*field_period
)

'''MsC potential'''

poincare_map = MsC_poincare_energies(
    alpha=alpha,
    Energies=initial_energy_set,
    field_params=field_params,
    section_points=section_points,
    num_trajectories=num_trajectories,
    dt=delta_t
)
angle_values  = poincare_map[:, 0]
energy_values = poincare_map[:, 1]

output_data  = {
        "parameters" : {
            "potential"          : "Morse-soft-Coulomb",
            "alpha"              : alpha,
            "energy_set"         : list(initial_energy_set),
            "initial_conditions" : int( (num_trajectories-2)*2 ),
            "field_parameters"   : field_params._asdict()
        },
        "results" : {
            "angle_values"  : list(angle_values),
            "energy_values" : list(energy_values)
        }
    }

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--MsC-poincare-energies--a--{alpha:<.3f}--F0--{field_amplitude:<.3f}--Omg--{field_frequency:<.3f}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)


'''sC potential'''


# poincare_map = sC_poincare_energies(
#     alpha=alpha,
#     Energies=initial_energy_set,
#     field_params=field_params,
#     section_points=section_points,
#     num_trajectories=num_trajectories,
#     dt=delta_t
# )
# angle_values  = poincare_map[:, 0]
# energy_values = poincare_map[:, 1]

# output_data  = {
#         "parameters" : {
#             "potential"          : "soft-Coulomb",
#             "alpha"              : alpha,
#             "energy_set"         : list(initial_energy_set),
#             "initial_conditions" : int( (num_trajectories-2)*2 ),
#             "field_parameters"   : field_params._asdict()
#         },
#         "results" : {
#             "angle_values"  : list(angle_values),
#             "energy_values" : list(energy_values)
#         }
#     }

# now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

# filename = f"{now}--sC-poincare-energies--a--{alpha:<.3f}--F0--{field_amplitude:<.3f}--Omg--{field_frequency:<.3f}.json"
# with open(filename, "w") as filehandle:
#     json.dump(output_data, filehandle)


