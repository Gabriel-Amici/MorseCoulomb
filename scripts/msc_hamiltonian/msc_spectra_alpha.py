'''
30/03/2025

Program to calculate and save the energies of the unperturbed hamiltonian of an elecron subject to the MsC potential for varying alpha.

Parameters:

alpha
'''

from emerald.quantum.msc_unperturbed import MsC_hamiltonian, MsC_eigstates
from emerald.potentials.msc_potential import MsC_return_points

import sys
import numpy as np
import time
import datetime
import json


alpha_array = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

data = {}


data["alphas"] = alpha_array

position_grid_extension = 200
position_grid_size = 10001
position_grid_start_array = []


data["position_grid"] = {
    "position_grid_end" : position_grid_extension,
    "position_grid_size" : position_grid_size
}

data["compute_times"] = {}

data["results"] = {}
data["results"]["energy_levels"] = {}

hamiltonian_compute_time_array = []
diagonalization_compute_time_array = []

bound_states_array = []


for alpha in alpha_array:

    position_grid_start = -3.5*alpha#MsC_return_points(alpha, 1.e6)[0]
    position_grid = np.linspace(position_grid_start, position_grid_extension, position_grid_size)

    position_grid_start_array.append(position_grid_start)

    position_grid_start_array.append(position_grid_start)

    start_time = time.time()
    unperturbed_hamiltonian, position_grid = MsC_hamiltonian(alpha, position_grid)
    end_time = time.time()

    hamiltonian_compute_time_array.append(end_time-start_time)

    print("Unperturbed hamiltonian calculated: {:.2f} seconds".format(end_time-start_time))

    start_time = time.time()
    msc_energies, msc_states = MsC_eigstates(unperturbed_hamiltonian, position_grid, 10)
    end_time = time.time()

    diagonalization_compute_time_array.append((end_time-start_time)/60)
    print("Unperturbed hamiltonian diagonalized: {:.2f} minutes".format((end_time-start_time)/60))

    bound_states_array.append(float(np.sum(np.where(msc_energies<0, 1, 0))))

    data["results"]["energy_levels"][f"alpha={alpha}"] = list(msc_energies)

    unperturbed_hamiltonian = []

data["position_grid"]["position_grid_start_array"] = position_grid_start_array

data["results"]["bound_states"] = list(bound_states_array)

data["compute_times"]["hamiltonian_matrix_calculation"] = list(hamiltonian_compute_time_array)
data["compute_times"]["hamiltonian_diagonalization_calculation"] = list(diagonalization_compute_time_array)

results_path = "../../results/data/msc_spectra/take_2/"
filename = f"L--{position_grid_extension}--N--{position_grid_size}"+"-msc_spectra.json"

with open(filename, "w") as filehandle:
    json.dump(data, filehandle)

