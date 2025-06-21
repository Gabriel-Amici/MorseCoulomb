'''
30/03/2025

Program to calculate and save the energies of the unperturbed hamiltonian of an elecron subject to the sC potential for varying alpha.

Parameters:

alpha
'''

from emerald.quantum.sc_unperturbed import sC_hamiltonian, sC_eigstates
from emerald.potentials.sc_potential import sC_return_points

import sys
import numpy as np
import time
import datetime
import json


#alpha_array = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
alpha_array = [0.0001]

data = {}


data["alphas"] = alpha_array

position_grid_extension = 200
position_grid_size = 20001


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

    
    position_grid = np.linspace(-position_grid_extension/2, position_grid_extension/2, position_grid_size) 
    position_grid += (position_grid[1] - position_grid[0])/2

    start_time = time.time()
    unperturbed_hamiltonian, position_grid = sC_hamiltonian(alpha, position_grid)
    end_time = time.time()

    hamiltonian_compute_time_array.append(end_time-start_time)

    print("Unperturbed hamiltonian calculated: {:.2f} seconds".format(end_time-start_time))

    start_time = time.time()
    msc_energies, msc_states = sC_eigstates(unperturbed_hamiltonian, position_grid, 10)
    end_time = time.time()

    diagonalization_compute_time_array.append((end_time-start_time)/60)
    print("Unperturbed hamiltonian diagonalized: {:.2f} minutes".format((end_time-start_time)/60))

    bound_states_array.append(float(np.sum(np.where(msc_energies<0, 1, 0))))

    data["results"]["energy_levels"][f"alpha={alpha}"] = list(msc_energies)

    unperturbed_hamiltonian = []


data["results"]["bound_states"] = list(bound_states_array)

data["compute_times"]["hamiltonian_matrix_calculation"] = list(hamiltonian_compute_time_array)
data["compute_times"]["hamiltonian_diagonalization_calculation"] = list(diagonalization_compute_time_array)

results_path = "../../results/data/sc_spectra/take_1/"
filename = f"L--{position_grid_extension}--N--{position_grid_size}"+"-sc_spectra.json"

with open(results_path+filename, "w") as filehandle:
    json.dump(data, filehandle)

