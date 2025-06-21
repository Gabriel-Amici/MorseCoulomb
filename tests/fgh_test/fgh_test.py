'''
28/03/2025

Program to calculate and save the energies and eigenstates of the unperturbed hamiltonian of an elecron subject to the MsC potential.

Parameters:

alpha
position_grid_extension
postion_grid_size
'''

from emerald.quantum.msc_unperturbed import MsC_hamiltonian, MsC_eigstates
from emerald.potentials.msc_potential import MsC_return_points

import sys
import numpy as np
import time
import datetime
import json

args = {}
argstrings = ["alpha", "position_grid_extension", "position_grid_size"]
for arg in sys.argv[1:]:
    for argstr in argstrings:
        if argstr in arg:
            args[argstr] = float(arg.removeprefix("--"+argstr+"="))

data = {}

alpha = args["alpha"]
data["alpha"] = alpha

position_grid_extension = args["position_grid_extension"]
position_grid_size = int(args["position_grid_size"])
position_grid_start = (MsC_return_points(alpha, -0.001)*2)[0]

position_grid = np.linspace((MsC_return_points(alpha, -0.001)*2)[0], (MsC_return_points(alpha, -0.001)*2)[1], position_grid_size)

data["position_grid"] = {
    "postion_grid_start" : position_grid_start,
    "position_grid_extension" : position_grid_extension,
    "position_grid_size" : position_grid_size
}

data["compute_times"] = {}

start_time = time.time()
unperturbed_hamiltonian, position_grid = MsC_hamiltonian(alpha, position_grid)
end_time = time.time()

data["compute_times"]["hamiltonian_matrix_calculation"] = end_time-start_time
print("Unperturbed hamiltonian calculated: {:.2f} seconds".format(data["compute_times"]["hamiltonian_matrix_calculation"]))

start_time = time.time()
msc_energies, msc_states = MsC_eigstates(unperturbed_hamiltonian, position_grid, 10)
end_time = time.time()

data["compute_times"]["hamiltonian_diagonalization_calculation"] = (end_time-start_time)/60
print("Unperturbed hamiltonian diagonalized: {:.2f} minutes".format(data["compute_times"]["hamiltonian_diagonalization_calculation"]))

unperturbed_hamiltonian = []

data["results"] = {
    "msc_energies" : list(msc_energies),
    #"msc_states" : [list(x) for x in msc_states],
    "bound_states" : float(np.sum(np.where(msc_energies<0, 1, 0)))
}

results_path = ""
filename = datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")+"-msc_hamiltonian.json"

with open(results_path+filename, "w") as filehandle:
    json.dump(data, filehandle)

