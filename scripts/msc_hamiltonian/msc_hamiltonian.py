'''
28/03/2025

Program to calculate and save the energies and eigenstates of the unperturbed hamiltonian of an elecron subject to the MsC potential.

Parameters:

alpha
position_grid_extension
postion_grid_size
'''

from emerald.quantum.msc_unperturbed import MsC_hamiltonian, MsC_eigstates
from emerald.potentials.msc_potential import MsC_potential, MsC_return_points, MsC_potential_vec

import sys
import numpy as np
import time
import datetime
import json


def MsC_hamiltonian_old(alpha, position_grid):
    
    N = len(position_grid)
    L = position_grid[-1] - position_grid[0]
    delta_r = position_grid[1] - position_grid[0]

    delta_k = 2*np.pi/L
    n = int((N-1)/2)

    def T_l(l, L):
        return 2*(np.pi*l/(L))**2

    H = np.empty( (N, N) )

    for i in range(1, N+1):
        for j in range(1, N+1):
            l_sum = 0
            for l in range(1, n+1):
                l_sum += np.cos( l*(2*np.pi)*(i - j)/(N) )*T_l(l, L)
            #print(l_sum)
            H[i-1, j-1] = (2/(N))*l_sum + MsC_potential(alpha, position_grid[i])*(i == j)
            #print( str( round(H[i, j], 3) ) + " ", end='')
        #print("\n")
    return H, position_grid

def MsC_hamiltonian_2(alpha, position_grid):
    # Grid setup
    from scipy.linalg import toeplitz
    
    N = len(position_grid)
    L = position_grid[-1] - position_grid[0]
    delta_r = position_grid[1] - position_grid[0]


    #N += int(N % 2 == 0)  # Ensure N is odd
    
    
    #r_grid = np.linspace(position_grid[0], position_grid[-1], N)
    
    n = (N-1) // 2

    # Compute the kinetic energy first row
    l = np.arange(n+1)
    g = 2 * (np.pi * l / (L)) ** 2  # g[0] = 0 naturally
    theta = 2 * np.pi / (N-1)
    k = np.arange(N)
    C = np.cos(l[:, None] * k * theta)  # Shape: (n, N)
    t = (2 / (N-1)) * np.dot(g, C)  # First row of T

    # Construct Hamiltonian
    T = toeplitz(t)  # Symmetric Toeplitz matrix
    V = np.diag(MsC_potential_vec(alpha, position_grid))  # Diagonal potential
    H = T + V

    return H, position_grid

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
position_grid_start = -3.5*alpha#MsC_return_points(alpha, 1.e5)[0]

position_grid = np.linspace(position_grid_start, position_grid_extension, position_grid_size)

data["position_grid"] = {
    "position_grid_start" : position_grid_start,
    "position_grid_extension" : position_grid_extension,
    "position_grid_size" : position_grid_size
}

data["compute_times"] = {}

start_time = time.time()
unperturbed_hamiltonian, position_grid = MsC_hamiltonian_2(alpha, position_grid)
end_time = time.time()

data["compute_times"]["hamiltonian_matrix_calculation"] = end_time-start_time
print("Unperturbed hamiltonian calculated: {:.2f} seconds".format(data["compute_times"]["hamiltonian_matrix_calculation"]))

start_time = time.time()
msc_energies, msc_states = MsC_eigstates(unperturbed_hamiltonian, position_grid, 500)
end_time = time.time()

data["compute_times"]["hamiltonian_diagonalization_calculation"] = (end_time-start_time)/60
print("Unperturbed hamiltonian diagonalized: {:.2f} minutes".format(data["compute_times"]["hamiltonian_diagonalization_calculation"]))

unperturbed_hamiltonian = []

data["results"] = {
    "msc_energies" : list(msc_energies),
    "msc_states" : [list(x) for x in msc_states],
    "bound_states" : float(np.sum(np.where(msc_energies<0, 1, 0)))
}

print(msc_energies[0:4])

results_path = "../../results/data/msc_hamiltonian/take_5/"
#filename = datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")+"-msc_hamiltonian.json"
filename = f"L--{position_grid_extension}--N--{position_grid_size}"+"--msc_hamiltonian.json"

print(msc_energies)

with open(results_path+filename, "w") as filehandle:
    json.dump(data, filehandle)
