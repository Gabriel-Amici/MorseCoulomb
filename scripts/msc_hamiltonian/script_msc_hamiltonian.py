'''
28/03/2025

Program to calculate and save the energies and eigenstates of the unperturbed hamiltonian of an elecron subject to the MsC potential.

Parameters:

alpha
position_grid_extension
postion_grid_size
'''

import sys
import numpy as np
import time
import json
from numba import njit, prange

@njit
def MsC_potential(alpha: float, r: float) -> float:

    """Calculates the 1D Morse-Coulomb potencial given the position r"""
    

    D = 1/alpha ; beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = -1/np.sqrt(r*r + alpha*alpha)
    else:
        pot = D*( np.exp( -2*beta*r) -2*np.exp( -beta*r ) )
    return pot

@njit(parallel=True)
def MsC_hamiltonian(alpha, position_grid):
    
    N = len(position_grid)
    L = position_grid[-1] - position_grid[0]
    delta_r = position_grid[1] - position_grid[0]

    delta_k = 2*np.pi/L
    n = int((N-1)/2)

    def T_l(l, L):
        return 2*(np.pi*l/(L))**2

    H = np.empty( (N, N) )

    for i in prange(0, N):
        for j in prange(0, N):
            l_sum = 0
            for l in range(1, n+1):
                l_sum += np.cos( l*(2*np.pi)*(i - j)/(N-1) )*T_l(l, L)
            #print(l_sum)
            H[i, j] = (2/(N-1))*l_sum + MsC_potential(alpha, position_grid[i])*(i == j)
            #print( str( round(H[i, j], 3) ) + " ", end='')
        #print("\n")
    return H, position_grid

def MsC_return_points(alpha: float, E: float) -> np.ndarray:

    """Calculates the return points of a particle based on alpha and the total Energy"""
    
    rm = -alpha*np.sqrt(2)*np.log( np.sqrt( alpha*E + 1 ) + 1 )
    rM = np.sqrt( 1/(E**2) - alpha**2 )
    
    return np.array([rm, rM])

def MsC_eigstates(H, position_grid, base_size):

    delta_r = position_grid[1] - position_grid[0]

    # Eigendecomposition
    eig_energies, eig_states = np.linalg.eigh(H)
    sort_indexes = np.argsort(eig_energies)
    eig_energies = eig_energies[sort_indexes]
    eig_states = eig_states[:, sort_indexes] / np.sqrt(delta_r)

    return eig_energies[:base_size], eig_states[:, :base_size]


args = {}
argstrings = ["alpha", "position_grid_extension", "position_grid_size"]
for arg in sys.argv[1:]:
    for argstr in argstrings:
        if argstr in arg:
            args[argstr] = float(arg.removeprefix("--"+argstr+"="))

data = {}

alpha = 0.001
data["alpha"] = alpha

position_grid_extension = 200
position_grid_size = 8039
position_grid_start = MsC_return_points(alpha, 1.e5)[0]

position_grid = np.linspace(position_grid_start, position_grid_extension, position_grid_size)

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

#filename = datetime.datetime.now().strftime(r"%Y-%m-%d-%H-%M-%S")+"-msc_hamiltonian.json"
filename = f"L--{position_grid_extension}--N--{position_grid_size}"+"--msc_hamiltonian.json"

print(msc_energies)

with open(filename, "w") as filehandle:
    json.dump(data, filehandle)
