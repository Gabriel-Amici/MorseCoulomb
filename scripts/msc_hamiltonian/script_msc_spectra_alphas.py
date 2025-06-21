
'''
30/03/2025

Program to calculate and save the energies of the unperturbed hamiltonian of an elecron subject to the MsC potential for varying alpha.

Parameters:

alpha
'''


import numpy as np
import time
import json

def MsC_potential(alpha: float, r: float) -> float:

    """Calculates the 1D Morse-Coulomb potencial given the position r"""
    

    D = 1/alpha ; beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = -1/np.sqrt(r*r + alpha*alpha)
    else:
        pot = D*( np.exp( -2*beta*r) -2*np.exp( -beta*r ) )
    return pot

def MsC_hamiltonian(alpha, position_grid):
    
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


alpha_array = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]

data = {}


data["alphas"] = alpha_array

position_grid_extension = 8000
position_grid_size = 8193
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

    position_grid_start = MsC_return_points(alpha, 1.e6)[0]
    position_grid = np.linspace(position_grid_start, position_grid_extension, position_grid_size)

    position_grid_start_array.append(position_grid_start)

    start_time = time.time()
    unperturbed_hamiltonian, position_grid = MsC_hamiltonian(alpha, position_grid)
    end_time = time.time()

    hamiltonian_compute_time_array.append(end_time-start_time)

    print("Unperturbed hamiltonian calculated: {:.2f} seconds".format(end_time-start_time))

    start_time = time.time()
    msc_energies, msc_states = MsC_eigstates(unperturbed_hamiltonian, position_grid, 100)
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

filename = f"L--{position_grid_extension}--N--{position_grid_size}"+"-msc_spectra.json"

with open(filename, "w") as filehandle:
    json.dump(data, filehandle)

