'''
15/03/2025 - Gabriel A. Amici

Program to perform transitions from the ground state to the fist excited state of the Morse-soft-Coulomb potential.
Testing the time evolution algorythm.

Parameters range:

L  in [100, 10.000]      #grid length
N  in [513, 10.001]      #number of points in grid
n  in [100, 1.000]       #basis set size
Δt in [1, 0.0001]        #time step size


Base parameters
α  = 0.5
L  = 1.000
N  = 2.501
n  = 200
Δt = 0.01
'''

from emerald.quantum.msc_unperturbed import MsC_hamiltonian, MsC_eigstates
from emerald.quantum.coupling_utils import interaction_matrix, wavefunction_evolution
from emerald.potentials.msc_potential import MsC_return_points

import numpy as np
import time
from datetime import datetime
import json


def converge_test(alpha, L, N, n, total_time, delta_t, F_0):

    data = {}

    data["alpha"] = alpha
    data["base_size"] = n 
    data["field_amplitude"] = F_0

    data["compute_times"] = {}

    r_min = MsC_return_points(alpha, 1.e6)[0] #lower point on the grid far enough to contain a large number of states

    position_grid = np.linspace(r_min, r_min + L, N)
    data["position_grid"] = {"lower_bound": r_min,
                             "extension": L,
                             "points": N}

    start_time = time.time()
    H_0, position_grid = MsC_hamiltonian(alpha, position_grid)
    end_time = time.time()

    print("Unperturbed hamiltonian calculated: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["Hamiltonian calculation"] = end_time-start_time

    start_time = time.time()
    msc_energies, msc_states = MsC_eigstates(H_0, position_grid, n)
    end_time = time.time()

    print("Eigenergies and eigenstates calculated: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["Hamiltonian eigendecomposition"] = end_time-start_time


    start_time = time.time()
    Xi_vectors, Xi_values, Xi_inv = interaction_matrix(position_grid, msc_states)
    end_time = time.time()

    print("Interaction matrix calculated: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["Interaction matrix calculation"] = end_time-start_time

    bohr_frequency = msc_energies[1] - msc_energies[0]                              #bohr frequency is given by ΔE/ħ
    data["field_frequency"] = bohr_frequency

    time_grid = np.arange( 0, total_time, delta_t ) + np.pi/(2*bohr_frequency)      #shift t₀ to where cos(t₀) = 0
    data["time_grid"] = {"lower_bound": time_grid[0],
                             "extension": time_grid[-1]-time_grid[0],
                             "points": len(time_grid)}

    initial_state = np.zeros(n) ; initial_state[0] = 1                              #ground state
    data["initial_state"] = list(initial_state)

    start_time = time.time()
    coefficients_history = wavefunction_evolution(time_grid, msc_energies, initial_state, Xi_values, Xi_vectors, Xi_inv, F_0, bohr_frequency)
    end_time = time.time()
    print("Wavefunction evolution done: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["wavefunction calculation"] = end_time-start_time


    data["Simulation Results"] = {}

    data["Simulation Results"]["msc_energies"] = list(msc_energies)
    data["Simulation Results"]["bound_state_count"] = float(np.sum(np.where(msc_energies<0, 1, 0)))
    data["Simulation Results"]["coeficcients_history"] = [list(x) for x in np.abs(coefficients_history[:10, :]).T**2]

    return data