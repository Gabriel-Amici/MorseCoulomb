'''
27/03/2025 - Gabriel A. Amici

Program to check the convergence of the ionization probability of an atom modeled by the MsC potential subject to periodic perturbation. 
Input parameters:

alpha
position_grid_extension
position_grid_size
energy_base_size
time_grid_step
total_time_array
field_amplitude
field_frequency

result: array of ionization probabilities paired with total time array
'''

from emerald.quantum.msc_unperturbed import MsC_hamiltonian, MsC_eigstates
from emerald.quantum.coupling_utils import interaction_matrix, wavefunction_evolution
from emerald.potentials.msc_potential import MsC_return_points

import numpy as np
import time

def ionization_convergence_total_time(
        alpha,
        position_grid_extension,
        position_grid_size,
        energy_base_size,
        time_grid_step,
        total_time_array,
        field_amplitude,
        field_frequency
):

    data = {}

    data["alpha"] = alpha
    data["energy_base_size"] = energy_base_size 
    data["field_amplitude"] = field_amplitude
    data["field_frequency"] = field_frequency

    data["compute_times"] = {}

    r_min = MsC_return_points(alpha, 1.e6)[0] #lower point on the grid far enough to contain a large number of states

    position_grid = np.linspace(r_min, r_min + position_grid_extension, position_grid_size)
    data["position_grid"] = {"start": r_min,
                             "extension": position_grid_extension,
                             "size": position_grid_size}

    start_time = time.time()
    unperturbed_hamiltonian, position_grid = MsC_hamiltonian(alpha, position_grid)
    end_time = time.time()

    print("Unperturbed hamiltonian calculated: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["Hamiltonian calculation"] = end_time-start_time

    start_time = time.time()
    msc_energies, msc_states = MsC_eigstates(unperturbed_hamiltonian, position_grid, energy_base_size)
    end_time = time.time()

    print("Eigenergies and eigenstates calculated: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["Hamiltonian eigendecomposition"] = end_time-start_time

    field_frequency = -msc_energies[0]
    data["field_frequency"] = field_frequency

    start_time = time.time()
    Xi_vectors, Xi_values, Xi_inv = interaction_matrix(position_grid, msc_states)
    end_time = time.time()

    print("Interaction matrix calculated: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["Interaction matrix calculation"] = end_time-start_time

    time_grid_start = np.pi/(2*field_frequency)
    data["time_grid"] = {"start": time_grid_start,
                         "extensions": list(total_time_array),
                         "size": list((np.array(total_time_array)/time_grid_step).astype(np.int64))}

    initial_state = np.zeros(energy_base_size) ; initial_state[0] = 1                                  #ground state
    data["initial_state"] = list(initial_state)

    ionized_array = []

    start_time = time.time()
    for total_time in total_time_array:

        time_grid = np.arange( 0, total_time, time_grid_step ) + time_grid_start      #shift t₀ to where cos(t₀) = 0
        
        coefficients_history = wavefunction_evolution(time_grid, msc_energies, initial_state, Xi_values, Xi_vectors, Xi_inv, field_amplitude, field_frequency)
        print("Wavefunction evolution done")
        
        ionized = 1 - np.dot(np.where(msc_energies<0, 1, 0), np.abs(coefficients_history[:, len(time_grid)-1])**2)
        ionized_array.append(ionized)
    
    end_time = time.time()
    data["compute_times"]["convergence test"] = end_time-start_time

    data["Simulation Results"] = {}

    data["Simulation Results"]["msc_energies"] = list(msc_energies)
    data["Simulation Results"]["bound_state_count"] = float(np.sum(np.where(msc_energies<0, 1, 0)))
    data["Simulation Results"]["ionization_probabilities"] = list(ionized_array)

    return data