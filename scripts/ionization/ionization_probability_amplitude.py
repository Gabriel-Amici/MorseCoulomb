'''
27/03/2025 - Gabriel A. Amici

Program to calculate the probability of ionization of an atom modeled by the MsC potential subject to periodic perturbation for varying field amplitudes. 
Input parameters:

alpha
position_grid_extension
position_grid_size
energy_base_size
time_grid_step
total_time
field_amplitude_array
field_frequency
'''

from emerald.quantum.msc_unperturbed import MsC_hamiltonian, MsC_eigstates
from emerald.quantum.coupling_utils import interaction_matrix, wavefunction_evolution
from emerald.potentials.msc_potential import MsC_return_points

import numpy as np
import time

def ionization_probability_amplitude(
        alpha,
        position_grid_extension,
        position_grid_size,
        energy_base_size,
        time_grid_step,
        total_time,
        field_amplitude_array,
        field_frequency
):

    data = {}

    data["alpha"] = alpha
    data["energy_base_size"] = energy_base_size 
    data["field_amplitude_array"] = list(field_amplitude_array)
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


    start_time = time.time()
    Xi_vectors, Xi_values, Xi_inv = interaction_matrix(position_grid, msc_states)
    end_time = time.time()

    print("Interaction matrix calculated: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["Interaction matrix calculation"] = end_time-start_time


    time_grid = np.arange( 0, total_time, time_grid_step ) + np.pi/(2*field_frequency)      #shift t₀ to where cos(t₀) = 0
    data["time_grid"] = {"start": time_grid[0],
                         "extension": time_grid[-1]-time_grid[0],
                         "size": len(time_grid)}

    initial_state = np.zeros(energy_base_size) ; initial_state[0] = 1                                  #ground state
    #initial_state = np.ones(energy_base_size)*np.sqrt(1/energy_base_size)                              #distributed state
    #initial_state = np.where(msc_energies<0, 1, 0)*np.sqrt(1/np.sum(np.where(msc_energies<0, 1, 0)))    #distributed bound state
    data["initial_state"] = list(initial_state)

    ionized_array = []

    start_time = time.time()
    for field_amplitude in field_amplitude_array:
        print("Probability ionization calculation for field amplitude of {}".format(field_amplitude))
        coefficients_history = wavefunction_evolution(time_grid, msc_energies, initial_state, Xi_values, Xi_vectors, Xi_inv, field_amplitude, field_frequency)
        ionized = 1 - np.dot(np.where(msc_energies<0, 1, 0), np.abs(coefficients_history[:, len(time_grid)-1])**2)

        ionized_array.append(ionized)

    end_time = time.time()
    print("Simulation done: {:.2f} seconds".format(end_time-start_time))
    data["compute_times"]["main simulation"] = end_time-start_time


    data["Simulation Results"] = {}

    data["Simulation Results"]["msc_energies"] = list(msc_energies)
    data["Simulation Results"]["bound_state_count"] = float(np.sum(np.where(msc_energies<0, 1, 0)))
    data["Simulation Results"]["ionization_probability_array"] = list(ionized_array)

    return data