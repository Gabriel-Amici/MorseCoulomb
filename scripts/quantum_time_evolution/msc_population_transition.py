'''
12/04/2025 - Gabriel A. Amici

Script to perform state transition given the energies and dipole matrix of a system described by the Morse-soft-Coulomb potential.

PARAMETERS

field_amplitude 
initial_state
target_state
delta_t
total_time
key_frames              --total exported screenshots of the system evolution: for large total_time and small delta_t there's an overwhelming ammount of data to be exportes, to reduce file size, only a handful of it get's exported, leaving the inbetweens behind

OUTPUT

data -- object containing simulation parameters and results (history of expansion coefficients, ionization probability)
'''

import numpy as np
import json
import time

from emerald.quantum.coupling_utils import wavefunction_evolution

'''
PARAMETERS
'''

field_amplitude = 0.02    #lower to reach two-level system approximation
initial_state = 0          #zero-based, 0 is the ground state
target_state = 1           #zero-based, 0 is the ground state

delta_t = 0.01
total_time = 1000
key_frames = 1000

#Unperturbed hamiltonian spectrum
with open("../../results/data/msc_hamiltonian/definitive/a--1-0--msc_hamiltonian.json", "r") as filehandle:
    energy_data = json.load(filehandle)
    filehandle.close()

alpha = energy_data["alpha"]

msc_energies = np.array(energy_data["msc_energies"])
base_size = len(msc_energies)
bound_states = int(np.sum(np.where(msc_energies<0, 1, 0)))

#dipole matrix
with open("../../results/data/fortran/msc_interaction_matrix/take_2/a--1-0--dipole_matrix.json", "r") as filehandle:
    dipole_data = json.load(filehandle)
    filehandle.close()

Xi_values = np.array(dipole_data["results"]["Xi_values"])
Xi_vectors = np.array(dipole_data["results"]["Xi_vectors"])
Xi_inv = np.linalg.inv(Xi_vectors)

#field parameter and time grid
field_frequency = msc_energies[target_state] - msc_energies[initial_state]

time_grid_start = np.pi/(2*field_frequency)
time_grid = np.arange(0, total_time+delta_t, delta_t) + time_grid_start

#initial_state
initial_state_vector = np.zeros(base_size) ; initial_state_vector[initial_state] = 1

coefficients_history = wavefunction_evolution(time_grid, msc_energies, initial_state_vector, Xi_values, Xi_vectors, Xi_inv, field_amplitude, field_frequency)

#trim this matrix and save only the squared magnitude

spacing = int(len(time_grid)/key_frames)

coefficients_history = np.abs(coefficients_history[:, ::spacing])**2

mask = np.where(msc_energies<0, 1, 0).astype(float)
ionization_probability = 1 - np.dot(mask, coefficients_history)

output_data  = {
    "parameters" : {
        "alpha" : alpha,
        "initial_state" : initial_state,
        "target_state" : target_state,
        "field_amplitude" : field_amplitude,
        "field_freqency" : field_frequency,
        "key_frames" : key_frames,
        "time_grid": {
            "time_grid_start" : time_grid_start,
            "total_time" : total_time,
            "delta_t" : delta_t
        }
    },
    "results" : {
        "coefficients_history" : [list(x) for x in coefficients_history],
        "ionization_probability" : list(ionization_probability),
        "sliced_time_array" : list(time_grid[::spacing])
    }
}

filename = f"a--{alpha}--F0--{field_amplitude}--transition--{initial_state}-{target_state}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)

import matplotlib.pyplot as plt

for j in range(10):
    plt.plot(output_data["results"]["sliced_time_array"], output_data["results"]["coefficients_history"][:][j], label=f"$n={int(j+1)}$")
#plt.plot(output_data["results"]["sliced_time_array"], output_data["results"]["coefficients_history"][:][2])
plt.legend()
plt.show()

plt.ylim(0, 1)
plt.plot(output_data["results"]["sliced_time_array"], output_data["results"]["ionization_probability"])
plt.show()

