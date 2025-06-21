'''
12/04/2025 - Gabriel A. Amici

Script to simulate the effects of a laser interacting with a system modeled by the Morse-soft-Coulomb potential.

PARAMETERS

field_amplitude_set
field_frequency
delta_t
total_time

OUTPUT

data -- object containing simulation parameters and results (history of expansion coefficients, ionization probability)
'''

import numpy as np
import json
from datetime import datetime

from emerald.quantum.coupling_utils import wavefunction_evolution

'''
PARAMETERS
'''

field_amplitude = 0.005
field_frequency_set = np.arange( 0.01, .62, 0.02 )


delta_t = 0.1

initial_state = 1

#Unperturbed hamiltonian spectrum
with open("../../results/data/msc_hamiltonian/definitive/a--0-1--msc_hamiltonian.json", "r") as filehandle:
    energy_data = json.load(filehandle)
    filehandle.close()

alpha = energy_data["alpha"]

msc_energies = np.array(energy_data["msc_energies"])
base_size = len(msc_energies)
bound_states = int(np.sum(np.where(msc_energies<0, 1, 0)))

#dipole matrix
with open("../../results/data/fortran/msc_interaction_matrix/take_2/a--0-1--dipole_matrix.json", "r") as filehandle:
    dipole_data = json.load(filehandle)
    filehandle.close()

Xi_values = np.array(dipole_data["results"]["Xi_values"])
Xi_vectors = np.array(dipole_data["results"]["Xi_vectors"])
Xi_inv = np.linalg.inv(Xi_vectors)

#time grid


ionization_probabilities_final = []
ionization_probabilities_mean = []
ionization_probabilities_final_mean = []
ionization_probabilities_max = []

for field_frequency in field_frequency_set:
    
    field_period = 2*np.pi/field_frequency
    total_time = 100*field_period


    time_grid_start = np.pi/(2*field_frequency)
    time_grid = np.arange(0, total_time+delta_t, delta_t) + time_grid_start
    #initial_state
    print("="*20)
    print(F"--FREQUENCY: {field_frequency}")
    initial_state_vector = np.zeros(base_size) ; initial_state_vector[initial_state] = 1

    coefficients_history = wavefunction_evolution(time_grid, msc_energies, initial_state_vector, Xi_values, Xi_vectors, Xi_inv, field_amplitude, field_frequency)


    #trim this matrix and save only the squared magnitude
    mask = np.where(msc_energies<0, 1, 0).astype(float)
    ionization_history = 1 - np.dot(mask, np.abs(coefficients_history)**2)
    
    ionization_probabilities_final.append(ionization_history[-1])
    ionization_probabilities_mean.append(np.mean(ionization_history))
    ionization_probabilities_final_mean.append(np.mean(ionization_history[int(0.95*len(time_grid))::]))
    ionization_probabilities_max.append(np.max(ionization_history))

    print(f"--IONIZATION(final, mean, max): {round(ionization_probabilities_final[-1]*100, 3)}%, {round(ionization_probabilities_mean[-1]*100, 3)}%, {round(ionization_probabilities_max[-1]*100, 3)}%, ")

output_data  = {
    "parameters" : {
        "alpha" : alpha,
        "initial_energy" : msc_energies[initial_state],
        "field_amplitude" : field_amplitude,
        "field_frequency_set" : list(field_frequency_set),
        "time_grid": {
            "time_grid_starts" : list(np.pi/(2*field_frequency_set)),
            "total_time" : total_time,
            "delta_t" : delta_t
        }
    },
    "results" : {
        "ionization_probabilities_final" : list(ionization_probabilities_final),
        "ionization_probabilities_mean" : list(ionization_probabilities_mean),
        "ionization_probabilities_final_mean" : list(ionization_probabilities_final_mean),
        "ionization_probabilities_max" : list(ionization_probabilities_max)
    }
}

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

import matplotlib.pyplot as plt
plt.ylim(0, 1)
plt.plot(output_data["parameters"]["field_frequency_set"], output_data["results"]["ionization_probabilities_final"], label="final")
plt.plot(output_data["parameters"]["field_frequency_set"], output_data["results"]["ionization_probabilities_mean"], label="mean")
plt.plot(output_data["parameters"]["field_frequency_set"], output_data["results"]["ionization_probabilities_final_mean"], label="final mean")
plt.plot(output_data["parameters"]["field_frequency_set"], output_data["results"]["ionization_probabilities_max"], label="max")
plt.legend()
plt.show()

filename = f"{now}--quantum-ionization-frequency--a--{alpha}--F0--{field_amplitude}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)



