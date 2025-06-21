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

import matplotlib.pyplot as plt

'''
PARAMETERS
'''

field_amplitude_set = np.arange(0, 0.075, 0.005)/2 #np.arange(0, 0.11, 0.01)
field_frequency = 0.25
field_period = 2*np.pi/field_frequency

delta_t = 0.1
total_time = 50*field_period

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
time_grid_start = np.pi/(2*field_frequency)
time_grid = np.arange(0, total_time+delta_t, delta_t) + time_grid_start



ionization_probabilities_final = []
ionization_probabilities_mean = []
ionization_probabilities_final_mean = []
ionization_probabilities_max = []

ionization_probability_history = np.zeros((len(field_amplitude_set), len(time_grid)))

i = 0
for field_amplitude in field_amplitude_set:

    #initial_state

    print(F"--AMPLITUDE: {field_amplitude}")
    initial_state_vector = np.zeros(base_size) ; initial_state_vector[1] = 1

    coefficients_history = wavefunction_evolution(time_grid, msc_energies, initial_state_vector, Xi_values, Xi_vectors, Xi_inv, field_amplitude, field_frequency)


    #trim this matrix and save only the squared magnitude
    mask = np.where(msc_energies<0, 1, 0).astype(float)
    ionization_history = 1 - np.dot(mask, np.abs(coefficients_history)**2)
    
    #plt.plot(time_grid, ionization_history, label=str(field_amplitude))
    #plt.legend()
    #plt.show()

    ionization_probabilities_final.append(ionization_history[-1])
    ionization_probabilities_mean.append(np.mean(ionization_history))
    ionization_probabilities_final_mean.append(np.mean(ionization_history[int(0.95*len(time_grid))::]))
    ionization_probabilities_max.append(np.max(ionization_history))

    ionization_probability_history[i] = ionization_history.copy() ; i += 1

    print(f"--IONIZATION(final, mean, max): {round(ionization_probabilities_final[-1]*100, 3)}%, {round(ionization_probabilities_mean[-1]*100, 3)}%, {round(ionization_probabilities_max[-1]*100, 3)}%, ")

output_data  = {
    "parameters" : {
        "alpha" : alpha,
        "initial_energy" : msc_energies[0],
        "field_amplitude_set" : list(field_amplitude_set),
        "field_frequency" : field_frequency,
        "time_grid": {
            "time_grid_start" : time_grid_start,
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

import matplotlib.pyplot as plt
plt.ylim(0, 1)
plt.plot(output_data["parameters"]["field_amplitude_set"], output_data["results"]["ionization_probabilities_final"], label="final")
plt.plot(output_data["parameters"]["field_amplitude_set"], output_data["results"]["ionization_probabilities_mean"], label="mean")
plt.plot(output_data["parameters"]["field_amplitude_set"], output_data["results"]["ionization_probabilities_final_mean"], label="final mean")
plt.plot(output_data["parameters"]["field_amplitude_set"], output_data["results"]["ionization_probabilities_max"], label="max")
plt.legend()
plt.show()

now = datetime.now().strftime(r"%Y-%m-%d--%H-%M")

filename = f"{now}--quantum-ionization-amplitude--a--{alpha}--Omg--{field_frequency}.json"
with open(filename, "w") as filehandle:
    json.dump(output_data, filehandle)



