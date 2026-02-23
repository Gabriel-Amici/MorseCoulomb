from emerald.potentials.sc_potential import sC_total_energy, sC_return_points
from emerald.utils import FieldParams, external_field_array
from emerald.classical.sc_unperturbed import sC_momentum
from emerald.classical.sc_ionization import sC_ionization_amplitude_criteria
 
import numpy as np
import matplotlib.pyplot as plt

#potential parameters
alpha = 1.
E0 = -0.6689    #soft-Coulomb ground state

#field parameters
omg = 0.8
amplitudes = np.arange(0., 5, 1)

field_period = 2*np.pi/omg

total_time = 50*field_period
rampup_time = 6*field_period
rampdown_time = 6*field_period

field_params = FieldParams(
    amplitude=0.5,
    frequency=omg,
    envelope='linear',
    form='sin',
    rampup_time=rampup_time,
    rampdown_time=rampdown_time,
    operation_time=total_time
)

dt = 1.e-4
num_trajectories = 15

PIs = np.array(sC_ionization_amplitude_criteria(
    alpha=alpha,
    E_0=E0,
    amplitudes=amplitudes,
    field_params=field_params, 
    total_time=total_time,
    Num_trajectories=num_trajectories,
    poly_degree=100,
    dt=dt
))

print(PIs)

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axs[0].plot(amplitudes, PIs[:, 1], marker='o', label='Energy Criterion')
axs[1].plot(amplitudes, PIs[:, 0], marker='s', label='Distance Criterion')
axs[0].set_xlabel('Field Amplitude')
axs[0].set_ylabel('Ionization Probability')
axs[1].set_xlabel('Field Amplitude')
axs[1].set_ylabel('Ionization Probability')
axs[0].set_title('Ionization Probability vs Field Amplitude')
axs[0].legend()
axs[0].grid()
axs[1].grid()
axs[1].legend()
plt.tight_layout()
plt.show()