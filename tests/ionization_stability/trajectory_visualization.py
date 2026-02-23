from emerald.classical.sc_driven import sC_driven_trajectory
from emerald.potentials.sc_potential import sC_total_energy, sC_return_points
from emerald.utils import FieldParams, external_field_array
from emerald.classical.sc_unperturbed import sC_momentum
import numpy as np
import matplotlib.pyplot as plt

#potential parameters
alpha = 1.
E0 = -0.6689    #soft-Coulomb ground state

#field parameters
omg = 0.8
amplitudes = np.arange(0., 10.5, 0.5)

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


#integration of the parameters
dt = 1.e-4
time_array = np.arange(0, total_time, .1)

field_array = external_field_array(time_array, field_params)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

from time import time

ionized_energy = 0
ionized_distance = 0

trajectories = 500

#initial conditions
for r0 in np.array([round(r, 3) for r in np.linspace(sC_return_points(alpha, E0)[0], sC_return_points(alpha, E0)[1], trajectories)]):
    start = time()
    r0 = r0
    p0 = sC_momentum(alpha, E0, r0)

    X0 = np.array([r0, p0])
    trajectory = sC_driven_trajectory(alpha, time_array, X0, field_params, dt)

    end = time()
    print('Time for r0 = {}: {:.2f} seconds'.format(r0, end - start))
    print('Final state for r0 = {}: r = {:.4f}, p = {:.4f}'.format(r0, trajectory[-1,0], trajectory[-1,1]))

    energy_array = np.array([ sC_total_energy(alpha, r, p) for r,p in trajectory ])
    axs[1,0].plot(time_array, energy_array, label='$r_0={}$'.format(r0))
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('Energy')


    #plot the trajectory, field, energy and phase space
    axs[0,0].plot(time_array, trajectory[:,0], label='$r_0={}$'.format(r0))
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('$r$')
    axs[0,0].legend()

    axs[1,1].scatter(trajectory[::10,0], trajectory[::10,1],s=0.1, label='$r_0={}$'.format(r0))
    axs[1,1].set_xlabel('$r$')
    axs[1,1].set_ylabel('$p$')

    ionized_energy += energy_array[-1] > 0
    ionized_distance += np.abs(trajectory[-1,0]) > (total_time/field_period)*sC_return_points(alpha, E0)[1]

axs[0,1].plot(time_array, field_array)
axs[0,1].set_xlabel('Time')
axs[0,1].set_ylabel('Field')

print('Ionization probability (energy criterion): {:.2f}%'.format(100*ionized_energy/trajectories))
print('Ionization probability (distance criterion): {:.2f}%'.format(100*ionized_distance/trajectories))

# old_trajectory = trajectory.copy()
# dt = 1.e-4
# time_array = np.arange(0, total_time, dt)
# trajectory = sC_driven_trajectory(alpha, time_array, X0, field_params, dt)
# axs[0, 0].plot(time_array[::1000], trajectory[::1000,0], label='$r(t)$ (dt = 1e-4)')
# axs[0, 0].legend()

plt.tight_layout()
plt.show()

# trajectory = trajectory[::1000, :]

# print(np.mean(np.abs(trajectory[:1000, 0] - old_trajectory[:1000, 0])))
# print(np.mean(np.abs(trajectory[:1000, 1] - old_trajectory[:1000, 1])))