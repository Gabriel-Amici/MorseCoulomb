# Re-import necessary packages after environment reset
import matplotlib.pyplot as plt

# New x-axis values
field_amplitude_set = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

# New y-axis values for each curve
quantum_data_2 = [
    3.5129676945189203e-12, 0.6173932000984244, 0.868619136992971,
    0.593697259272901, 0.3648553062685631, 0.24031542373487358,
    0.21347205915188283, 0.42789450828744136, 0.10032341520110244,
    0.17806197820274905, 0.2709583039598775
]

ionization_8rM_2 = [
    0.0, 0.0, 0.2533783783783784, 0.4391891891891892, 0.6013513513513513,
    0.6554054054054054, 0.75, 0.8040540540540541, 0.8243243243243243,
    0.8614864864864865, 0.8783783783783784
]

ionization_4rM_2 = [
    0.0, 0.0, 0.2905405405405405, 0.4560810810810811, 0.6148648648648649,
    0.6722972972972973, 0.7567567567567568, 0.8175675675675675,
    0.8344594594594594, 0.8682432432432432, 0.8851351351351351
]

ionization_comp_2 = [0.0, 0.0, 0.7635135135135135, 0.9864864864864865, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(field_amplitude_set, quantum_data_2, 'o:', label='Quantum')
plt.plot(field_amplitude_set, ionization_8rM_2, 'o-', label=r"$r>8r_M$")
plt.plot(field_amplitude_set, ionization_4rM_2, 'o-', label=r"$r>4r_M$")
plt.plot(field_amplitude_set, ionization_comp_2, 'o-', label="compensate")

plt.xlabel(r"$F_0$")
plt.ylabel(r"$P_i$")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.title("Ionization Probability vs $F_0$")
plt.tight_layout()
plt.show()

