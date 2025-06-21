import numpy as np
import json

# Step 1: Load the data
data = np.abs(np.loadtxt("take_2/Dipole_n500alph_a0.500.txt"))

# Step 2: Determine the matrix size n from the number of elements
n = 500

# Step 3: Populate the full matrix
Xi = np.zeros((n, n))
k = 0
for i in range(n):
    for j in range(i, n):
        Xi[i, j] = data[k]
        Xi[j, i] = data[k]
        k += 1

Xi_values, Xi_vectors = np.linalg.eigh(Xi)

dic = {}

dic["alpha"] = 1.0
dic["parameters"] = {
    "method": "B-splines",
    "position_grid_start" : -3.5*dic["alpha"],
    "position_grid_end" : 2300,
    "position_grid_size" : 5000,
    "position_grid_distibution" : "logarithmic"
}
dic["results"] = {
    "Xi_values" : list(Xi_values),
    "Xi_vectors" : [ list(x) for x in Xi_vectors ]
}

with open(f"test--a--{str(dic["alpha"]).replace(".", "-")}--dipole_matrix.json", "w") as filehandle:
    json.dump(dic, filehandle)