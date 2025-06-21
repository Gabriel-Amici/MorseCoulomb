import numpy as np
from emerald.quantum.coupling_utils import interaction_matrix
import json

filename = "../../results/data/msc_hamiltonian/take_5/L--1000.0--N--15001--msc_hamiltonian.json"

with open(filename, "r") as filehandle:
    data = json.load(filehandle)

r_m = data["position_grid"]["postion_grid_start"]
r_M = data["position_grid"]["position_grid_extension"]
N = data["position_grid"]["position_grid_size"]

position_grid = np.linspace(r_m, r_M, N)

msc_states = np.array(data["results"]["msc_states"])

dr = position_grid[1] - position_grid[0]
    
# Compute weights
w = np.full(N, 2.0)
w[1::2] = 4.0
w[0] = 1.0
w[-1] = 1.0

# Efficient matrix computation
temp = (w * position_grid)[:, None] * msc_states
Xi = (dr / 3) * np.dot(msc_states.T, temp)

print(Xi[2, :10])

'''
Xi_vectors, Xi_values, Xi_inv = interaction_matrix(position_grid, msc_states)

dic = {}

dic["alpha"] = data["alpha"]
dic["parameters"] = {
    "method": "Fourier Grid Hamiltonian",
    "position_grid_start" : data["position_grid"]["postion_grid_start"],
    "position_grid_end" : data["position_grid"]["position_grid_extension"],
    "position_grid_size" : data["position_grid"]["position_grid_size"],
    "position_grid_distibution" : "uniform"
}
dic["results"] = {
    "Xi_values" : list(Xi_values),
    "Xi_vectors" : [ list(x) for x in Xi_vectors ]
}

with open(f"a--{str(dic["alpha"]).replace(".", "-")}--dipole_matrix.json", "w") as filehandle:
    json.dump(dic, filehandle)
'''