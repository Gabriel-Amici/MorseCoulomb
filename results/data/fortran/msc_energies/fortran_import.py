import numpy as np
import json

data = np.loadtxt("take_2/En500_a1.000.txt")

dic = {}

dic["alpha"] = 1.0
dic["parameters"] = {
    "method": "B-splines",
    "position_grid_start" : -3.5*dic["alpha"],
    "position_grid_end" : 2300,
    "position_grid_size" : 5000,
    "position_grid_distibution" : "logarithmic"
}
dic["base_size"] = len(data)

dic["msc_energies"] = list(data)

with open(f"a--{str(dic["alpha"]).replace(".", "-")}--msc_hamiltonian.json", "w") as filehandle:
    json.dump(dic, filehandle)

################################################################################################

data = np.loadtxt("take_2/En500_a0.500.txt")

dic = {}

dic["alpha"] = 0.5
dic["parameters"] = {
    "method": "B-splines",
    "position_grid_start" : -3.5*dic["alpha"],
    "position_grid_end" : 2300,
    "position_grid_size" : 5000,
    "position_grid_distibution" : "logarithmic"
}
dic["base_size"] = len(data)

dic["msc_energies"] = list(data)

with open(f"a--{str(dic["alpha"]).replace(".", "-")}--msc_hamiltonian.json", "w") as filehandle:
    json.dump(dic, filehandle)
    
################################################################################################

data = np.loadtxt("take_2/En500_a0.100.txt")

dic = {}

dic["alpha"] = 0.1
dic["parameters"] = {
    "method": "B-splines",
    "position_grid_start" : -3.5*dic["alpha"],
    "position_grid_end" : 2300,
    "position_grid_size" : 5000,
    "position_grid_distibution" : "logarithmic"
}
dic["base_size"] = len(data)

dic["msc_energies"] = list(data)

with open(f"a--{str(dic["alpha"]).replace(".", "-")}--msc_hamiltonian.json", "w") as filehandle:
    json.dump(dic, filehandle)    

################################################################################################

data = np.loadtxt("take_2/En500_a0.001.txt")

dic = {}

dic["alpha"] = 0.001
dic["parameters"] = {
    "method": "B-splines",
    "position_grid_start" : -3.5*dic["alpha"],
    "position_grid_end" : 2300,
    "position_grid_size" : 5000,
    "position_grid_distibution" : "logarithmic"
}
dic["base_size"] = len(data)

dic["msc_energies"] = list(data)

with open(f"a--{str(dic["alpha"]).replace(".", "-")}--msc_hamiltonian.json", "w") as filehandle:
    json.dump(dic, filehandle)    