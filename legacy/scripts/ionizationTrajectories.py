from MorseCoulombUtils import *
import time
from datetime import datetime
import json

Num_trajectories = np.arange(30, 630, 30)
alpha = 0.5 ; E_0 = -0.5 ; F0 = 0.05 ; Omg = 1. ; t_0 = np.pi/2 ; total_time = 2000.

start = time.time()

Pi = MC_Ionization_Trajectories( alpha, E_0, F0, Omg, Num_trajectories, t_0, total_time, 1.e-4)

end = time.time()

filesave = {
        "Alpha" : alpha,
        "E_0" : E_0,
        "Amplitude" : F0,
        "Frequency" : float(Omg),
        "Initial_conditions" : [ float(x-2)*2 for x in Num_trajectories],
        "Simulation_time" : float(total_time),
        "Integration_step" : 1.e-4,
        "Elapsed_Time" : (end-start)/3600,
        "data" : dict(Pi)
}

now = datetime.now().strftime(r"%d-%m-%Y--%H-%M")

with open("resultados/" + now + "-Morse-Coulomb-ionizationNumba-Trajectories.json", "w") as filehandle: 
        json.dump( filesave, filehandle )