from MorseCoulombUtils import *
import time
from datetime import datetime
import json

'''CONDICOES INICIAIS'''

E_0 = -0.75
Omg = 1.
Num_trajectories = 150
t_0 = np.pi/2
total_time = 2000

F0s = np.arange( 0., 0.105, 5.e-3 )


'''AMPLITUDE'''

'''MORSE-COULOMB'''

'''ALPHA = 1'''
alpha = 1
h = 1.e-4

start = time.time()

Pi = MC_Ionization_Amplitude( alpha, E_0, F0s, Omg, Num_trajectories, t_0, total_time, 80, h)

end = time.time()

filesave = {
        "Alpha" : alpha,
        "E_0" : E_0,
        "Amplitudes" : list(F0s),
        "Frequency" : Omg,
        "Initial_conditions" : int((Num_trajectories-2)*2),
        "Simulation_time" : total_time,
        "Integration_step" : h,
        "Elapsed_Time" : (end-start)/3600,
        "data" : dict(Pi)
}

now = datetime.now().strftime(r"%d-%m-%Y--%H-%M")

with open("resultados/" + now + "-Morse-Coulomb-ionizationNumba-Amplitude.json", "w") as filehandle: 
        json.dump( filesave, filehandle )

print('Alpha: ', alpha, 'finalizado. Tempo total de execução: ', (end - start)/3600, ' horas')


'''ALPHA = 0.5'''
alpha = 0.5
h = 1.e-4

start = time.time()

Pi = MC_Ionization_Amplitude( alpha, E_0, F0s, Omg, Num_trajectories, t_0, total_time, 160, h)

end = time.time()

filesave = {
        "Alpha" : alpha,
        "E_0" : E_0,
        "Amplitudes" : list(F0s),
        "Frequency" : Omg,
        "Initial_conditions" : int((Num_trajectories-2)*2),
        "Simulation_time" : total_time,
        "Integration_step" : h,
        "Elapsed_Time" : (end-start)/3600,
        "data" : dict(Pi)
}

now = datetime.now().strftime(r"%d-%m-%Y--%H-%M")

with open("resultados/" + now + "-Morse-Coulomb-ionizationNumba-Amplitude.json", "w") as filehandle: 
        json.dump( filesave, filehandle )

print('Alpha: ', alpha, 'finalizado. Tempo total de execução: ', (end - start)/3600, ' horas')


'''ALPHA = 0.1'''
alpha = 0.1
h = 1.e-4

start = time.time()

Pi = MC_Ionization_Amplitude( alpha, E_0, F0s, Omg, Num_trajectories, t_0, total_time, 100, h)

end = time.time()

filesave = {
        "Alpha" : alpha,
        "E_0" : E_0,
        "Amplitudes" : list(F0s),
        "Frequency" : Omg,
        "Initial_conditions" : int((Num_trajectories-2)*2),
        "Simulation_time" : total_time,
        "Integration_step" : h,
        "Elapsed_Time" : (end-start)/3600,
        "data" : dict(Pi)
}

now = datetime.now().strftime(r"%d-%m-%Y--%H-%M")

with open("resultados/" + now + "-Morse-Coulomb-ionizationNumba-Amplitude.json", "w") as filehandle: 
        json.dump( filesave, filehandle )

print('Alpha: ', alpha, 'finalizado. Tempo total de execução: ', (end - start)/3600, ' horas')


'''ALPHA = 0.05'''
alpha = 0.05
h = 1.e-4

start = time.time()

Pi = MC_Ionization_Amplitude( alpha, E_0, F0s, Omg, Num_trajectories, t_0, total_time, 180, h)

end = time.time()

filesave = {
        "Alpha" : alpha,
        "E_0" : E_0,
        "Amplitudes" : list(F0s),
        "Frequency" : Omg,
        "Initial_conditions" : int((Num_trajectories-2)*2),
        "Simulation_time" : total_time,
        "Integration_step" : h,
        "Elapsed_Time" : (end-start)/3600,
        "data" : dict(Pi)
}

now = datetime.now().strftime(r"%d-%m-%Y--%H-%M")

with open("resultados/" + now + "-Morse-Coulomb-ionizationNumba-Amplitude.json", "w") as filehandle: 
        json.dump( filesave, filehandle )

print('Alpha: ', alpha, 'finalizado. Tempo total de execução: ', (end - start)/3600, ' horas')


'''ALPHA = 0.01'''
alpha = 0.01
h = 1.e-5

start = time.time()

Pi = MC_Ionization_Amplitude( alpha, E_0, F0s, Omg, Num_trajectories, t_0, total_time, 100, h)

end = time.time()

filesave = {
        "Alpha" : alpha,
        "E_0" : E_0,
        "Amplitudes" : list(F0s),
        "Frequency" : Omg,
        "Initial_conditions" : int((Num_trajectories-2)*2),
        "Simulation_time" : total_time,
        "Integration_step" : h,
        "Elapsed_Time" : (end-start)/3600,
        "data" : dict(Pi)
}

now = datetime.now().strftime(r"%d-%m-%Y--%H-%M")

with open("resultados/" + now + "-Morse-Coulomb-ionizationNumba-Amplitude.json", "w") as filehandle: 
        json.dump( filesave, filehandle )

print('Alpha: ', alpha, 'finalizado. Tempo total de execução: ', (end - start)/3600, ' horas')


'''COULOMB'''
'''ALPHA = 0'''
h = 1.e-5           #passo no tempo para o Coulomb é geralmente 4x maior que h

start = time.time()

Pi = C_Ionization_Amplitude( E_0, F0s, Omg, Num_trajectories, t_0, total_time, h)

end = time.time()

filesave = {
        "E_0" : E_0,
        "Amplitudes" : list(F0s),
        "Frequency" : Omg,
        "Initial_conditions" : int((Num_trajectories-2)*2),
        "Simulation_time" : total_time,
        "Integration_step" : h,
        "Elapsed_Time" : (end-start)/3600,
        "data" : dict(Pi)
}

now = datetime.now().strftime(r"%d-%m-%Y--%H-%M")

with open("resultados/" + now + "-Coulomb-ionizationNumba-Amplitude.json", "w") as filehandle: 
        json.dump( filesave, filehandle )

print('Alpha: 0 finalizado. Tempo total de execução: ', (end - start)/3600, ' horas')