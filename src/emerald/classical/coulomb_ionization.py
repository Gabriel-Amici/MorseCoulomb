import numpy as np
from numba import njit, prange, vectorize
from ..potentials.coulomb_potential import C_return_points, C_total_energy
from .coulomb_driven import C_transformed_runge_kutta_4
from .coulomb_unperturbed import C_position, C_transformed_momentum


@njit(parallel=True)
def C_ionization_probability(E: float, F_0: float, Omg: float, q1s: np.ndarray, p1s: np.ndarray, t_0: float, total_time: float, h: float = 1.e-5) -> float:

    rM = C_return_points(E)[1]
    Num_conditions = len(q1s)
    ionized = 0
    #togo = Num_conditions
    for n in prange(0, Num_conditions):

        Q = np.array([ q1s[n], p1s[n], t_0, -E ])

        tau = t_0
        while Q[2] < total_time:
            tau += h
            Q = C_transformed_runge_kutta_4( tau, Q, F_0, Omg, h )
    
            if ( Q[0]**2 > 8*rM ):
                if ( C_total_energy( Q[0]**2, Q[1]/(2*Q[0]) ) > 0 ): 
                    ionized += 1
                    break

    P_ion = ionized/Num_conditions

    return P_ion


@njit(parallel=True)
def C_ionization_amplitude( E: float, F_0s: np.ndarray, Omg: float, Num_trajectories: int, t_0: float, total_time: float, h: float = 1.e-4 ):
    
    rmin_C, rmax_C = C_return_points(E)
    angles = np.linspace(0, np.pi, Num_trajectories)[1:Num_trajectories-1]
    positions = np.array( [ C_position( ang, E, rmin_C, rmax_C ) for ang in angles ] )
    
    Num_conditions = int((Num_trajectories-2)*2)
    q1s = np.empty( Num_conditions )
    p1s = np.empty( Num_conditions )

    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        q1s[2*i] = np.sqrt(r_0)
        q1s[2*i+1] = np.sqrt(r_0)

        #momento 
        p1 = C_transformed_momentum( np.sqrt(r_0), E )
        p1s[2*i] = p1 
        p1s[2*i+1] = -p1

    print("Condições iniciais calculadas")

    Pis = { }
    
    for f in prange(len(F_0s)):
        Pi = C_ionization_probability( E, F_0s[f], Omg, q1s, p1s, t_0, total_time, h )
        Pis[F_0s[f]] = Pi

    return sorted(Pis.items())

@njit(parallel=True)
def C_ionization_frequency( E: float, F_0: float, Omgs: np.ndarray, Num_trajectories: int, total_time: float, h: float = 1.e-4 ):
    
    rmin_C, rmax_C = C_return_points(E)
    angles = np.linspace(0, np.pi, Num_trajectories)[1:Num_trajectories-1]
    positions = np.array( [ C_position( ang, E, rmin_C, rmax_C ) for ang in angles ] )
    
    Num_conditions = int((Num_trajectories-2)*2)
    q1s = np.empty( Num_conditions )
    p1s = np.empty( Num_conditions )

    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        q1s[2*i] = np.sqrt(r_0)
        q1s[2*i+1] = np.sqrt(r_0)

        #momento 
        p1 = C_transformed_momentum( np.sqrt(r_0), E )
        p1s[2*i] = p1 
        p1s[2*i+1] = -p1
    
    print("Condições iniciais calculadas")

    Pis = {}

    for o in prange(len(Omgs)):
        Omg = Omgs[o]
        t_0 = np.pi/(2*Omg)
        Pi = C_ionization_probability( E, F_0, Omg, q1s, p1s, t_0, total_time, h )
        Pis[Omg] = Pi

    return sorted(Pis.items())