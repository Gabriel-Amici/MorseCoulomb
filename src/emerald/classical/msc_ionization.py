import numpy as np
from numba import njit, prange
from ..potentials.msc_potential import MsC_potential, MsC_return_points, MsC_total_energy
from .msc_driven import MsC_driven_runge_kutta_4
from .msc_unperturbed import MsC_angle, MsC_position, MsC_momentum, MsC_angular_frequency

@njit(parallel=True)
def MsC_ionization_probability(alpha: float, E_0: float, F_0: float, Omg: float, r0s: np.ndarray, p0s: np.ndarray, t_0: float, total_time: float, dt: float = 1.e-5) -> float:

    rM = MsC_return_points(alpha, E_0)[1]
    Num_conditions = len(r0s)
    ionized = 0
    for n in prange(0, Num_conditions):

        X = np.array([ r0s[n], p0s[n] ])

        t = t_0
        while t < total_time:
            t += dt
            X = MsC_driven_runge_kutta_4( alpha, t, X, F_0, Omg, dt )
    
            if ( X[0] > 8*rM ):
                if ( MsC_total_energy(alpha, X[0], X[1]) > 0 ): 
                    ionized += 1
                    break

    P_ion = ionized/Num_conditions

    return P_ion


@njit('f8(f8, f8, f8, f8, f8, f8)')
def MC_compensate_energy(alpha: float, r: float, p: float, F_0:float, FieldFrequency: float, time: float ) -> float:
    
    E = (1/2)*( p + (F_0/FieldFrequency)*np.sin(FieldFrequency*time) )**2 + MsC_potential(alpha, r)

    return E


@njit(parallel=True)
def MsC_ionization_probability_criteria(alpha: float, E_0: float, F_0: float, Omg: float, r0s: np.ndarray, p0s: np.ndarray, t_0: float, total_time: float, dt: float = 1.e-5) -> float:

    rM = MsC_return_points(alpha, E_0)[1]
    Num_conditions = len(r0s)

    field_period = 2*np.pi/Omg
    total_time = 100*field_period + t_0

    ionized_8rM = 0
    ionized_4rM = 0
    ionized_comp = 0

    for n in prange(0, Num_conditions):

        X = np.array([ r0s[n], p0s[n] ])

        occurences = 0
        comp_ionized = False
        rM8_ionized = False
        rM4_ionized = False
        
        t = t_0
        while t < total_time + t_0:
            t += dt
            X = MsC_driven_runge_kutta_4( alpha, t, X, F_0, Omg, dt )
    
            if (MC_compensate_energy(alpha, X[0], X[1], F_0, Omg, t) > 0):
                occurences += 1
            else:
                occurences = 0
            
            if (occurences >= 10) and (comp_ionized == False):
                ionized_comp += 1
                comp_ionized = True

            if ( MsC_total_energy(alpha, X[0], X[1]) > 0 ): 
                if ( X[0] > 8*rM ) and (rM8_ionized == False):
                    ionized_8rM += 1
                    rM8_ionized = True
                if ( X[0] > 4*rM ) and (rM4_ionized == False):
                    ionized_4rM += 1
                    rM4_ionized = True

    Pi_8rM = ionized_8rM/Num_conditions
    Pi_4rM = ionized_4rM/Num_conditions
    Pi_comp = ionized_comp/Num_conditions

    return np.array([Pi_8rM, Pi_4rM, Pi_comp])

@njit(parallel=True)
def MsC_ionization_amplitude( alpha: float, E_0: float, F_0s: np.ndarray, Omg: float, Num_trajectories: int, t_0: float, total_time: float, poly_degree: int, dt: float = 1.e-4 ):
    
    angles = np.linspace( 0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = MsC_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = MsC_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = { }

    for f in prange(len(F_0s)):
        Pi = MsC_ionization_probability( alpha, E_0, F_0s[f], Omg, r0s, p0s, t_0, total_time, dt )
        Pis[F_0s[f]] = Pi

    sorted_keys = sorted(Pis.keys())
    sorted_items = [Pis[key] for key in sorted_keys]
    return sorted_items


@njit(parallel=True)
def MsC_ionization_amplitude_criteria( alpha: float, E_0: float, F_0s: np.ndarray, Omg: float, Num_trajectories: int, t_0: float, total_time: float, poly_degree: int, dt: float = 1.e-4 ):
    
    omg_n = MsC_angular_frequency(alpha, E_0, dt)
    theta_0 = MsC_angle(alpha, E_0, 0, omg_n, 1.e-5)

    angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = MsC_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = MsC_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = { }

    for f in prange(len(F_0s)):
        Pi_8rM, Pi_4rM, Pi_comp = MsC_ionization_probability_criteria( alpha, E_0, F_0s[f], Omg, r0s, p0s, t_0, total_time, dt )
        Pis[F_0s[f]] = np.array([Pi_8rM, Pi_4rM, Pi_comp])

    sorted_keys = sorted(Pis.keys())
    sorted_items = [Pis[key] for key in sorted_keys]
    return sorted_items

@njit(parallel=True)
def MsC_ionization_frequency( alpha: float, E_0: float, F_0: float, Omgs: np.ndarray, Num_trajectories: int, total_time: float, poly_degree: int, dt: float = 1.e-4 ):

    angles = np.linspace( 0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = MsC_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = MsC_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = {}

    for o in prange(len(Omgs)):
        Omg = Omgs[o]
        t_0 = np.pi/(2*Omg)
        Pi = MsC_ionization_probability( alpha, E_0, F_0, Omg, r0s, p0s, t_0, total_time, dt )
        Pis[Omg] = Pi

    sorted_keys = sorted(Pis.keys())
    sorted_items = [Pis[key] for key in sorted_keys]
    return sorted_items



@njit #(parallel=True)
def MsC_ionization_frequency_criteria( alpha: float, E_0: float, F_0: float, Omgs: np.ndarray, Num_trajectories: int, total_time: float, poly_degree: int, dt: float = 1.e-4 ):
    
    omg_n = MsC_angular_frequency(alpha, E_0, dt)
    theta_0 = MsC_angle(alpha, E_0, 0, omg_n, 1.e-5)

    angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = MsC_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = MsC_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = {}

    for o in range(len(Omgs)):
        Omg = Omgs[o]
        t_0 = np.pi/(2*Omg)
        Pi_8rM, Pi_4rM, Pi_comp = MsC_ionization_probability_criteria( alpha, E_0, F_0, Omg, r0s, p0s, t_0, total_time, dt )
        Pis[Omg] = np.array([Pi_8rM, Pi_4rM, Pi_comp])

    sorted_keys = sorted(Pis.keys())
    sorted_items = [Pis[key] for key in sorted_keys]
    return sorted_items


@njit #(parallel=True)
def MsC_ionization_alphas( alphas: np.ndarray, E_0: float, F_0: float, Omg: float, Num_trajectories: int, t_0: float, total_time: float, poly_degree: int, dt: float = 1.e-4 ):


    Pis = {}

    for alpha in alphas:

        omg_n = MsC_angular_frequency(alpha, E_0, dt)
        theta_0 = MsC_angle(alpha, E_0, 0, omg_n, 1.e-5)
    
        angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]
    
        positions = MsC_position(angles, alpha, E_0, poly_degree, 1.e-5)
    
        Num_conditions = int((Num_trajectories-2)*2)
        r0s = np.empty( Num_conditions )
        p0s = np.empty( Num_conditions )
    
    
        for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
            r_0 = positions[i]
            #posicao
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0
    
            #momento 
            p_0 = MsC_momentum(alpha, E_0, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0
    
        print("Condições iniciais calculadas")
        Pi = MsC_ionization_probability( alpha, E_0, F_0, Omg, r0s, p0s, t_0, total_time, dt )
        Pis[alpha] = Pi

    return sorted(Pis.items())


@njit #(parallel=True)
def MsC_ionization_trajectories( alpha: float, E_0: float, F_0: float, Omg: float, Num_trajectories: np.ndarray, t_0: float, total_time: float, dt: float = 1.e-4 ):
    
    omg_n = MsC_angular_frequency(alpha, E_0, dt)
    theta_0 = MsC_angle(alpha, E_0, 0, omg_n, 1.e-5)



    Pis = { }
    for n in range(len(Num_trajectories)):
        angles = np.linspace( theta_0, np.pi, Num_trajectories[n] )[1:Num_trajectories[n]-1]

        positions = MsC_position(angles, alpha, E_0, Num_trajectories[n], 1.e-5)

        Num_conditions = int((Num_trajectories[n]-2)*2)
        r0s = np.empty( Num_conditions )
        p0s = np.empty( Num_conditions )


        for i in range(Num_trajectories[n]-2):               #duplicando o numero de condicoes iniciais para p e -p
            r_0 = positions[i]
            #posicao
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0

            #momento 
            p_0 = MsC_momentum(alpha, E_0, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0

        print("Condições iniciais calculadas")


        Pi = MsC_ionization_probability( alpha, E_0, F_0, Omg, r0s, p0s, t_0, total_time, dt )
        Pis[Num_conditions] = Pi

    return sorted(Pis.items())