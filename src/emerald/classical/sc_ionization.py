import numpy as np
from numba import njit, prange
from ..potentials.sc_potential import sC_potential, sC_return_points, sC_total_energy
from .sc_driven import sC_driven_runge_kutta_4
from .sc_unperturbed import sC_angle, sC_position, sC_momentum, sC_angular_frequency
from ..utils import FieldParams

@njit(parallel=True)
def sC_ionization_probability(alpha: float, E_0: float, field_params, r0s: np.ndarray, p0s: np.ndarray, t_0: float, total_time: float, dt: float = 1.e-5) -> float:

    rM = sC_return_points(alpha, E_0)[1]
    Num_conditions = len(r0s)
    ionized = 0
    for n in prange(0, Num_conditions):

        X = np.array([ r0s[n], p0s[n] ])

        t = t_0
        while t < total_time:
            t += dt
            X = sC_driven_runge_kutta_4( alpha, t, X, field_params, dt )
    
            if ( X[0] > 8*rM ):
                if ( sC_total_energy(alpha, X[0], X[1]) > 0 ): 
                    ionized += 1
                    break

    P_ion = ionized/Num_conditions

    return P_ion


@njit('f8(f8, f8, f8, f8, f8, f8)')
def sC_compensate_energy(alpha: float, r: float, p: float, F_0:float, FieldFrequency: float, time: float ) -> float:
    
    E = (1/2)*( p + (F_0/FieldFrequency)*np.sin(FieldFrequency*time) )**2 + sC_potential(alpha, r)

    return E


@njit(parallel=True)
def sC_ionization_probability_criteria(alpha: float, E_0: float, field_params, r0s: np.ndarray, p0s: np.ndarray, total_time: float, t_0: float = 0, dt: float = 1.e-5) -> float:

    rM = sC_return_points(alpha, E_0)[1]
    Num_conditions = len(r0s)

    field_period = 2*np.pi/field_params.frequency

    ionized_energy = 0
    ionized_distance = 0

    for n in prange(0, Num_conditions):

        X = np.array([ r0s[n], p0s[n] ])

        occurences = 0
        energy_ionized = False
        distance_ionized = False
        
        t = t_0
        while t < total_time + t_0:
            t += dt
            X = sC_driven_runge_kutta_4( alpha, t, X, field_params, dt )
            
            if ( sC_total_energy(alpha, X[0], X[1]) > 0 ) and (energy_ionized == False):
                ionized_energy += 1 ; energy_ionized = True 
            if ( np.abs(X[0]) > (total_time/field_period)*rM ) and (distance_ionized == False):
                ionized_distance += 1
                distance_ionized = True
    Pi_energy = ionized_energy/Num_conditions
    Pi_distance = ionized_distance/Num_conditions

    return np.array([Pi_distance, Pi_energy])

@njit(parallel=True)
def sC_ionization_amplitude( alpha: float, E_0: float, F_0s: np.ndarray, Omg: float, Num_trajectories: int, t_0: float, total_time: float, poly_degree: int, dt: float = 1.e-4 ):

    angles = np.linspace( 0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = sC_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = sC_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = { }

    for f in prange(len(F_0s)):
        Pi = sC_ionization_probability( alpha, E_0, F_0s[f], Omg, r0s, p0s, t_0, total_time, dt )
        Pis[F_0s[f]] = Pi

    sorted_keys = sorted(Pis.keys())
    sorted_items = [Pis[key] for key in sorted_keys]
    return sorted_items


@njit(parallel=True)
def sC_ionization_amplitude_criteria( alpha: float, E_0: float, amplitudes: np.ndarray, field_params, Num_trajectories: int, total_time: float, poly_degree: int = 100, t_0: float = 0, dt: float = 1.e-4 ):
    
    omg_n = sC_angular_frequency(alpha, E_0, dt)
    theta_0 = sC_angle(alpha, E_0, 0, omg_n, 1.e-5)

    angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = sC_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = sC_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = { }
    done = 0
    for f in prange(len(amplitudes)):
        amplitudes[f] = round(amplitudes[f], 5)
        new_field_params = FieldParams(
                            amplitudes[f],
                            field_params.frequency,
                            field_params.form,
                            field_params.envelope,
                            field_params.rampup_time,
                            field_params.rampdown_time,
                            field_params.operation_time)
        Pi_distance, Pi_energy = sC_ionization_probability_criteria( alpha, E_0, new_field_params, r0s, p0s, total_time, t_0, dt )
        Pis[amplitudes[f]] = np.array([Pi_distance, Pi_energy])
        done += 1
        rounded_Pi_distance = round(Pi_distance, 4) ; rounded_Pi_energy = round(Pi_energy, 4)
        print("(", f, "/", len(amplitudes), ") Amplitude", amplitudes[f], "calculada: Pi_distance =", rounded_Pi_distance, "Pi_energy =", rounded_Pi_energy)
    
    sorted_keys = sorted(Pis.keys())
    sorted_items = [Pis[key] for key in sorted_keys]
    return sorted_items

# @njit(parallel=True)
# def sC_ionization_frequency( alpha: float, E_0: float, F_0: float, Omgs: np.ndarray, Num_trajectories: int, total_time: float, poly_degree: int, dt: float = 1.e-4 ):

#     angles = np.linspace( 0, np.pi, Num_trajectories )[1:Num_trajectories-1]

#     positions = sC_position(angles, alpha, E_0, poly_degree, 1.e-5)

#     Num_conditions = int((Num_trajectories-2)*2)
#     r0s = np.empty( Num_conditions )
#     p0s = np.empty( Num_conditions )


#     for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
#         r_0 = positions[i]
#         #posicao
#         r0s[2*i] = r_0
#         r0s[2*i+1] = r_0

#         #momento 
#         p_0 = sC_momentum(alpha, E_0, r_0)
#         p0s[2*i] = p_0 
#         p0s[2*i+1] = -p_0

#     print("Condições iniciais calculadas")

#     Pis = {}

#     for o in prange(len(Omgs)):
#         Omg = Omgs[o]
#         t_0 = np.pi/(2*Omg)
#         Pi = sC_ionization_probability( alpha, E_0, field_params, r0s, p0s, t_0, total_time, dt )
#         Pis[Omg] = Pi

#     sorted_keys = sorted(Pis.keys())
#     sorted_items = [Pis[key] for key in sorted_keys]
#     return sorted_items



# @njit #(parallel=True)
# def sC_ionization_frequency_criteria( alpha: float, E_0: float, F_0: float, Omgs: np.ndarray, Num_trajectories: int, total_time: float, poly_degree: int, dt: float = 1.e-4 ):
    
#     omg_n = sC_angular_frequency(alpha, E_0, dt)
#     theta_0 = sC_angle(alpha, E_0, 0, omg_n, 1.e-5)

#     angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]

#     positions = sC_position(angles, alpha, E_0, poly_degree, 1.e-5)

#     Num_conditions = int((Num_trajectories-2)*2)
#     r0s = np.empty( Num_conditions )
#     p0s = np.empty( Num_conditions )


#     for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
#         r_0 = positions[i]
#         #posicao
#         r0s[2*i] = r_0
#         r0s[2*i+1] = r_0

#         #momento 
#         p_0 = sC_momentum(alpha, E_0, r_0)
#         p0s[2*i] = p_0 
#         p0s[2*i+1] = -p_0

#     print("Condições iniciais calculadas")

#     Pis = {}

#     for o in range(len(Omgs)):
#         Omg = Omgs[o]
#         t_0 = np.pi/(2*Omg)
#         Pi_8rM, Pi_4rM, Pi_comp = sC_ionization_probability_criteria( alpha, E_0, field_params, r0s, p0s, t_0, total_time, dt )
#         Pis[Omg] = np.array([Pi_8rM, Pi_4rM, Pi_comp])

#     sorted_keys = sorted(Pis.keys())
#     sorted_items = [Pis[key] for key in sorted_keys]
#     return sorted_items


@njit #(parallel=True)
def sC_ionization_alphas( alphas: np.ndarray, E_0: float, field_params, Num_trajectories: int, t_0: float, total_time: float, poly_degree: int, dt: float = 1.e-4 ):


    Pis = {}

    for alpha in alphas:

        omg_n = sC_angular_frequency(alpha, E_0, dt)
        theta_0 = sC_angle(alpha, E_0, 0, omg_n, 1.e-5)
    
        angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]
    
        positions = sC_position(angles, alpha, E_0, poly_degree, 1.e-5)
    
        Num_conditions = int((Num_trajectories-2)*2)
        r0s = np.empty( Num_conditions )
        p0s = np.empty( Num_conditions )
    
    
        for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
            r_0 = positions[i]
            #posicao
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0
    
            #momento 
            p_0 = sC_momentum(alpha, E_0, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0
    
        print("Condições iniciais calculadas")
        Pi = sC_ionization_probability( alpha, E_0, field_params, r0s, p0s, t_0, total_time, dt )
        Pis[alpha] = Pi

    return sorted(Pis.items())


@njit #(parallel=True)
def sC_ionization_trajectories( alpha: float, E_0: float, field_params, Num_trajectories: np.ndarray, t_0: float, total_time: float, dt: float = 1.e-4 ):
    
    omg_n = sC_angular_frequency(alpha, E_0, dt)
    theta_0 = sC_angle(alpha, E_0, 0, omg_n, 1.e-5)



    Pis = { }
    for n in range(len(Num_trajectories)):
        angles = np.linspace( theta_0, np.pi, Num_trajectories[n] )[1:Num_trajectories[n]-1]

        positions = sC_position(angles, alpha, E_0, Num_trajectories[n], 1.e-5)

        Num_conditions = int((Num_trajectories[n]-2)*2)
        r0s = np.empty( Num_conditions )
        p0s = np.empty( Num_conditions )


        for i in range(Num_trajectories[n]-2):               #duplicando o numero de condicoes iniciais para p e -p
            r_0 = positions[i]
            #posicao
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0

            #momento 
            p_0 = sC_momentum(alpha, E_0, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0

        print("Condições iniciais calculadas")


        Pi = sC_ionization_probability( alpha, E_0, field_params, r0s, p0s, t_0, total_time, dt )
        Pis[Num_conditions] = Pi

    return sorted(Pis.items())