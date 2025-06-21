import numpy as np
from numba import njit, prange, vectorize
from .sc_driven import sC_driven_runge_kutta_4
from .sc_unperturbed import sC_momentum, sC_angular_frequency, sC_angle, sC_action, sC_position
from ..potentials.sc_potential import sC_total_energy

@njit
def sC_Poincare_section_angle_action( alpha, E: float, F_0: float, Omg: float, section_points: int, r0: float, p0: float, t0: float = 0, dt: float = 1.e-4):

    X = np.empty(2)

    Y = np.empty((section_points, 2))

    X[0] = r0
    X[1] = p0
    k = 0
    it = 0
    t = t0
    j = 0

    period = (2*np.pi/Omg)

    while (k < section_points) and (it < 1.e8) and ( np.isnan(X[0]) == False):
        X = sC_driven_runge_kutta_4( alpha, t, X, F_0, Omg, dt )
        t += dt
        it += 1
        if t > (k+1)*period+t0:

            r = X[0] ; p = X[1]
            E = sC_total_energy(alpha, r, p)

            if (E < 0):

                omg_n = sC_angular_frequency(alpha, E, 1.e-4)
                theta = sC_angle( alpha, E, r, omg_n, 1.e-5) ; action = sC_action(alpha, E, 1.e-6)
                
                if p >= 0:
                    aux = np.array([theta, action])
                else:
                    aux = np.array([-theta, action])
                Y[j] = aux
                j+=1
            k += 1

    Y = Y[:j]            
    '''print( "F_0:", F_0,
            "Omg: ", round(Omg, 3),
            "Pontos:", k, 
            "Iter.:", it,
            "E:", round(E, 4))'''
    return Y
    

def sC_section_trajectories_angle_action(alpha, E: float, F_0: float, Omg: float, section_points: int, t0: float = 0, rs=np.ndarray, dt: float = 1.e-4):
    
    num_trajectories = len(rs)
    arrays = [np.zeros((section_points, 2)) for _ in range(num_trajectories * 2)]
    data = np.empty((0, 2))

    r0s = np.empty(num_trajectories * 2)
    p0s = np.empty(num_trajectories * 2)

    for i in range(num_trajectories):
        if not np.isnan(rs[i]):
            r0s[2 * i] = rs[i]
            r0s[2 * i + 1] = rs[i]
            
            p0 = sC_momentum(alpha, E, rs[i])
            p0s[2 * i] = p0
            p0s[2 * i + 1] = -p0

    for i in range(num_trajectories * 2):
        array = sC_Poincare_section_angle_action(alpha, E, F_0, Omg, section_points, r0s[i], p0s[i], t0 / Omg, dt)
        # Find the index of the first row with all zeros
        first_zero_row = np.where(~array.any(axis=1))[0]
        # Trim the array up to the first row with all zeros
        arrays[i] = array[:first_zero_row[0]] if first_zero_row.size > 0 else array

    for array in arrays:
        data = np.append(data, array, axis=0)

    return data


@njit
def sC_poincare_angle_energy( alpha, F_0: float, Omg: float, section_points: int, r0: float, p0: float, t0: float = 0, dt: float = 1.e-4):

    X = np.empty(2)

    Y = np.empty((section_points, 2))

    X[0] = r0
    X[1] = p0
    k = 0
    it = 0
    t = t0
    j = 0

    period = (2*np.pi/Omg)

    while (k < section_points) and (it < 1.e8) and ( np.isnan(X[0]) == False):
        X = sC_driven_runge_kutta_4( alpha, t, X, F_0, Omg, dt )
        t += dt
        it += 1
        if t > (k+1)*period+t0:

            r = X[0] ; p = X[1]
            E = sC_total_energy(alpha, r, p)

            if (E < 0):

                omg_n = sC_angular_frequency(alpha, E, 1.e-4)
                theta = sC_angle( alpha, E, r, omg_n, 1.e-5)
                
                if p >= 0:
                    aux = np.array([theta, E])
                else:
                    aux = np.array([-theta, E])
                Y[j] = aux
                j+=1
            k += 1

    Y = Y[:j]
    return Y
    

def sC_poincare_energies(alpha, Energies: np.ndarray, F_0: float, Omg: float, section_points: int, num_trajectories: int, t0: float = 0, dt: float = 1.e-4):
    
    num_energies = len(Energies)
    arrays = [np.zeros((section_points, 2)) for _ in range(int(num_trajectories*2*num_energies))]
    data = np.empty((0, 2))

    for i in prange(len(Energies)):
        E = Energies[i]
        omg_n = sC_angular_frequency(alpha, E, dt)
        theta_0 = sC_angle(alpha, E, 0, omg_n, 1.e-5)
    
        angles = np.linspace( theta_0, np.pi, num_trajectories )[1:num_trajectories-1]
    
        positions = sC_position(angles, alpha, E, 50, 1.e-5)
    
        Num_conditions = int((num_trajectories-2)*2)
        r0s = np.empty( Num_conditions )
        p0s = np.empty( Num_conditions )
    
    
        for i in range(num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
            r_0 = positions[i]
            #posicao
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0
    
            #momento 
            p_0 = sC_momentum(alpha, E, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0
    
        print("Condições iniciais calculadas")

        for i in prange(Num_conditions):
            array = sC_poincare_angle_energy(alpha, E, F_0, Omg, section_points, r0s[i], p0s[i], t0 / Omg, dt)
            # Find the index of the first row with all zeros
            first_zero_row = np.where(~array.any(axis=1))[0]
            # Trim the array up to the first row with all zeros
            arrays[i] = array[:first_zero_row[0]] if first_zero_row.size > 0 else array

        for array in arrays:
            data = np.append(data, array, axis=0)

    return data