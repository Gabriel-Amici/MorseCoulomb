import numpy as np
from numba import njit, prange
from .msc_driven import MsC_driven_runge_kutta_4
from .msc_unperturbed import MsC_momentum, MsC_angular_frequency, MsC_angle, MsC_action, MsC_position
from ..potentials.msc_potential import MsC_total_energy
from ..utils import FieldParams

### GLARING PROBLEMS WITH THIS MODULE: ###
'''
1. Horible code formatting, lines are too long,
   function calls become giant lines
2. Calculation of angle-energy and angle-action
   poincare maps could be done in a sinle function
3. Outuput is badly shaped: (N, 2), takes up a lot
of space to save just "["s, better to return two
(1, N) arrays.
4. Bottom line: this should be a method from a class
'''

@njit
def MsC_Poincare_section_angle_action( alpha, E0: float, field_params: FieldParams, section_points: int, r0: float, p0: float, t0: float = 0, dt: float = 1.e-4):

    X = np.empty(2)

    Y = np.empty((section_points, 2))

    X[0] = r0
    X[1] = p0
    k = 0
    it = 0
    t = t0
    j = 0

    period = (2*np.pi/field_params.frequency)

    while (k < section_points) and (it < 1.e8) and ( not np.isnan(X[0])):
        X   = MsC_driven_runge_kutta_4( alpha, t, X, field_params, dt )
        t  += dt
        it += 1
        if t > (k+1)*period+t0:

            r, p = X
            E    = MsC_total_energy(alpha, r, p)

            if (E < 0):

                omg_n  = MsC_angular_frequency(alpha, E, 1.e-4)
                theta  = MsC_angle( alpha, E, r, omg_n, 1.e-5)
                action = MsC_action(alpha, E, 1.e-6)
                
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
    

def MsC_section_trajectories_angle_action(alpha, E0: float, field_params: FieldParams, section_points: int, t0: float = 0, rs=np.ndarray, dt: float = 1.e-4):
    
    num_trajectories = len(rs)
    arrays = [np.zeros((section_points, 2)) for _ in range(num_trajectories * 2)]
    data = np.empty((0, 2))

    r0s = np.empty(num_trajectories * 2)
    p0s = np.empty(num_trajectories * 2)

    for i in range(num_trajectories):
        if not np.isnan(rs[i]):
            r0s[2*i]     = rs[i]
            r0s[2*i + 1] = rs[i]
            
            p0           = MsC_momentum(alpha, E0, rs[i])
            p0s[2*i]     = p0
            p0s[2*i + 1] = -p0

    for i in range(num_trajectories * 2):
        array = MsC_Poincare_section_angle_action(alpha, E0, field_params, section_points, r0s[i], p0s[i], t0 / field_params.frequency, dt)
        # Find the index of the first row with all zeros
        first_zero_row = np.where(~array.any(axis=1))[0]
        # Trim the array up to the first row with all zeros
        arrays[i] = array[:first_zero_row[0]] if first_zero_row.size > 0 else array

    for array in arrays:
        data = np.append(data, array, axis=0)

    return data


@njit
def MsC_poincare_angle_energy( alpha: float, field_params: FieldParams, section_points: int, r0: float, p0: float, t0: float = 0, dt: float = 1.e-4):

    X = np.empty(2)
    Y = np.empty((section_points, 2))

    X  = np.array([r0, p0])
    k  = 0
    it = 0
    t  = t0
    j  = 0

    period = (2*np.pi/field_params.frequency)

    while (k < section_points) and (it < 1.e8) and ( not np.isnan(X[0]) ):
        X = MsC_driven_runge_kutta_4( alpha, t, X, field_params, dt )
        t += dt
        it += 1
        if t > (k+1)*period+t0:

            r, p = X
            E    = MsC_total_energy(alpha, r, p)

            if (E < 0):

                omg_n = MsC_angular_frequency(alpha, E, 1.e-4)
                theta = MsC_angle( alpha, E, r, omg_n, 1.e-5)
                
                if p >= 0:
                    aux = np.array([theta, E])
                else:
                    aux = np.array([-theta, E])
                Y[j] = aux
                j+=1
            k += 1

    Y = Y[:j]
    return Y
    
@njit(parallel=True)
def MsC_poincare_energies(alpha, Energies: np.ndarray, field_params: FieldParams, section_points: int, num_trajectories: int, t0: float = 0, dt: float = 1.e-4):
    
    # num_energies = len(Energies)
    all_results = []  # Will collect all arrays here
    
    for e in prange(len(Energies)):
        E = Energies[e]
    
        # Initial conditions
        angles    = np.linspace(0, np.pi, num_trajectories)[1:num_trajectories-1]
        positions = MsC_position(angles, alpha, E, 50, 1.e-5)
    
        Num_conditions = int((num_trajectories-2)*2)
        r0s = np.empty(Num_conditions)
        p0s = np.empty(Num_conditions)
    
        for i in range(num_trajectories-2):
            r_0 = positions[i]
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0
    
            p_0 = MsC_momentum(alpha, E, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0
    
        print("Initial conditions calculated")
        
        # Store results from parallel computation
        arrays = [np.empty((0, 2)) for _ in range(Num_conditions)]
        
        for i in prange(Num_conditions):
            arrays[i] = MsC_poincare_angle_energy(alpha, field_params, section_points, 
                                                  r0s[i], p0s[i], t0 / field_params.frequency, dt)
        
        # Sequential concatenation after parallel work
        for array in arrays:
            all_results.append(array)
    
    # Final concatenation of all results
    if len(all_results) == 0:
        return np.empty((0, 2))
    
    # Calculate total size needed
    total_rows = 0
    for arr in all_results:
        total_rows += arr.shape[0]
    
    # Create final result array
    data = np.empty((total_rows, 2))
    current_idx = 0
    
    for arr in all_results:
        if arr.shape[0] > 0:
            data[current_idx:current_idx + arr.shape[0]] = arr
            current_idx += arr.shape[0]
    
    return data[:current_idx]