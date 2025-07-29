import numpy as np
from numba import njit, prange
from ..potentials.coulomb_potential import C_total_energy, C_return_points
from .coulomb_unperturbed import C_angle, C_action, C_transformed_momentum, C_position
from .coulomb_driven import C_transformed_runge_kutta_4

parallelization_depth = 0

def set_parallelization_depth(depth):
    global parallelization_depth
    parallelization_depth = depth

def get_parallelization_depth():
    return parallelization_depth


@njit
def C_poincare_angle_action( E: float, F_0: float, Omg: float, section_points: int, q1: float, p1: float, q20: float = 0, h: float = 1.e-4):
    X = np.empty(4)

    Y = np.empty((section_points, 2))

    X[0] = q1
    X[1] = p1
    X[2] = q20
    X[3] = -E
    k = 0
    it = 0
    tau = q20
    j = 0

    period = (2*np.pi/Omg)

    while (k < section_points) and (it < 1.e8) and ( np.isnan(X[0]) == False):
        X = C_transformed_runge_kutta_4( tau, X, F_0, Omg, h )
        tau += h
        it += 1
        if X[2] > (k+1)*period+q20:

            r = X[0]**2 ; p = X[1]/(2*X[0])
            E = C_total_energy(r, p)

            if (E < 0) and ( np.abs(E)*r < 1 ):
                theta = C_angle(E, r) ; action = C_action(E)
                
                if p >= 0:
                    aux = np.array([theta, action])
                else:
                    aux = np.array([-theta, action])
                Y[j] = aux
                j+=1
            k += 1

    Y = Y[:j]
    return Y


@njit(parallel=lambda: parallelization_depth >= 1, cache=False)
def C_section_trajectories_angle_action(E: float, F_0: float, Omg: float, section_points: int, q20: float = 0, rs=np.ndarray, h: float = 1.e-4):
    num_trajectories = len(rs)
    arrays = [np.zeros((section_points, 2)) for _ in range(num_trajectories * 2)]
    data = np.empty((0, 2))

    r0s = np.empty(num_trajectories * 2)
    q1s = np.empty(num_trajectories * 2)
    p1s = np.empty(num_trajectories * 2)

    for i in range(num_trajectories):
        if not np.isnan(rs[i]):
            r0s[2 * i] = rs[i]
            r0s[2 * i + 1] = rs[i]
            q10 = np.sqrt(rs[i])
            q1s[2 * i] = q10
            q1s[2 * i + 1] = q10
            p10 = C_transformed_momentum(q10, E)
            p1s[2 * i] = p10
            p1s[2 * i + 1] = -p10

    for i in range(num_trajectories * 2):
        array = C_poincare_angle_action(E, F_0, Omg, section_points, q1s[i], p1s[i], q20 / Omg, h)
        # Find the index of the first row with all zeros
        first_zero_row = np.where(~array.any(axis=1))[0]
        # Trim the array up to the first row with all zeros
        arrays[i] = array[:first_zero_row[0]] if first_zero_row.size > 0 else array

    for array in arrays:
        data = np.append(data, array, axis=0)

    return data


@njit
def C_poincare_angle_energy( E: float, F_0: float, Omg: float, section_points: int, q1: float, p1: float, q20: float = 0, h: float = 1.e-4):

    X = np.empty(4)

    Y = np.empty((section_points, 2))

    X[0] = q1
    X[1] = p1
    X[2] = q20
    X[3] = -E
    k = 0
    it = 0
    tau = q20
    j = 0

    period = (2*np.pi/Omg)

    while (k < section_points) and (it < 1.e8) and ( np.isnan(X[0]) == False):
        X = C_transformed_runge_kutta_4( tau, X, F_0, Omg, h )
        tau += h
        it += 1
        if X[2] > (k+1)*period+q20:

            r = X[0]**2 ; p = X[1]/(2*X[0])
            E = C_total_energy(r, p)

            if (E < 0) and ( np.abs(E)*r < 1 ):
                theta = C_angle(E, r)
                
                if p >= 0:
                    aux = np.array([theta, E])
                else:
                    aux = np.array([-theta, E])
                Y[j] = aux
                j+=1
            k += 1

    Y = Y[:j]
    return Y
    

@njit(parallel=lambda: parallelization_depth >= 1, cache=False)
def C_section_angle_energy(E: float, F_0: float, Omg: float, section_points: int, q20: float = 0, rs=np.ndarray, h: float = 1.e-4):

    num_trajectories = len(rs)
    arrays = [np.zeros((section_points, 2)) for _ in range(num_trajectories * 2)]
    data = np.empty((0, 2))

    r0s = np.empty(num_trajectories * 2)
    q1s = np.empty(num_trajectories * 2)
    p1s = np.empty(num_trajectories * 2)

    for i in range(num_trajectories):
        if not np.isnan(rs[i]):
            r0s[2 * i] = rs[i]
            r0s[2 * i + 1] = rs[i]
            q10 = np.sqrt(rs[i])
            q1s[2 * i] = q10
            q1s[2 * i + 1] = q10
            p10 = C_transformed_momentum(q10, E)
            p1s[2 * i] = p10
            p1s[2 * i + 1] = -p10

    for i in range(num_trajectories * 2):
        array = C_poincare_angle_energy(E, F_0, Omg, section_points, q1s[i], p1s[i], q20 / Omg, h)
        # Find the index of the first row with all zeros
        first_zero_row = np.where(~array.any(axis=1))[0]
        # Trim the array up to the first row with all zeros
        arrays[i] = array[:first_zero_row[0]] if first_zero_row.size > 0 else array

    for array in arrays:
        data = np.append(data, array, axis=0)

    return data


#@njit(parallel=lambda: parallelization_depth >= 2, cache=False)
@njit(parallel=True)
def C_poincare_energies(Energies: np.ndarray, F_0: float, Omg: float, section_points: int, num_trajectories: int, t0: float = 0, dt: float = 1.e-4):

    all_results = []  # Will collect all arrays here
    
    for e in range(len(Energies)):
        E = Energies[e]

        r_min, r_max = C_return_points(E)

        angles = np.linspace(0, np.pi, num_trajectories)[1:num_trajectories-1]
        positions = np.array([ C_position( ang, E, r_min, r_max ) for ang in angles ])

        Num_conditions = int((num_trajectories-2)*2)
        r0s = np.empty(Num_conditions)
        q1s = np.empty(Num_conditions)
        p1s = np.empty(Num_conditions)

        for i in range(num_trajectories-2):
            if not np.isnan(positions[i]):
                r0s[2 * i] = positions[i]
                r0s[2 * i + 1] = positions[i]
                
                q10 = np.sqrt(positions[i])
                q1s[2 * i] = q10
                q1s[2 * i + 1] = q10
                
                p10 = C_transformed_momentum(q10, E)
                p1s[2 * i] = p10
                p1s[2 * i + 1] = -p10

        print("Condições iniciais calculadas")

        # Store results from parallel computation
        arrays = [np.empty((0, 2)) for _ in range(Num_conditions)]

        for i in prange(Num_conditions):
            arrays[i] = C_poincare_angle_energy(E, F_0, Omg, section_points, q1s[i], p1s[i], t0 / Omg, dt)

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