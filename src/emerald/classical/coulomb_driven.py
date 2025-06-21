import numpy as np
from numba import njit
from .coulomb_unperturbed import C_transformed_momentum

@njit
def C_transformed_diff_system(tau: float, Q: np.ndarray, F_0: float, Omg: float) -> np.ndarray:

    """Differencial Equations system whose solution is the trajectory of a particle in extended phase space in Coulomb potential."""


    q1, p1, q2, p2 = Q[0], Q[1], Q[2], Q[3]

    f1 = p1
    f2 = -8*q1*p2 - 16*(q1**3)*F_0*np.cos(Omg*q2)
    f3 = 4*q1**2
    f4 = 4*(q1**4)*F_0*Omg*np.sin(Omg*q2)

    return np.array([ f1, f2, f3, f4 ])

@njit
def C_transformed_runge_kutta_4(tau: float, Q: np.ndarray, F_0: float, Omg: float, h:float = 1.e-4) -> np.ndarray:

    """Order 4 Runge-Kutta method for solving differencial equation systems"""
    

    k1 = C_transformed_diff_system(tau,         Q,            F_0, Omg)
    k2 = C_transformed_diff_system(tau + 0.5*h, Q + 0.5*h*k1, F_0, Omg)
    k3 = C_transformed_diff_system(tau + 0.5*h, Q + 0.5*h*k2, F_0, Omg)
    k4 = C_transformed_diff_system(tau + h,     Q + h*k3,     F_0, Omg)

    return ( Q + (h/6)*(k1 + 2*k2 + 2*k3 + k4) )


@njit
def C_forced_Phase_Space(E: float, r_0: float, F_0: float, Omg: float, t_0: float, total_time: float, h: float = 1.e-5, dt: float = 1.e-3 ) -> tuple:
    N = int(total_time/dt)
    
    q1_0 = np.sqrt(r_0)
    p1_0 = C_transformed_momentum(q1_0, E)
    q2_0 = t_0
    p2_0 = -E

    p_0 = p1_0/(2*q1_0)

    Q = np.empty( 4 )
    Y = np.empty( (N, 2) )

    Q = np.array([ q1_0, p1_0, q2_0, p2_0 ])
    Y[0] = np.array([ r_0, p_0 ])

    time_array=np.empty(0)

    i = 0
    j=0
    tau = t_0
    while Q[2] < total_time:
        tau += h
        Q = C_transformed_runge_kutta_4( tau, Q, F_0, Omg, h )
        if (i == 0):
            t1 = Q[2]

        if (i == 1):
            print(Q[2] - t1)
        i += 1

        if Q[2] >= t_0+j*dt:
            r = Q[0]**2 ; p = Q[1]/(2*Q[0])
            Y[j] = np.array([ r, p ])
            time_array = np.append( time_array, Q[2] )
            j += 1

    Y = Y[:j]

    return Y, time_array

