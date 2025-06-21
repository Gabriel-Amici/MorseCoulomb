import numpy as np
from numba import njit
from ..potentials.normalized_msc_potential import nMsC_potential, nMsC_potential_1st_derivative, nMsC_return_points
from ..utils import inter_period, chebyshev_nodes, check_no_duplicates, bisection_method

@njit
def nMsC_momentum(alpha: float, E: float, r: float) -> float:

    """Calculates the Momentum of a bound particle given its position and total Energy"""
    

    return ( np.sqrt(2*(E - nMsC_potential(alpha, r))) )


@njit
def nMsC_diff_system(alpha: float, t: float, X: np.ndarray) -> np.ndarray:

    """ Differencial Equation system that describes bound motion of a particle in the Morse-Coulomb potential."""


    r, p = X[0], X[1]

    f1 = p
    f2 = - nMsC_potential_1st_derivative(alpha, r)

    return np.array([ f1, f2 ])


@njit
def runge_kuta_4(alpha: float, t: float, X: np.ndarray, dt: float = 1.e-4) -> np.ndarray:

    """Order 4 Runge-Kutta method for solving differencial equation systems"""


    k1 = nMsC_diff_system(alpha, t, X)
    k2 = nMsC_diff_system(alpha, t + (1/2)*dt, X + (1/2)*dt*k1)
    k3 = nMsC_diff_system(alpha, t + (1/2)*dt, X + (1/2)*dt*k2)
    k4 = nMsC_diff_system(alpha, t + dt, X + dt*k3)

    return ( X + (dt/6)*(k1 + 2*k2 + 2*k3 + k4) )


@njit 
def nMsC_bound_solution( alpha: float, r0: float, E: float, tot_time: float, m: float = 1, dt: float = 1.e-4, one_cycle: bool = False ):
    """Solves the differential equation system for a particle in the Morse-Coulomb potential"""


    #function's parameters

    crss = 0


    #systems initial conditions

    n = int(tot_time/dt) ; t = np.linspace(0, tot_time, n)       #time array

    p0 = nMsC_momentum(alpha, E, r0)                 #momentum at the start (x0 and t0) 

    X = np.empty(( n, 2 )) ; X[0] = np.array([ r0, p0 ])        #array where position and momentum whill be stored
    period = 0

    #main loop

    i = 0
    while i < n - 1:                                            #loops until we run out of time

        X[i+1] = np.array([runge_kuta_4(alpha, t[i], X[i])])

        if ( (X[i+1, 0] - r0)*(X[i, 0] - r0) < 0 ):
            crss += 1
        
        if crss == 2:

            #get values to find the period

            
            if period == 0:
                xi1 = X[i+1, 0] ;   xi = X[i, 0]
                ti = t[i]

                period = inter_period( xi1, xi, r0, ti, dt )

            if one_cycle:
            
                #trim the vectors

                X = np.copy(X[0:i+1,:])
                t = np.copy(t[0:i+1])
                break

        i += 1

    
    return X, t, period


@njit
def nMsC_phase_space( alpha: float, E: float, dt: float = 1.e-4, rM: float = 0. ) -> np.ndarray:
    
    """Calculates the Mourse-Coulomb phase-space using momentum"""


    #function's parameters

    if E < 0:
        rm, rM = nMsC_return_points( alpha, E )
    else:
        rm = nMsC_return_points( alpha, E )[0]

    rs = np.arange( rm, rM, dt )
    ps = np.zeros_like(rs)
    ps[1:len(ps)-1] = np.array([ nMsC_momentum(alpha, E, r) for r in np.arange( rm+dt, rM-dt, dt )] )

    rs = np.append( rs, np.flip(rs) )
    ps = np.append( ps, -np.flip(ps) )

    X = np.column_stack( (rs, ps) )                             #array where position and momentum whill be stored

    return X


@njit
def nMsC_action( alpha: float, E: float, dr: float = 1.e-6 ) -> float:
    
    """Calculates the action of a bound particle in the Morse-Coulomb potential"""


    #function's parameters

    rm, rM = nMsC_return_points( alpha, E )

    n = int( (rM - rm)/dr )


    sum1 = sum2 = 0

    for i in range(1, n):
        r = rm + i * dr
        if i % 2 == 0:
            sum2 += nMsC_momentum( alpha, E, r )
        else:
            sum1 += nMsC_momentum( alpha, E, r )
    integral = (dr / 3) * ( 2 * sum2 + 4 * sum1 )

    return integral/np.pi


@njit
def nMsC_angular_frequency( alpha: float, E: float, dE: float = 1.e-4 ) -> float:
    
    """Calculates the agular frequency of a particle in the Morse-Coulomb potential derivating action"""


    return ( (1/(12*dE))*( nMsC_action(alpha, E-2*dE, dE ) 
                          - 8*nMsC_action(alpha, E-dE, dE ) 
                          + 8*nMsC_action(alpha, E+dE, dE ) 
                          - nMsC_action(alpha, E+2*dE, dE ) 
                          ) 
            )**(-1)


@njit
def nMsC_angle( alpha: float, E: float, r: float, omg_n: float, dr: float = 1.e-5 ) -> float:

    rm = nMsC_return_points( alpha, E )[0]

    n = int( (r - rm)/dr )

    sum1 = sum2 = 0

    for i in range(1, n):
        ri = rm + i * dr
        if i % 2 == 0:
            sum2 += ( 2*(E - nMsC_potential(alpha, ri)) )**(-1/2)
        else:
            sum1 += ( 2*(E - nMsC_potential(alpha, ri)) )**(-1/2)

    integral = (dr / 3) * ( 2 * sum2 + 4 * sum1 )

    #print("angle calculated")

    return omg_n*integral


@njit
def nMsC_position( angles: np.ndarray, alpha: float, Ei: float, N: int, dr: float = 1.e-5 ) -> np.ndarray:

    success = False

    while success == False:
        rmin_mc, rmax_mc = nMsC_return_points(alpha, Ei)                                         #pontos de retorno
        rs_mc = np.array(sorted( chebyshev_nodes( rmin_mc, rmax_mc, N ) ))                                #distribuicao de chebyshev para evitar fenomeno de runge

        omg_n = nMsC_angular_frequency(alpha, Ei, dr)

        thetas_mc = np.array( [ nMsC_angle( alpha, Ei, r, omg_n, 1.e-6 ) for r in rs_mc ] )

        points = np.empty( (N, 2) )

        for i in range(N):
            points[i, 0] = rs_mc[i] ; points[i, 1] = thetas_mc[i]

        rs = np.array( [ bisection_method( ang, points, rmin_mc, rmax_mc, 1.e-8, 1000 ) for ang in angles ] ) #calculo das posicoes a partir dos angulos
        
        if check_no_duplicates(rs):
            success = True
        else:
            N += 10
        
    #print(N)
    return rs