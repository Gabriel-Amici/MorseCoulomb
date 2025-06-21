import numpy as np
from numba import njit, vectorize
from ..potentials.coulomb_potential import C_potential, C_transformed_return_points


@njit
def C_momentum(E: float, r: float) -> float:

    """Calculates the Momentum of a particle in Coulomb potential given its position (by means of potential) and total Energy"""

    return ( np.sqrt(2*(E - C_potential(r))) )


@njit
def C_transformed_momentum(q1: float, E: float):
  
    K = 4
    p2 = -E

    radicand = 2*(K-4*p2*q1**2)

    if radicand >=0:
        p1 = np.sqrt( radicand )
    elif np.abs(radicand) < 1.e-5: #estava retornando valor invalido quando q1 era √2, radicando dava -1.e-15
        p1 = 0

    return p1


@njit
def C_action(E: float) -> float:

    """Calculates the classic action of a particlein Coulomb potential"""

    J = 1/np.sqrt( -2*E )
    return J
VEC_C_action = vectorize(C_action)


@njit
def C_angular_frequency(E: float) -> float:
    
    """Calculates the angular frequency of a particlein Coulomb potential"""

    omg = 1/C_action(E)**3
    return omg
VEC_C_angular_frequency = vectorize(C_angular_frequency)


@njit
def C_transformed_phase_space(E: float, dq: float = 1.e-5) -> np.ndarray:
   
    """Calculates the expanded Coulomb phase-space"""


    #function's parameters

    if E < 0:
        q1m, q1M = C_transformed_return_points(E)

    q1s = np.arange( q1m, q1M, dq )
    p1s = np.array([ C_transformed_momentum(q1, E) for q1 in q1s ])

    q1s = np.append( q1s, np.flip(q1s) )
    p1s = np.append( p1s, -np.flip(p1s) )


    Q = np.column_stack( (q1s, p1s) ) 

    return Q


@njit
def C_phase_space(E: float):

    Q = C_transformed_phase_space(E)

    X = np.empty((Q.shape))

    for i in range(len(X)):
        q1, p1, = Q[i, 0], Q[i, 1]
        if q1 != 0:
            X[i] = np.array( [ q1**2, p1/(2*q1) ] )
        else:
            X[i] = np.inf

    return X


@njit
def C_angle( E: float, r: float ) -> float:

    theta = 2*( np.arcsin( np.sqrt( np.abs(E)*r ) ) - np.sqrt( np.abs(E)*r )*np.sqrt( 1 - np.abs(E)*r ) )
    return theta


@njit
def C_position( angle: float, E0: float, a: float, b: float, tol: float = 1.e-8, max_iter:int = 1000):

    f = lambda r : C_angle(E0, r) - angle

    if abs( f(a) ) < tol:
        return a
    elif abs( f(b) ) < tol:
        return b

    iterations = 0

    while (b - a) / 2 > tol and iterations < max_iter:
        c = (a + b) / 2
        if abs( f(c) ) < tol:
            return c

        if f(c) * f(a) < 0:
            b = c
        else:
            a = c

        iterations += 1
    x0 = (a + b) / 2
    return x0