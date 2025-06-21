import numpy as np
from numba import njit, prange, vectorize

'''POTENCIAIS'''

@njit('f8(f8, f8)')
def V(alpha: float, r: float) -> float:

    """Calculates the 1D Morse-Coulomb potencial given the position r"""
    

    D = 1/alpha ; beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = -1/np.sqrt(r*r + alpha*alpha)
    else:
        pot = D*( np.exp( -2*beta*r) -2*np.exp( -beta*r ) )
    return pot

@njit('f8(f8, f8)')
def dV(alpha: float, r: float) -> float:
    
    """Calculates the first order derivative of V(r) at a given point r"""

    D = 1/alpha ; beta = 1/(alpha*np.sqrt(2))

    if r > 0:
        pot = r*(r*r + alpha*alpha)**(-3/2)
    else:
        pot = 2*D*beta*np.exp( -2*beta*r)*( np.exp( beta*r ) - 1 )
    return pot

@njit
def Vc(r: float) -> float:

    """Calculates the 1D Coulomb potencial given the position r"""

    if r > 0:
        pot = -1/r
    else:
        pot = np.inf
    return pot


'''MOMENTOS'''

@njit('f8(f8, f8, f8)')
def MC_bound_linear_momentum(alpha: float, E: float, r: float) -> float:

    """Calculates the Momentum of a bound particle given its position and total Energy"""
    

    return ( np.sqrt(2*(E - V(alpha, r))) )

@njit
def C_unforced_transformed_canon_momentum(q1: float, Ei: float):
  
    K = 4
    p2 = -Ei

    radicand = 2*(K-4*p2*q1**2)

    if radicand >=0:
        p1 = np.sqrt( radicand )
    elif -radicand < 1.e-10: #estava retornando valor invalido quando q1 era √2, radicando dava -1.e-15
        p1 = 0

    return p1


'''SISTEMAS DE EDOS'''

@njit('f8(f8, f8, f8)')
def external_field( F_0: float, AngFrequency: float, t: float ) -> float:

    """ External periodic force aplied to a particle """

    return F_0*np.cos( AngFrequency*t )

@njit('f8[:](f8, f8, f8[:], f8, f8)')
def MC_forced_diff_system(alpha: float, t: float, X: np.ndarray, F_0: float, AngFrequency: float) -> np.ndarray:

    """Differencial Equations system whose solution describes motion of a particle in the Morse-Coulomb potential."""


    r, p = X[0], X[1]

    f1 = p
    f2 = - dV(alpha, r) - F_0*np.cos( AngFrequency*t )

    return np.array([ f1, f2 ])

@njit
def C_transformed_forced_diff_system(tau: float, Q: np.ndarray, F_0: float, AngFrequency: float, m: float = 1.) -> np.ndarray:

    """Differencial Equations system whose solution is the trajectory of a particle in extended phase space in Coulomb potential."""


    q1, p1, q2, p2 = Q[0], Q[1], Q[2], Q[3]

    f1 = p1/m
    f2 = -8*q1*p2 - 16*(q1**3)*F_0*np.cos(AngFrequency*q2)
    f3 = 4*q1**2
    f4 = 4*(q1**4)*F_0*AngFrequency*np.sin(AngFrequency*q2)

    return np.array([ f1, f2, f3, f4 ])


'''RUNGE-KUTTA'''

@njit
def C_transformed_forced_runge_kutta_4(tau: float, Q: np.ndarray, F_0: float, FieldFrequency: float, h:float = 1.e-4) -> np.ndarray:

    """Order 4 Runge-Kutta method for solving differencial equation systems"""
    

    k1 = C_transformed_forced_diff_system(tau,         Q,            F_0, FieldFrequency)
    k2 = C_transformed_forced_diff_system(tau + 0.5*h, Q + 0.5*h*k1, F_0, FieldFrequency)
    k3 = C_transformed_forced_diff_system(tau + 0.5*h, Q + 0.5*h*k2, F_0, FieldFrequency)
    k4 = C_transformed_forced_diff_system(tau + h,     Q + h*k3,     F_0, FieldFrequency)

    return ( Q + (h/6)*(k1 + 2*k2 + 2*k3 + k4) )

@njit('f8[:](f8, f8, f8[:], f8, f8, f8)')
def MC_forced_runge_kutta_4(alpha: float, t: float, X: np.ndarray, F_0: float, FieldFrequency: float, h:float = 1.e-4) -> np.ndarray:

    """Order 4 Runge-Kutta method for solving differencial equation systems"""
    

    k1 = MC_forced_diff_system(alpha, t,         X,            F_0, FieldFrequency)
    k2 = MC_forced_diff_system(alpha, t + 0.5*h, X + 0.5*h*k1, F_0, FieldFrequency)
    k3 = MC_forced_diff_system(alpha, t + 0.5*h, X + 0.5*h*k2, F_0, FieldFrequency)
    k4 = MC_forced_diff_system(alpha, t + h,     X + h*k3,     F_0, FieldFrequency)

    return ( X + (h/6)*(k1 + 2*k2 + 2*k3 + k4) )

@njit('f8[:](f8, f8)')
def return_points(alpha: float, E: float) -> np.ndarray:

    """Calculates the return points of a particle based on alpha and the total Energy"""
    
    rm = -alpha*np.sqrt(2)*np.log( np.sqrt( alpha*E + 1 ) + 1 )
    rM = np.sqrt( 1/(E**2) - alpha**2 )
    
    return np.array([rm, rM])

def MC_phase_space( alpha: float, E: float, h: float = 1.e-4, rM: float = 0. ) -> np.ndarray:
    
    """Calculates the Mourse-Coulomb phase-space using momentum"""


    #function's parameters

    if E < 0:
        rm, rM = return_points( alpha, E )
    else:
        rm = return_points( alpha, E )[0]

    rs = np.arange( rm, rM, h )
    ps = np.zeros( len(rs) )
    ps[1:len(ps)-1] = np.array( [MC_bound_linear_momentum( alpha, E, r ) for r in np.arange( rm+h, rM-h, h )] )

    rs = np.append( rs, np.flip(rs) )
    ps = np.append( ps, -np.flip(ps) )

    X = np.column_stack( (rs, ps) )                             #array where position and momentum whill be stored

    return X

'''ESPAÇO DE FASES DOS SISTEMAS'''

@njit('f8[:,:](f8, f8, f8, f8, f8, f8, f8, f8, f8)')
def MC_forced_Phase_Space(alpha: float, E_0: float, r_0: float, F_0: float, Omg: float, t_0: float, total_time: float, h: float = 1.e-5, dt: float = 1.e-3 ) -> np.ndarray:
    
    N = int(total_time/dt)+1
    
    p_0 = -MC_bound_linear_momentum(alpha, E_0, r_0)

    X = np.empty( 2 )
    Y = np.empty( (N, 2) )

    X = np.array([ r_0, p_0 ])
    Y[0] = X

    i = 0
    j=0
    t = t_0
    while t < total_time:
        i += 1
        t += h
        X = MC_forced_runge_kutta_4( alpha, t, X, F_0, Omg, h )

        if t >= t_0+j*dt:
            Y[j] = np.array([ X[0], X[1] ])
            j += 1

    Y = Y[:j]

    return Y

@njit
def C_forced_Phase_Space(E_0: float, r_0: float, F_0: float, Omg: float, t_0: float, total_time: float, h: float = 1.e-5, dt: float = 1.e-3 ) -> tuple:
    N = int(total_time/dt)
    
    q1_0 = np.sqrt(r_0)
    p1_0 = C_unforced_transformed_canon_momentum(q1_0, E_0)
    q2_0 = t_0
    p2_0 = -E_0

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
        Q = C_transformed_forced_runge_kutta_4( tau, Q, F_0, Omg, h )
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


'''ENERGIA TOTAL'''

@njit('f8(f8, f8)')
def C_total_energy( r, p ):

    return (p**2)/2 + Vc(r)

@njit('f8(f8, f8, f8)')
def MC_total_energy( alpha: float, r: float, p: float ) -> float:

    return (p**2)/2 + V(alpha, r)

@njit(parallel=True)
def MC_ionization_probability(alpha: float, E_0: float, F_0: float, Omg: float, r0s: np.ndarray, p0s: np.ndarray, t_0: float, total_time: float, h: float = 1.e-5) -> float:

    rM = return_points(alpha, E_0)[1]
    Num_conditions = len(r0s)
    ionized = 0
    for n in prange(0, Num_conditions):

        X = np.array([ r0s[n], p0s[n] ])

        t = t_0
        while t < total_time:
            t += h
            X = MC_forced_runge_kutta_4( alpha, t, X, F_0, Omg, h )
    
            if ( X[0] > 8*rM ):
                if ( MC_total_energy(alpha, X[0], X[1]) > 0 ): 
                    ionized += 1
                    break

    P_ion = ionized/Num_conditions

    return P_ion

@njit(parallel=True)
def C_ionization_probability(E_0: float, F_0: float, Omg: float, q1s: np.ndarray, p1s: np.ndarray, t_0: float, total_time: float, h: float = 1.e-5) -> float:

    rM = return_points(0, E_0)[1]
    Num_conditions = len(q1s)
    ionized = 0
    #togo = Num_conditions
    for n in prange(0, Num_conditions):

        Q = np.array([ q1s[n], p1s[n], t_0, -E_0 ])

        tau = t_0
        while Q[2] < total_time:
            tau += h
            Q = C_transformed_forced_runge_kutta_4( tau, Q, F_0, Omg, h )
    
            if ( Q[0]**2 > 8*rM ):
                if ( C_total_energy( Q[0]**2, Q[1]/(2*Q[0]) ) > 0 ): 
                    ionized += 1
                    break

    P_ion = ionized/Num_conditions

    return P_ion


@njit('f8[:](f8, f8, i8)')
def Chebyshev_nodes( a: float, b:float, N: int ) -> np.ndarray:
    
    delta_theta = np.pi/N
    thetas = np.array( [i*delta_theta for i in range(N+1)] )

    r = (b-a)/2
    mu = (b+a)/2

    nodes = mu + r*np.cos(thetas)

    return nodes

@njit('f8(f8, i8, f8[:,:])')
def Lagrange_polynomial(x: float, j: int, points: np.ndarray) -> float:
    n = len(points)
    res = 1
    for m in range(n):
        if m != j:
            res *= (x - points[m, 0])/(points[j, 0] - points[m, 0])
    return res

@njit('f8(f8, f8[:, :])')
def Polynomial_fit(r: float, points: np.ndarray) -> float:
    n = len(points)
    res = 0
    for j in range(n):
        res += (points[j, 1])*Lagrange_polynomial(r, j, points)
    return res
VEC_P = vectorize(Polynomial_fit)

@njit('f8(f8, f8[:, :], f8, f8, f8, i8)')
def bisection_method( angle: float, points: np.ndarray, a: float, b: float, tol: float = 1.e-7, max_iter:int = 1000) -> float:

    f = lambda r : Polynomial_fit(r, points) - angle

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
    #print( "root found", iterations)
    
    return x0

@njit('f8(f8, f8, f8, f8, f8)')
def MC_unforced_angle( alpha: float, Ei: float, r: float, omg_n: float, dr: float = 1.e-5 ) -> float:

    rm = return_points( alpha, Ei )[0]

    n = int( (r - rm)/dr )

    sum1 = sum2 = 0

    for i in range(1, n):
        ri = rm + i * dr
        if i % 2 == 0:
            sum2 += ( 2*(Ei - V(alpha, ri)) )**(-1/2)
        else:
            sum1 += ( 2*(Ei - V(alpha, ri)) )**(-1/2)

    integral = (dr / 3) * ( 2 * sum2 + 4 * sum1 )

    #print("angle calculated")

    return omg_n*integral


@njit('f8(f8, f8, f8)')
def MC_classic_action( alpha: float, Ei: float, dr: float = 1.e-6 ) -> float:
    
    """Calculates the action of a bound particle in the Morse-Coulomb potential"""


    #function's parameters

    rm, rM = return_points( alpha, Ei )

    n = int( (rM - rm)/dr )


    sum1 = sum2 = 0

    for i in range(1, n):
        r = rm + i * dr
        if i % 2 == 0:
            sum2 += MC_bound_linear_momentum( alpha, Ei, r )
        else:
            sum1 += MC_bound_linear_momentum( alpha, Ei, r )
    integral = (dr / 3) * ( 2 * sum2 + 4 * sum1 )

    return integral/np.pi


@njit('f8(f8, f8, f8)')
def MC_angular_frequency( alpha: float, Ei: float, dE: float = 1.e-4 ) -> float:
    
    """Calculates the agular frequency of a particle in the Morse-Coulomb potential derivating action"""


    return ( (1/(12*dE))*( MC_classic_action(alpha, Ei-2*dE, dE ) 
                          - 8*MC_classic_action(alpha, Ei-dE, dE ) 
                          + 8*MC_classic_action(alpha, Ei+dE, dE ) 
                          - MC_classic_action(alpha, Ei+2*dE, dE ) 
                          ) 
            )**(-1)

@njit
def check_no_duplicates(arr):
    # Convert the array to a set
    unique_elements = set(arr)
    
    # Compare the lengths
    return len(unique_elements) == len(arr)

@njit('f8[:](f8[:], f8, f8, i8, f8)')
def MC_unforced_position( angles: np.ndarray, alpha: float, Ei: float, N: int, dr: float = 1.e-5 ) -> np.ndarray:

    success = False

    while success == False:
        rmin_mc, rmax_mc = return_points(alpha, Ei)                                         #pontos de retorno
        rs_mc = np.array(sorted( Chebyshev_nodes( 0, rmax_mc, N ) ))                                #distribuicao de chebyshev para evitar fenomeno de runge

        omg_n = MC_angular_frequency(alpha, Ei, dr)

        thetas_mc = np.array( [ MC_unforced_angle( alpha, Ei, r, omg_n, 1.e-6 ) for r in rs_mc ] )

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

@njit( 'f8(f8)' )
def C_classic_action(E: float) -> float:

    """Calculates the classic action of a particlein Coulomb potential"""

    J = 1/np.sqrt( -2*E )
    return J

@njit
def C_unforced_angle( Ei, r ):

    theta = 2*( np.arcsin( np.sqrt( np.abs(Ei)*r ) ) - np.sqrt( np.abs(Ei)*r )*np.sqrt( 1 - np.abs(Ei)*r ) )
    return theta


@njit
def C_r_from_theta( angle: float, E0: float, a: float, b: float, tol: float = 1.e-8, max_iter:int = 1000):

    f = lambda r : C_unforced_angle(E0, r) - angle

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


@njit #(parallel=True)
def MC_Ionization_Amplitude( alpha: float, E_0: float, F_0s: np.ndarray, Omg: float, Num_trajectories: int, t_0: float, total_time: float, poly_degree: int, h: float = 1.e-4 ):
    
    omg_n = MC_angular_frequency(alpha, E_0, h)
    theta_0 = MC_unforced_angle(alpha, E_0, 0, omg_n, 1.e-5)

    angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = MC_unforced_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = MC_bound_linear_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = { }

    for f in range(len(F_0s)):
        Pi = MC_ionization_probability( alpha, E_0, F_0s[f], Omg, r0s, p0s, t_0, total_time, h )
        Pis[F_0s[f]] = Pi

    return sorted(Pis.items())


@njit #(parallel=True)
def MC_Ionization_Frequency( alpha: float, E_0: float, F_0: float, Omgs: np.ndarray, Num_trajectories: int, total_time: float, poly_degree: int, h: float = 1.e-4 ):
    
    omg_n = MC_angular_frequency(alpha, E_0, h)
    theta_0 = MC_unforced_angle(alpha, E_0, 0, omg_n, 1.e-5)

    angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]

    positions = MC_unforced_position(angles, alpha, E_0, poly_degree, 1.e-5)

    Num_conditions = int((Num_trajectories-2)*2)
    r0s = np.empty( Num_conditions )
    p0s = np.empty( Num_conditions )


    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        r0s[2*i] = r_0
        r0s[2*i+1] = r_0

        #momento 
        p_0 = MC_bound_linear_momentum(alpha, E_0, r_0)
        p0s[2*i] = p_0 
        p0s[2*i+1] = -p_0

    print("Condições iniciais calculadas")

    Pis = {}

    for o in range(len(Omgs)):
        Omg = Omgs[o]
        t_0 = np.pi/(2*Omg)
        Pi = MC_ionization_probability( alpha, E_0, F_0, Omg, r0s, p0s, t_0, total_time, h )
        Pis[Omg] = Pi

    return sorted(Pis.items())

@njit #(parallel=True)
def MC_Ionization_Alphas( alphas: np.ndarray, E_0: float, F_0: float, Omg: float, Num_trajectories: int, t_0: float, total_time: float, poly_degree: int, h: float = 1.e-4 ):


    Pis = {}

    for alpha in alphas:

        omg_n = MC_angular_frequency(alpha, E_0, h)
        theta_0 = MC_unforced_angle(alpha, E_0, 0, omg_n, 1.e-5)
    
        angles = np.linspace( theta_0, np.pi, Num_trajectories )[1:Num_trajectories-1]
    
        positions = MC_unforced_position(angles, alpha, E_0, poly_degree, 1.e-5)
    
        Num_conditions = int((Num_trajectories-2)*2)
        r0s = np.empty( Num_conditions )
        p0s = np.empty( Num_conditions )
    
    
        for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
            r_0 = positions[i]
            #posicao
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0
    
            #momento 
            p_0 = MC_bound_linear_momentum(alpha, E_0, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0
    
        print("Condições iniciais calculadas")
        Pi = MC_ionization_probability( alpha, E_0, F_0, Omg, r0s, p0s, t_0, total_time, h )
        Pis[alpha] = Pi

    return sorted(Pis.items())


@njit #(parallel=True)
def C_Ionization_Amplitude( E_0: float, F_0s: np.ndarray, Omg: float, Num_trajectories: int, t_0: float, total_time: float, h: float = 1.e-4 ):
    
    rmin_C, rmax_C = return_points(0, E_0)
    angles = np.linspace(0, np.pi, Num_trajectories)[1:Num_trajectories-1]
    positions = np.array( [ C_r_from_theta( ang, E_0, rmin_C, rmax_C ) for ang in angles ] )
    
    Num_conditions = int((Num_trajectories-2)*2)
    q1s = np.empty( Num_conditions )
    p1s = np.empty( Num_conditions )

    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        q1s[2*i] = np.sqrt(r_0)
        q1s[2*i+1] = np.sqrt(r_0)

        #momento 
        p1 = C_unforced_transformed_canon_momentum( np.sqrt(r_0), E_0 )
        p1s[2*i] = p1 
        p1s[2*i+1] = -p1

    print("Condições iniciais calculadas")

    Pis = { }
    
    for f in range(len(F_0s)):
        Pi = C_ionization_probability( E_0, F_0s[f], Omg, q1s, p1s, t_0, total_time, h )
        Pis[F_0s[f]] = Pi

    return sorted(Pis.items())

@njit #(parallel=True)
def C_Ionization_Frequency( E_0: float, F_0: float, Omgs: np.ndarray, Num_trajectories: int, total_time: float, h: float = 1.e-4 ):
    
    rmin_C, rmax_C = return_points(0, E_0)
    angles = np.linspace(0, np.pi, Num_trajectories)[1:Num_trajectories-1]
    positions = np.array( [ C_r_from_theta( ang, E_0, rmin_C, rmax_C ) for ang in angles ] )
    
    Num_conditions = int((Num_trajectories-2)*2)
    q1s = np.empty( Num_conditions )
    p1s = np.empty( Num_conditions )

    for i in range(Num_trajectories-2):               #duplicando o numero de condicoes iniciais para p e -p
        r_0 = positions[i]
        #posicao
        q1s[2*i] = np.sqrt(r_0)
        q1s[2*i+1] = np.sqrt(r_0)

        #momento 
        p1 = C_unforced_transformed_canon_momentum( np.sqrt(r_0), E_0 )
        p1s[2*i] = p1 
        p1s[2*i+1] = -p1
    
    print("Condições iniciais calculadas")

    Pis = {}

    for o in range(len(Omgs)):
        Omg = Omgs[o]
        t_0 = np.pi/(2*Omg)
        Pi = C_ionization_probability( E_0, F_0, Omg, q1s, p1s, t_0, total_time, h )
        Pis[Omg] = Pi

    return sorted(Pis.items())

@njit
def MC_test_iterations( alpha, E, F, Omg, tot, r0, h ):
    p_0 = MC_bound_linear_momentum(alpha, E, r0)

    X = np.empty( 2 )

    X = np.array([ r0, p_0 ])

    t = np.pi/2
    for i in range(tot):
        t += h
        X = MC_forced_runge_kutta_4( alpha, t, X, F, Omg, h )

@njit
def C_test_iterations( E, F, Omg, tot, r0, h ):
    q1_0 = np.sqrt(r0)
    p1_0 = C_unforced_transformed_canon_momentum(q1_0, E)
    q2_0 = np.pi/2
    p2_0 = -E

    p_0 = p1_0/(2*q1_0)

    Q = np.empty( 4 )
    Q = np.array([ q1_0, p1_0, q2_0, p2_0 ])
    tau = q2_0
    X = np.array([ r0, p_0 ])
    for i in range(tot):
        tau += h
        Q = C_transformed_forced_runge_kutta_4( tau, Q, F, Omg, h )

        r = Q[0]**2 ; p = Q[1]/(2*Q[0])
        X = np.array([ r, p ])


@njit
def Coulomb_time_step(E_0: float, F_0: float, Omg: float, h: float, total_time: float):
    print("Inside")

    q1_0 = 1.
    p1_0 = C_unforced_transformed_canon_momentum(q1_0, E_0)
    q2_0 = np.pi/(2*Omg)
    p2_0 = -E_0

    #tm_steps = np.empty(int(N))
    time_arr = np.empty(int(4e7))


    Q = np.array([ q1_0, p1_0, q2_0, p2_0 ])
    tau = q2_0

    it = 0
    while Q[2] < total_time:
        it += 1
        tau += h
        Q = C_transformed_forced_runge_kutta_4( tau, Q, F_0, Omg, h )
        time_arr[it] = Q[2]
    print(it)
    return time_arr[:it]


@njit #(parallel=True)
def MC_Ionization_Trajectories( alpha: float, E_0: float, F_0: float, Omg: float, Num_trajectories: np.ndarray, t_0: float, total_time: float, h: float = 1.e-4 ):
    
    omg_n = MC_angular_frequency(alpha, E_0, h)
    theta_0 = MC_unforced_angle(alpha, E_0, 0, omg_n, 1.e-5)



    Pis = { }
    for n in range(len(Num_trajectories)):
        angles = np.linspace( theta_0, np.pi, Num_trajectories[n] )[1:Num_trajectories[n]-1]

        positions = MC_unforced_position(angles, alpha, E_0, Num_trajectories[n], 1.e-5)

        Num_conditions = int((Num_trajectories[n]-2)*2)
        r0s = np.empty( Num_conditions )
        p0s = np.empty( Num_conditions )


        for i in range(Num_trajectories[n]-2):               #duplicando o numero de condicoes iniciais para p e -p
            r_0 = positions[i]
            #posicao
            r0s[2*i] = r_0
            r0s[2*i+1] = r_0

            #momento 
            p_0 = MC_bound_linear_momentum(alpha, E_0, r_0)
            p0s[2*i] = p_0 
            p0s[2*i+1] = -p_0

        print("Condições iniciais calculadas")


        Pi = MC_ionization_probability( alpha, E_0, F_0, Omg, r0s, p0s, t_0, total_time, h )
        Pis[Num_conditions] = Pi

    return sorted(Pis.items())