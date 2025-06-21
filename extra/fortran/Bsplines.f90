    module gaussm3

    ! Common Global variables within module !
    implicit none
    INTEGER, parameter :: dbp = SELECTED_REAL_KIND (15,307)
    !   PRIVATE
    !  REAL*8 (dbp) :: newv
    !  REAL*8(dbp)  :: EPS, M_PI
    real*8,PARAMETER:: EPS=3.0d-15       	!EPS is the relative precision
    real*8,PARAMETER:: M_PI=3.141592654d0      ! Pi value

    !   PUBLIC :: newv, EPS, M_PI, n, xabsc, weig, dbp, qgss2d

    INTERFACE

    END INTERFACE
    CONTAINS
    !* This module has the following INTERNAL FUNCTIONS:
    !* gauleg, qgauss, qgss3d, qgss2d, gsselm, identity_matrix
    !* This module has the following INTERNAL SUBROUTINES:
    !* linear_solver
    !* They can call on each other without first specifying their type
    !* NO INTERFACE nor EXTERNAL is required since they are INTERNAL functions

    !********************************************************************************
    !* Calculation of GAUSS-LEGENDRE abscissas and weights for Gaussian Quadrature
    !* integration of polynomial functions.
    !*      For normalized lower and upper limits of integration -1.0 & 1.0, and
    !* given n, this routine calculates, arrays xabsc(1:n) and  weig(1:n) of length n,
    !* containing the abscissas and weights of the Gauss-Legendre n-point quadrature
    !* formula.  For detailed explanations finding weights & abscissas, see
    !* "Numerical Recipes in Fortran */
    !********************************************************************************
    !##############################################################
    SUBROUTINE  gauleg(ngp, xabsc, weig)

    implicit none
    INTEGER  i, j, m
    REAL*8::  p1, p2, p3, pp, z, z1
    INTEGER, INTENT(IN) :: ngp            ! # of Gauss Points
    REAL*8,dimension(dbp), INTENT(OUT) :: xabsc(ngp), weig(ngp)


    m = (ngp + 1) / 2
    !* Roots are symmetric in the interval - so only need to find half of them  */

    do i = 1, m				! Loop over the desired roots */

        z = cos( M_PI * (i-0.25d0) / (ngp+0.5d0) )
        !*   Starting with the above approximation to the ith root,
        !*          we enter the main loop of refinement by NEWTON'S method   */
100     p1 = 1.0d0
        p2 = 0.0d0
        !*  Loop up the recurrence relation to get the Legendre
        !*  polynomial evaluated at z                 */

        do j = 1, ngp
            p3 = p2
            p2 = p1
            p1 = ((2.0d0*j-1.0d0) * z * p2 - (j-1.0d0)*p3) / j
        enddo

        !* p1 is now the desired Legendre polynomial. We next compute pp,
        !* its derivative, by a standard relation involving also p2, the
        !* polynomial of one lower order.      */
        pp = ngp*(z*p1-p2)/(z*z-1.0d0)
        z1 = z
        z = z1 - p1/pp             ! Newton's Method  */

        if (dabs(z-z1) .gt. EPS) GOTO  100

        xabsc(i) =  - z                    	! Roots will be bewteen -1.0 & 1.0 */
        xabsc(ngp+1-i) =  + z                	! and symmetric about the origin  */
        weig(i) = 2.0d0/((1.0d0-z*z)*pp*pp) ! Compute the weight and its       */
        weig(ngp+1-i) = weig(i)               ! symmetric counterpart         */

    end do     ! i loop

    End subroutine gauleg

    !********************************************************************************
    !*     Returns the SINGLE integral of the function (of ONE VARIABLE) "func"
    !* between x1 and x2 by N-point Gauss-Legendre integration. The function
    !* is evaluated exactly N times at interior points in the range of
    !* integration.       */
    !********************************************************************************
    recursive function qgauss(func, x1, x2, ngp) RESULT(intgrl)
    implicit none
    REAL(dbp)  intgrl, x1, x2, func
    REAL(dbp)  xm, xl
    INTEGER j
    INTEGER, INTENT(IN) :: ngp            ! # of Gauss Points
    REAL(dbp) :: xabsc(ngp), weig(ngp)

    call gauleg(ngp, xabsc, weig)

    intgrl = 0.0d0
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    do j = 1, ngp
        intgrl = intgrl + weig(j) * func( xm + xl*xabsc(j) )
    END do

    intgrl = intgrl * xl;    !Scale the answer to the range of integration  */
    END function qgauss


    recursive function qgaussmat1(func, x1, x2, ngp, frow, fcol) RESULT(intgrl)
    implicit none
    REAL(dbp), INTENT(IN) :: x1, x2
    INTEGER :: frow, fcol
    REAL(dbp) :: intgrl(frow, fcol), tmpm(frow,fcol)
    REAL(dbp) ::  func
    REAL(dbp) ::  xm, xl, arg
    INTEGER j

    INTEGER, INTENT(IN) :: ngp            ! # of Gauss Points
    REAL(dbp) :: xabsc(ngp), weig(ngp)

    call gauleg(ngp, xabsc, weig)

    intgrl(:,:) = 0.0d0
    tmpm(:,:) = 0.0d0
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    do j = 1, ngp
        arg =  xm + xl*xabsc(j)
        PRINT *, 'szhgd ds'
        PRINT *,arg
        tmpm = func(arg)
        intgrl = intgrl + weig(j) * tmpm
    END do

    intgrl = intgrl * xl;    !Scale the answer to the range of integration  */
    END function qgaussmat1

    end module gaussm3
    
    module bsplinesparameters
    integer ( kind = 4 ), parameter :: k = 15 ! B-spline order
    integer ( kind = 4 ), parameter :: n = 20000!9960 ! number of B-splines
    real ( kind = 8 ), dimension (n+k) :: t
    integer ( kind = 4 ) i,j,l
    real*8::alpha

    end module bsplinesparameters


    program eigen
 !   USE mkl95_LAPACK
    use gaussm3
    use bsplinesparameters

    implicit none
    ! parameters for the dsygv subroutine that solves the generalized eigenvalue problem
    integer ( kind = 4 ), parameter :: lwork=3*(n-2)-1
! eigenvalue equation: HP*Psi = E*S*Psi
    real*8, dimension(:,:),allocatable::S,S2,HP,HPG,HPE,D 
 !   real*8, dimension(n-2):: W,fv1,fv2
    real*8, dimension(:),allocatable:: W,fv1,fv2  
  !  real*8, dimension(lwork) :: work
    real*8, dimension(:),allocatable :: work
    integer :: ierr,matz,itype,info
    character*1 :: jobz,uplo

    integer:: m,inu
    real*8 :: Enu,r   

    real*8, parameter :: hwn=219474.63d0 ! one Hartree in wavenumber
    real*8, parameter :: rmax=2300.d0,gama=5.d0 ! grid parameters
    real*8::rmin
    integer,parameter :: Nle=20 ! Number of points for the Guass-Legendre quadrature

    integer,parameter :: nmax=1000 ! maximum number of energy levels
    integer, parameter:: lmax=1 ! number or excited electronic states
    real*8::normal,auxn
    real*8::wavefunc1,wavefunc2,wavefunc3,wavefunc4
    real*8::func,func2,func3,func4,VMC
    integer::mflag,left,kn
    integer( kind = 4 ), parameter :: indx = 1
    real ( kind = 8 ) values(n)
    real ( kind = 8 ), dimension (n+k) :: tgrid
        
    integer,parameter::nrgrid=20000
    real*8,parameter::deltarplot=0.1d0
    real*8,dimension(0:nrgrid)::points
    real*8::soma,term,time
    Character(4)::knstring2,knstring3
    integer::nalpha
    
    external func,func2,func3,func4
    
    ! allocation of large arrays
    allocate(S(n-2,n-2),S2(n-2,n-2),HP(n-2,n-2),HPG(n-2,n-2),HPE(n-2,n-2),D(n-2,n-2))
    allocate(W(n-2),fv1(n-2),fv2(n-2))
    allocate(work(lwork))
       
    
    write ( *, '(a)' ) ' '
    call timestamp ( )
    write ( *, '(a)' ) ' '
    write ( *, '(a)' ) '  Program for calculating eigenvalues and eigenvectors'
    write ( *, '(a)' ) '                      using B-splines'
    write ( *, '(a)' ) ' '
    !exponential breakpoint sequence
    
    do nalpha=0,50
        
    if (nalpha==0) then 
        alpha=0.001
    else
        alpha=nalpha*0.005d0
    endif    
        
    rmin=-alpha
    
    i=1
    t(1:k)=rmin
    write(20,*)rmin,i
    do i=2,n-k+1
        r=rmin+(rmax-rmin)*(dexp(gama*(i-1)/(n-1))-1.d0)/(dexp(gama)-1.d0)
        write(20,*)r,i
 !        write(46,*)r,VMC(alpha,r)
        t(i+k-1)=r
    enddo
  !  write(20,*)rmax,i
    
  !  stop
    
    t(n+1:n+k)=rmax
    
    tgrid=t
!Calculating the matrices for the generalized eigenvalue problem

        S=0.d0
        HP=0.d0  
        ! Ground State
        print*,"Ground state..."
        do i=2,n-1
            do j=i,min(i+k,n-1)
                do m=max(k,i),min(i+k,n)
                    t=tgrid
                    S(i-1,j-1)=S(i-1,j-1)+qgauss(func, t(m), t(m+1), Nle)
                    t=tgrid
                    HP(i-1,j-1)=HP(i-1,j-1)+qgauss(func2, t(m), t(m+1), Nle)
                enddo
            enddo
        enddo
        open(1,file='Sn1000L2300.txt')
        do i=2,n-1
            do j=i,min(i+k,n-1)
              !  write(1,*)S(i-1,j-1)
                S(j-1,i-1)=S(i-1,j-1)
            enddo
        enddo
        S2=S
        print*,"Matrix calculation for ground state completed..."
        print*
        call timestamp ( )
        print*
        print*,"Solving the generalized eigenvalue problem..."
        print*
        ! solving the generalized eigenvalue problem
        itype=1
        jobz='N'! N only eigenvalues, V eigenvectors
        uplo='U'
        call dsygv(itype, jobz, uplo, n-2, HP, n-2, S, n-2, w, work, lwork, info)
        write(*,*)"diagonalization error code:",info
        ! writing output files eigenvalues and eigenstates   
       ! open(10,file='eigenGroundn1000L2300sc.txt')
        print*,alpha,W(1)
        do j=1,nmax
       !     if (j<=20) then
       !         print*,j,W(j)
       !     endif
            write(34+nalpha,*)W(j)
        enddo
        !close(10)
        
     enddo
        
        call timestamp ( )   
        
     
     
        stop
        
        print*,"normalizing..."
        do kn=1,nmax       
            normal=0.d0
            do i=2,n-1
                do j=max(i-k,2),min(i+k,n-1)
                    term=S2(i-1,j-1)*HP(i-1,kn)*HP(j-1,kn)
                    normal=normal+term
                enddo
            enddo
            HP(:,kn)=HP(:,kn)/dsqrt(normal)
        enddo             
        HPG=HP
        open(100,file="ground_datan1000L1250.txt")
        write(100,*)((HP(i,j),i=1,n-2),j=1,nmax)
         
call timestamp ( )
    stop
    print*,"writing eigenfunctions"
    points=0.d0
    do j=0,nrgrid
        r=deltarplot*j+rmin
        wavefunc1=0.d0;wavefunc2=0.d0   
        if (r<rmax) then
            t=tgrid
            call interv ( t, n+1, r, left, mflag )
            values(1:n) = 0.0D+00
            t=tgrid
            call bsplvb ( t, k, indx, r, left, values(left+1-k) ) 
            do i=2,n-1!left-k+1,left     
                wavefunc1=wavefunc1+HPG(i-1,51)*values(i)
                wavefunc2=wavefunc2+HPG(i-1,nmax)*values(i)
            enddo 
        endif
        points(j)=wavefunc4**2
        write(500,*)r,wavefunc1**2
        write(501,*)r,wavefunc2**2
  
    enddo
    call int_simp(nrgrid,deltarplot,points,soma) 
    print*,soma

    stop
    end program eigen
!**********************************************************************************    
    subroutine int_simp(Np,delta,points,soma)
    implicit none
    integer, intent(in)::Np
    real*8,intent(in)::delta
    real*8,dimension(0:Np),intent(in)::points
    real*8,intent(out)::soma
    integer::k,l,Nq
    soma=0.d0
    Nq=Np/2
    do k=1,Nq
        l=2*k-2
        soma=soma+(points(l)+4*points(l+1)+points(l+2))
    enddo
    soma=delta*soma/3.d0
    return
    end subroutine int_simp
   
!*************************************************************************************
real*8 function VMC(alpha,r)!Morse-Coulomb potential
!use parametros
  implicit double precision (a-h,o-z)
  
  if (r<=0.) then
    VMC=(1.d0/alpha)*(dexp(-2*r/(alpha*dsqrt(2.d0)))-2*dexp(-r/(alpha*dsqrt(2.d0))))
    else
    VMC=-1/dsqrt(alpha**2+r**2)
   endif
end function VMC

    !***************************************************************************************
    real*8 function func(r)
    use bsplinesparameters
    implicit none
    integer ( kind = 4 ), parameter :: indx = 1
    real ( kind = 8 ) values(n)
    real*8:: r
    integer::mflag,left
    call interv ( t, n+1, r, left, mflag )
    values(1:n) = 0.0D+00
    call bsplvb ( t, k, indx, r, left, values(left+1-k) )
    func=values(i)*values(j)
    return
    end function func

    !***********************************************************************************
    real*8 function func2(r)
    use bsplinesparameters
    implicit none
    integer( kind = 4 ), parameter :: indx = 1
    real*8, intent(IN):: r
    integer::mflag,left,iprime
    real*8:: VMC
    external VMC  
    real ( kind = 8 ) values(n)
    real ( kind = 8 ) dbiatx(k,3)
    real ( kind = 8 ) aw(k,k)

    call interv ( t, n+1, r, left, mflag )
    values(1:n) = 0.0D+00
    call bsplvb ( t, k, indx, r, left, values(left+1-k) )

    func2=values(i)*values(j)*VMC(alpha,r)

    iprime=k-left+j
    if ((iprime<=0).or.(iprime>=k+1)) then
        return
    else
        call interv ( t, n+1, r, left, mflag )
        dbiatx=0.d0
        call bsplvd ( t, k, r, left, aw, dbiatx, 3 )
        func2=func2-0.5d0*values(i)*dbiatx(iprime,3)
    endif
    return
    end function func2
    
       !***********************************************************************************
    real*8 function func4(r)
    use bsplinesparameters
    implicit none
    integer( kind = 4 ), parameter :: indx = 1
    real*8, intent(IN):: r
    integer::mflag,left,iprime
    real*8:: VMC
    external VMC  
    real ( kind = 8 ) values(n)
    real ( kind = 8 ) dbiatx(k,3)
    real ( kind = 8 ) aw(k,k)

    call interv ( t, n+1, r, left, mflag )
    values(1:n) = 0.0D+00
    call bsplvb ( t, k, indx, r, left, values(left+1-k) )


    func4=values(i)*values(j)*VMC(alpha,r)

    iprime=k-left+j
    if ((iprime<=0).or.(iprime>=k+1)) then
        return
    else
        call interv ( t, n+1, r, left, mflag )
        dbiatx=0.d0
        call bsplvd ( t, k, r, left, aw, dbiatx, 3 )
        func4=func4-0.5d0*values(i)*dbiatx(iprime,3)
    endif
    return
    end function func4
!***********************************************************************************
    real*8 function func3(r)
    use bsplinesparameters
    implicit none
    integer ( kind = 4 ), parameter :: indx = 1
    real*8, intent(IN):: r
    integer::mflag,left
    real ( kind = 8 ) values(n)

    call interv ( t, n+1, r, left, mflag )
    values(1:n) = 0.0D+00
    call bsplvb ( t, k, indx, r, left, values(left+1-k) )
    func3=values(i)*values(j)*r
    return
    end function func3

    !*****************************************************************
    subroutine bsplvb ( t, jhigh, index, x, left, biatx )

    !*****************************************************************************
    !
    !! BSPLVB evaluates B-splines at a point X with a given knot sequence.
    !
    !  Discusion:
    !
    !    BSPLVB evaluates all possibly nonzero B-splines at X of order
    !
    !      JOUT = MAX ( JHIGH, (J+1)*(INDEX-1) )
    !
    !    with knot sequence T.
    !
    !    The recurrence relation
    !
    !                     X - T(I)               T(I+J+1) - X
    !    B(I,J+1)(X) = ----------- * B(I,J)(X) + --------------- * B(I+1,J)(X)
    !                  T(I+J)-T(I)               T(I+J+1)-T(I+1)
    !
    !    is used to generate B(LEFT-J:LEFT,J+1)(X) from B(LEFT-J+1:LEFT,J)(X)
    !    storing the new values in BIATX over the old.
    !
    !    The facts that
    !
    !      B(I,1)(X) = 1  if  T(I) <= X < T(I+1)
    !
    !    and that
    !
    !      B(I,J)(X) = 0  unless  T(I) <= X < T(I+J)
    !
    !    are used.
    !
    !    The particular organization of the calculations follows
    !    algorithm 8 in chapter X of the text.
    !
    !  Modified:
    !
    !    14 February 2007
    !
    !  Author:
    !
    !    Carl de Boor
    !
    !  Reference:
    !
    !    Carl de Boor,
    !    A Practical Guide to Splines,
    !    Springer, 2001,
    !    ISBN: 0387953663,
    !    LC: QA1.A647.v27.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ) T(LEFT+JOUT), the knot sequence.  T is assumed to
    !    be nondecreasing, and also, T(LEFT) must be strictly less than
    !    T(LEFT+1).
    !
    !    Input, integer ( kind = 4 ) JHIGH, INDEX, determine the order
    !    JOUT = max ( JHIGH, (J+1)*(INDEX-1) )
    !    of the B-splines whose values at X are to be returned.
    !    INDEX is used to avoid recalculations when several
    !    columns of the triangular array of B-spline values are
    !    needed, for example, in BVALUE or in BSPLVD.
    !    If INDEX = 1, the calculation starts from scratch and the entire
    !    triangular array of B-spline values of orders
    !    1, 2, ...,JHIGH is generated order by order, that is,
    !    column by column.
    !    If INDEX = 2, only the B-spline values of order J+1, J+2, ..., JOUT
    !    are generated, the assumption being that BIATX, J,
    !    DELTAL, DELTAR are, on entry, as they were on exit
    !    at the previous call.  In particular, if JHIGH = 0,
    !    then JOUT = J+1, that is, just the next column of B-spline
    !    values is generated.
    !    Warning: the restriction  JOUT <= JMAX (= 20) is
    !    imposed arbitrarily by the dimension statement for DELTAL
    !    and DELTAR, but is nowhere checked for.
    !
    !    Input, real ( kind = 8 ) X, the point at which the B-splines
    !    are to be evaluated.
    !
    !    Input, integer ( kind = 4 ) LEFT, an integer chosen so that
    !    T(LEFT) <= X <= T(LEFT+1).
    !
    !    Output, real ( kind = 8 ) BIATX(JOUT), with BIATX(I) containing the
    !    value at X of the polynomial of order JOUT which agrees
    !    with the B-spline B(LEFT-JOUT+I,JOUT,T) on the interval
    !    (T(LEFT),T(LEFT+1)).
    !
    implicit none

    integer ( kind = 4 ), parameter :: jmax = 20

    integer ( kind = 4 ) jhigh

    real ( kind = 8 ) biatx(jhigh)
    real ( kind = 8 ), save, dimension ( jmax ) :: deltal
    real ( kind = 8 ), save, dimension ( jmax ) :: deltar
    integer ( kind = 4 ) i
    integer ( kind = 4 ) index
    integer ( kind = 4 ), save :: j = 1
    integer ( kind = 4 ) left
    real ( kind = 8 ) saved
    real ( kind = 8 ) t(left+jhigh)
    real ( kind = 8 ) term
    real ( kind = 8 ) x

    if ( index == 1 ) then
        j = 1
        biatx(1) = 1.0D+00
        if ( jhigh <= j ) then
            return
        end if
    end if

    if ( t(left+1) <= t(left) ) then
        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'BSPLVB - Fatal error!'
        write ( *, '(a)' ) '  It is required that T(LEFT) < T(LEFT+1).'
        write ( *, '(a,i8)' ) '  But LEFT = ', left
        write ( *, '(a,g14.6)' ) '  T(LEFT) =   ', t(left)
        write ( *, '(a,g14.6)' ) '  T(LEFT+1) = ', t(left+1)
        stop 1
    end if

    do

        deltar(j) = t(left+j) - x
        deltal(j) = x - t(left+1-j)

        saved = 0.0D+00
        do i = 1, j
            term = biatx(i) / ( deltar(i) + deltal(j+1-i) )
            biatx(i) = saved + deltar(i) * term
            saved = deltal(j+1-i) * term
        end do

        biatx(j+1) = saved
        j = j + 1

        if ( jhigh <= j ) then
            exit
        end if

    end do

    return
    end
    !*****************************************************************************
    subroutine interv ( xt, lxt, x, left, mflag )
    !
    !! INTERV brackets a real value in an ascending vector of values.
    !
    !  Discussion:
    !
    !    The XT array is a set of increasing values.  The goal of the routine
    !    is to determine the largest index I so that
    !
    !      XT(I) < XT(LXT)  and  XT(I) <= X.
    !
    !    The routine is designed to be efficient in the common situation
    !    that it is called repeatedly, with X taken from an increasing
    !    or decreasing sequence.
    !
    !    This will happen when a piecewise polynomial is to be graphed.
    !    The first guess for LEFT is therefore taken to be the value
    !    returned at the previous call and stored in the local variable ILO.
    !
    !    A first check ascertains that ILO < LXT.  This is necessary
    !    since the present call may have nothing to do with the previous
    !    call.  Then, if
    !      XT(ILO) <= X < XT(ILO+1),
    !    we set LEFT = ILO and are done after just three comparisons.
    !
    !    Otherwise, we repeatedly double the difference ISTEP = IHI - ILO
    !    while also moving ILO and IHI in the direction of X, until
    !      XT(ILO) <= X < XT(IHI)
    !    after which we use bisection to get, in addition, ILO + 1 = IHI.
    !    The value LEFT = ILO is then returned.
    !
    !    Thanks to Daniel Gloger for pointing out an important modification
    !    to the routine, so that the piecewise polynomial in B-form is
    !    left-continuous at the right endpoint of the basic interval,
    !    17 April 2014.
    !
    !  Modified:
    !
    !    17 April 2014
    !
    !  Author:
    !
    !    Carl de Boor
    !
    !  Reference:
    !
    !    Carl de Boor,
    !    A Practical Guide to Splines,
    !    Springer, 2001,
    !    ISBN: 0387953663,
    !    LC: QA1.A647.v27.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ) XT(LXT), a nondecreasing sequence of values.
    !
    !    Input, integer ( kind = 4 ) LXT, the dimension of XT.
    !
    !    Input, real ( kind = 8 ) X, the point whose location with
    !    respect to the sequence XT is to be determined.
    !
    !    Output, integer ( kind = 4 ) LEFT, the index of the bracketing value:
    !      1     if             X  <  XT(1)
    !      I     if   XT(I)  <= X  < XT(I+1)
    !      I     if   XT(I)  <  X == XT(I+1) == XT(LXT)
    !
    !    Output, integer ( kind = 4 ) MFLAG, indicates whether X lies within the
    !    range of the data.
    !    -1:            X  <  XT(1)
    !     0: XT(I)   <= X  < XT(I+1)
    !    +1: XT(LXT) <  X
    !
    implicit none

    integer ( kind = 4 ) lxt

    integer ( kind = 4 ) left
    integer ( kind = 4 ) mflag
    integer ( kind = 4 ) ihi
    integer ( kind = 4 ), save :: ilo = 1
    integer ( kind = 4 ) istep
    integer ( kind = 4 ) middle
    real ( kind = 8 ) x
    real ( kind = 8 ) xt(lxt)


    ! do ihi=1,lxt
    !     write(2,*)xt(ihi)
    ! enddo

    ! stop

    ihi = ilo + 1

    if ( lxt <= ihi ) then

        if ( xt(lxt) <= x ) then
            go to 110
        end if

        if ( lxt <= 1 ) then
            mflag = -1
            left = 1
            return
        end if

        ilo = lxt - 1
        ihi = lxt

    end if

    if ( xt(ihi) <= x ) then
        go to 20
    end if

    if ( xt(ilo) <= x ) then
        mflag = 0
        left = ilo
        return
    end if
    !
    !  Now X < XT(ILO).  Decrease ILO to capture X.
    !
    istep = 1

10  continue

    ihi = ilo
    ilo = ihi - istep

    if ( 1 < ilo ) then
        if ( xt(ilo) <= x ) then
            go to 50
        end if
        istep = istep * 2
        go to 10
    end if

    ilo = 1

    if ( x < xt(1) ) then
        mflag = -1
        left = 1
        return
    end if

    go to 50
    !
    !  Now XT(IHI) <= X.  Increase IHI to capture X.
    !
20  continue

    istep = 1

30  continue

    ilo = ihi
    ihi = ilo + istep

    if ( ihi < lxt ) then

        if ( x < xt(ihi) ) then
            go to 50
        end if

        istep = istep * 2
        go to 30

    end if

    if ( xt(lxt) <= x ) then
        go to 110
    end if
    !
    !  Now XT(ILO) < = X < XT(IHI).  Narrow the interval.
    !
    ihi = lxt

50  continue

    do

        middle = ( ilo + ihi ) / 2

        if ( middle == ilo ) then
            mflag = 0
            left = ilo
            return
        end if
        !
        !  It is assumed that MIDDLE = ILO in case IHI = ILO+1.
        !
        if ( xt(middle) <= x ) then
            ilo = middle
        else
            ihi = middle
        end if

    end do
    !
    !  Set output and return.
    !
110 continue

    mflag = 1

    if ( x == xt(lxt) ) then
        mflag = 0
    end if

    do left = lxt - 1, 1, -1
        if ( xt(left) < xt(lxt) ) then
            return
        end if
    end do

    return
    end

    !*****************************************************************************
    subroutine timestamp ( )

    !*****************************************************************************80
    !
    !! TIMESTAMP prints the current YMDHMS date as a time stamp.
    !
    !  Example:
    !
    !    31 May 2001   9:45:54.872 AM
    !
    !  Licensing:
    !
    !    This code is distributed under the GNU LGPL license.
    !
    !  Modified:
    !
    !    18 May 2013
    !
    !  Author:
    !
    !    John Burkardt
    !
    !  Parameters:
    !
    !    None
    !
    implicit none

    character ( len = 8 ) ampm
    integer ( kind = 4 ) d
    integer ( kind = 4 ) h
    integer ( kind = 4 ) m
    integer ( kind = 4 ) mm
    character ( len = 9 ), parameter, dimension(12) :: month = (/ &
        'January  ', 'February ', 'March    ', 'April    ', &
        'May      ', 'June     ', 'July     ', 'August   ', &
        'September', 'October  ', 'November ', 'December ' /)
    integer ( kind = 4 ) n
    integer ( kind = 4 ) s
    integer ( kind = 4 ) values(8)
    integer ( kind = 4 ) y

    call date_and_time ( values = values )

    y = values(1)
    m = values(2)
    d = values(3)
    h = values(5)
    n = values(6)
    s = values(7)
    mm = values(8)

    if ( h < 12 ) then
        ampm = 'AM'
    else if ( h == 12 ) then
        if ( n == 0 .and. s == 0 ) then
            ampm = 'Noon'
        else
            ampm = 'PM'
        end if
    else
        h = h - 12
        if ( h < 12 ) then
            ampm = 'PM'
        else if ( h == 12 ) then
            if ( n == 0 .and. s == 0 ) then
                ampm = 'Midnight'
            else
                ampm = 'AM'
            end if
        end if
    end if

    write ( *, '(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
        d, trim ( month(m) ), y, h, ':', n, ':', s, '.', mm, trim ( ampm )

    return
    end


    subroutine bsplvd ( t, k, x, left, a, dbiatx, nderiv )

    !*****************************************************************************
    !
    !! BSPLVD calculates the nonvanishing B-splines and derivatives at X.
    !
    !  Discussion:
    !
    !    Values at X of all the relevant B-splines of order K:K+1-NDERIV
    !    are generated via BSPLVB and stored temporarily in DBIATX.
    !
    !    Then the B-spline coefficients of the required derivatives
    !    of the B-splines of interest are generated by differencing,
    !    each from the preceding one of lower order, and combined with
    !    the values of B-splines of corresponding order in DBIATX
    !    to produce the desired values.
    !
    !  Modified:
    !
    !    14 February 2007
    !
    !  Author:
    !
    !    Carl de Boor
    !
    !  Reference:
    !
    !    Carl de Boor,
    !    A Practical Guide to Splines,
    !    Springer, 2001,
    !    ISBN: 0387953663,
    !    LC: QA1.A647.v27.
    !
    !  Parameters:
    !
    !    Input, real ( kind = 8 ) T(LEFT+K), the knot sequence.  It is assumed that
    !    T(LEFT) < T(LEFT+1).  Also, the output is correct only if
    !    T(LEFT) <= X <= T(LEFT+1).
    !
    !    Input, integer ( kind = 4 ) K, the order of the B-splines to be evaluated.
    !
    !    Input, real ( kind = 8 ) X, the point at which these values are sought.
    !
    !    Input, integer ( kind = 4 ) LEFT, indicates the left endpoint of the
    !    interval of interest.  The K B-splines whose support contains the interval
    !    ( T(LEFT), T(LEFT+1) ) are to be considered.
    !
    !    Workspace, real ( kind = 8 ) A(K,K).
    !
    !    Output, real ( kind = 8 ) DBIATX(K,NDERIV).  DBIATX(I,M) contains
    !    the value of the (M-1)st derivative of the (LEFT-K+I)-th B-spline
    !    of order K for knot sequence T, I=M,...,K, M=1,...,NDERIV.
    !
    !    Input, integer ( kind = 4 ) NDERIV, indicates that values of B-splines and
    !    their derivatives up to but not including the NDERIV-th are asked for.
    !
    implicit none

    integer ( kind = 4 ) k
    integer ( kind = 4 ) left
    integer ( kind = 4 ) nderiv

    real ( kind = 8 ) a(k,k)
    real ( kind = 8 ) dbiatx(k,nderiv)
    real ( kind = 8 ) factor
    real ( kind = 8 ) fkp1mm
    integer ( kind = 4 ) i
    integer ( kind = 4 ) ideriv
    integer ( kind = 4 ) il
    integer ( kind = 4 ) j
    integer ( kind = 4 ) jlow
    integer ( kind = 4 ) jp1mid
    integer ( kind = 4 ) ldummy
    integer ( kind = 4 ) m
    integer ( kind = 4 ) mhigh
    real ( kind = 8 ) sum1
    real ( kind = 8 ) t(left+k)
    real ( kind = 8 ) x

    mhigh = max ( min ( nderiv, k ), 1 )
    !
    !  MHIGH is usually equal to NDERIV.
    !
    call bsplvb ( t, k+1-mhigh, 1, x, left, dbiatx )

    if ( mhigh == 1 ) then
        return
    end if
    !
    !  The first column of DBIATX always contains the B-spline values
    !  for the current order.  These are stored in column K+1-current
    !  order before BSPLVB is called to put values for the next
    !  higher order on top of it.
    !
    ideriv = mhigh
    do m = 2, mhigh
        jp1mid = 1
        do j = ideriv, k
            dbiatx(j,ideriv) = dbiatx(jp1mid,1)
            jp1mid = jp1mid + 1
        end do
        ideriv = ideriv - 1
        call bsplvb ( t, k+1-ideriv, 2, x, left, dbiatx )
    end do
    !
    !  At this point, B(LEFT-K+I, K+1-J)(X) is in DBIATX(I,J) for
    !  I=J,...,K and J=1,...,MHIGH ('=' NDERIV).
    !
    !  In particular, the first column of DBIATX is already in final form.
    !
    !  To obtain corresponding derivatives of B-splines in subsequent columns,
    !  generate their B-representation by differencing, then evaluate at X.
    !
    jlow = 1
    do i = 1, k
        a(jlow:k,i) = 0.0D+00
        jlow = i
        a(i,i) = 1.0D+00
    end do
    !
    !  At this point, A(.,J) contains the B-coefficients for the J-th of the
    !  K B-splines of interest here.
    !
    do m = 2, mhigh

        fkp1mm = real ( k + 1 - m, kind = 8 )
        il = left
        i = k
        !
        !  For J = 1,...,K, construct B-coefficients of (M-1)st derivative of
        !  B-splines from those for preceding derivative by differencing
        !  and store again in  A(.,J).  The fact that  A(I,J) = 0 for
        !  I < J is used.
        !
        do ldummy = 1, k + 1 - m

            factor = fkp1mm / ( t(il+k+1-m) - t(il) )
            !
            !  The assumption that T(LEFT) < T(LEFT+1) makes denominator
            !  in FACTOR nonzero.
            !
            a(i,1:i) = ( a(i,1:i) - a(i-1,1:i) ) * factor

            il = il - 1
            i = i - 1

        end do
        !
        !  For I = 1,...,K, combine B-coefficients A(.,I) with B-spline values
        !  stored in DBIATX(.,M) to get value of (M-1)st derivative of
        !  I-th B-spline (of interest here) at X, and store in DBIATX(I,M).
        !
        !  Storage of this value over the value of a B-spline
        !  of order M there is safe since the remaining B-spline derivatives
        !  of the same order do not use this value due to the fact
        !  that  A(J,I) = 0  for J < I.
        !
        do i = 1, k

            jlow = max ( i, m )

            dbiatx(i,m) = dot_product ( a(jlow:k,i), dbiatx(jlow:k,m) )

        end do

    end do

    return
    end

    !****************************************************************************************

    SUBROUTINE spline(x,y,n,yp1,ypn,y2)
    INTEGER n
    REAL*8 yp1,ypn,x(n),y(n),y2(n)
    integer,PARAMETER:: NMAX=5000
    INTEGER i,k
    REAL*8 p,qn,sig,un,u(NMAX)

    if (yp1.gt.0.99d30) then
        y2(1)=0.d0
        u(1)=0.d0
    else
        y2(1)=-0.5d0
        u(1)=(3.d0/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
    endif
    do 11 i=2,n-1
        sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))
        p=sig*y2(i-1)+2.d0
        y2(i)=(sig-1.)/p
        u(i)=(6.d0*((y(i+1)-y(i))/(x(i+&
            1)-x(i))-(y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*&
            u(i-1))/p
11  continue
    if (ypn.gt..99d30) then
        qn=0.d0
        un=0.d0
    else
        qn=0.5d0
        un=(3.d0/(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))
    endif
    y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.d0)
    do 12 k=n-1,1,-1
        y2(k)=y2(k)*y2(k+1)+u(k)
12  continue
    return
    END subroutine spline
    !C  (C) Copr. 1986-92 Numerical Recipes Software 530.
    !*********************************************************************************
    SUBROUTINE splint(xa,ya,y2a,n,x,y)
    INTEGER n
    REAL*8 x,y,xa(n),y2a(n),ya(n)
    INTEGER k,khi,klo
    REAL*8 a,b,h
    klo=1
    khi=n
1   if (khi-klo.gt.1) then
        k=(khi+klo)/2
        if(xa(k).gt.x)then
            khi=k
        else
            klo=k
        endif
        goto 1
    endif
    h=xa(khi)-xa(klo)
    if (h.eq.0.) write(*,*)'bad xa input in splint'
    a=(xa(khi)-x)/h
    b=(x-xa(klo))/h
    y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**&
        2)/6.d0
    return
    END subroutine splint
    !C  (C) Copr. 1986-92 Numerical Recipes Software 530.
    !*************************************************************************************