module global_var
implicit none
real*8::alpha
real*8,parameter::sq2=dsqrt(2.d0)
integer,parameter::nvar=2
real*8::F0,W0


end module global_var
!*********************************************************

program Morse_Coulomb
use global_var
implicit none
character(60)::filename1,filename2
integer::i,j,k,ncont
real*8,parameter::tmax=2000.d0
integer,parameter::NF0=10
integer::iF0
real*8,parameter::passoF0=2.d-3
integer::Nr=150
integer::Nt=100000
real*8::r,rstep,r0,p0,t0,period,tper
real*8,external::VMC,dVMC,Ecomp
real*8::E,t,tstep,t1,t2
real*8,dimension(nvar)::y
real*8::eps,h1,hmin
integer::nbad,nok
real*8::rext,rmax,rint,rmin
real*8::cont,Pion
real*8::pi=dacos(-1.d0)
real*8::Esum,Emed
integer::nEm

print*, 'Starting at: '
print*
call timestamp ( )


eps=1.d-8! precisão
h1=eps!	passo inicial
hmin=0.d0 !menor passo permitido

alpha=0.05d0
W0=1.d0
period=2*pi/W0
E=-0.5d0



rext=dsqrt((1.d0/E)**2-alpha**2)
rstep=(rext)/Nr
rint=-alpha*sq2*dlog(dsqrt(alpha*E+1)+1)

tstep=tmax/Nt!1.d-3



!data files
write(filename1,'("D2Morsecoulomb_W",F5.2,"alpha",F5.2,"E0",F5.2,".txt")')W0,alpha,E
open(10,file=filename1)

t0=-pi/(2*W0) !Faz o campo zero para t=t0

do iF0=1,NF0-1
F0=passoF0*iF0+0.08d0
cont=0.d0

loop1:do i=1,Nr-1
   r0=i*rstep
 do k=1,2
   p0=(-1)**(k+1)*dsqrt(2*(E-VMC(r0)))
   y(1)=r0
   y(2)=p0

loop2:do j=1,Nt
      t1=j*tstep+t0
      t2=t1+tstep
      
      call odeint(y,nvar,t1,t2,eps,h1,hmin,nok,nbad)

      if ((0.5*y(2)**2+VMC(y(1))>0.d0).AND.(y(1)>16.d0)) then
         cont=cont+1
         exit loop2
      endif

!  if (Ecomp(y(1),y(2),t2)>0.d0.AND.(y(1)>16.d0)) print*,"Ionizou c2"

    enddo  loop2
    !print*,i,cont
enddo
enddo loop1

Pion=cont/(Nr-1)/2
print*,"Ionization probability= ",F0,Pion
write(10,*)F0,Pion

enddo

print*, 'Ending at: '
print*
call timestamp ( )

pause
end program Morse_Coulomb
!**************************

real*8 function VMC(r)
use global_var
implicit none
real*8,intent(IN)::r
real*8::aux1,aux2

if (r<0.d0) then
   aux1=1/alpha
   aux2=aux1/sq2
   VMC=aux1*(dexp(-2*aux2*r)-2*dexp(-aux2*r))
else
    VMC=-1/dsqrt(r**2+alpha**2)
endif
end function VMC

!**************************

real*8 function dVMC(r)
use global_var
implicit none
real*8,intent(IN)::r
real*8::aux1,aux2,aux3

if (r<0.d0) then
   aux1=(1/alpha)
   aux2=aux1/sq2
   aux3=aux1**2/sq2
   dVMC=-2*aux3*(dexp(-2*aux2*r)-dexp(-aux2*r))
else
    dVMC=r*(r**2+alpha**2)**(-1.5d0)
endif
end function dVMC

!****************************************************

real*8 function Ecomp(r,p,t) ! energia compensada
use global_var
implicit none
real*8::r,p,t
real*8,external::VMC

Ecomp=0.5*(p+(F0/W0)*dsin(W0*t))**2+VMC(r)

end function Ecomp
!**************************************************

subroutine derivs(t,y,dydt) !Sistema de equações diferenciais

use global_var
implicit none
real*8::t
real*8, dimension(nvar) :: y,dydt
real*8,external::dVMC

dydt(1)=y(2)
dydt(2)=-dVMC(y(1))-F0*dcos(W0*t)



end subroutine derivs

!**************************************************
SUBROUTINE odeint(ystart,nvar,x1,x2,eps,h1,hmin,nok,nbad)

  implicit double precision (a-h,o-z)

  double precision,dimension(nvar):: ystart(nvar)
  integer, parameter:: MAXSTP=10000,NMAX=1000,KMAXX=1000
  double precision, parameter :: TINY=1.d-30
  double precision,  dimension(NMAX):: dydx,y,yscal
  double precision,  dimension(KMAXX):: xp
  double precision,  dimension(NMAX,KMAXX)::yp

  COMMON /path/ kmax,kount ,dxsav,xp,yp
  x=x1
  h=dsign(h1,x2-x1)
  nok=0
  nbad=0
  kount=0
  do  i=1,nvar
     y(i)=ystart(i)
  enddo
  if (kmax.gt.0) xsav=x-2.d0*dxsav
  do nstp=1,MAXSTP
     call derivs(x,y,dydx)
     do i=1,nvar
        yscal(i)=dabs(y(i))+dabs(h*dydx(i))+TINY
     enddo
     if(kmax.gt.0)then
        if(dabs(x-xsav).gt.dabs(dxsav)) then
           if(kount.lt.kmax-1)then
              kount=kount+1
              xp(kount)=x
              do  i=1,nvar
                 yp(i,kount)=y(i)
              enddo
              xsav=x
           endif
        endif
     endif
     if((x+h-x2)*(x+h-x1).gt.0.d0) h=x2-x
     call rkqs(y,dydx,nvar,x,h,eps,yscal,hdid,hnext)
     if(hdid.eq.h)then
        nok=nok+1
     else
        nbad=nbad+1
     endif
     if((x-x2)*(x2-x1).ge.0.d0)then
        do  i=1,nvar
           ystart(i)=y(i)
        enddo
        if(kmax.ne.0)then
           kount=kount+1
           xp(kount)=x
           do  i=1,nvar
              yp(i,kount)=y(i)
           enddo
        endif
        return
     endif
     if(dabs(hnext).lt.hmin) write(*,*)&
          'stepsize smaller than minimum in odeint'
     h=hnext
  enddo
  write(*,*) 'too many steps in odeint'
  return
END SUBROUTINE odeint
!  (C) Copr. 1986-92 Numerical Recipes Software 530.
!*************************************************************************
SUBROUTINE rkck(y,dydx,n,x,h,yout,yerr)

  implicit double precision (a-h,o-z)

  double precision,dimension(n):: dydx,y,yerr,yout

  integer,PARAMETER ::NMAX=1000
  double precision, dimension (NMAX):: ak2,ak3,ak4,ak5,ak6,ytemp
  double precision, parameter::A2=.2d0,A3=.3d0,A4=.6d0,A5=1.d0,A6=.875d0,B21=.2d0&
       ,B31=3.d0/40.d0,B32=9.d0/40.d0,B41=.3d0,B42=-.9d0,B43=1.2d0&
       ,B51=-11.d0/54.d0,B52=2.5d0,B53=-70.d0/27.d0,B54=35.d0/27.d0&
       ,B61=1631.d0/55296.d0,B62=175.d0/512.d0,B63=575.d0/13824.d0&
       ,B64=44275.d0/110592.d0,B65=253.d0/4096.d0,C1=37.d0/378.d0,&
       C3=250.d0/621.d0,C4=125.d0/594.d0,C6=512.d0/1771.d0&
       ,DC1=C1-2825.d0/27648.d0,DC3=C3-18575.d0/48384.d0&
       ,DC4=C4-13525.d0/55296.d0,DC5=-277.d0/14336.d0,&
       DC6=C6-.25d0

  do i=1,n
     ytemp(i)=y(i)+B21*h*dydx(i)
  enddo
  call derivs(x+A2*h,ytemp,ak2)
  do  i=1,n
     ytemp(i)=y(i)+h*(B31*dydx(i)+B32*ak2(i))
  enddo
  call derivs(x+A3*h,ytemp,ak3)
  do  i=1,n
     ytemp(i)=y(i)+h*(B41*dydx(i)+B42*ak2(i)+B43*ak3(i))
  enddo
  call derivs(x+A4*h,ytemp,ak4)
  do  i=1,n
     ytemp(i)=y(i)+h*(B51*dydx(i)+B52*ak2(i)+B53*ak3(i)+B54*ak4(i))
  enddo
  call derivs(x+A5*h,ytemp,ak5)
  do  i=1,n
     ytemp(i)=y(i)+h*(B61*dydx(i)+B62*ak2(i)+B63*ak3(i)+B64*ak4(i)+&
     B65*ak5(i))
  enddo
  call derivs(x+A6*h,ytemp,ak6)
  do  i=1,n
     yout(i)=y(i)+h*(C1*dydx(i)+C3*ak3(i)+C4*ak4(i)+C6*ak6(i))
  enddo
  do i=1,n
     yerr(i)=h*(DC1*dydx(i)+DC3*ak3(i)+DC4*ak4(i)+DC5*ak5(i)+DC6*&
     ak6(i))
  enddo
  return
END SUBROUTINE rkck
!  (C) Copr. 1986-92 Numerical Recipes Software 530.
!*************************************************************************
SUBROUTINE rkqs(y,dydx,n,x,htry,eps,yscal,hdid,hnext)

  implicit double precision (a-h,o-z)

  double precision,dimension(n):: dydx,y,yscal
  integer,PARAMETER:: NMAX=1000
  double precision,dimension(NMAX):: yerr,ytemp
  double precision,PARAMETER ::SAFETY=0.9d0,PGROW=-.2d0,PSHRNK=-.25d0,ERRCON=1.89d-4

  h=htry
1 call rkck(y,dydx,n,x,h,ytemp,yerr)
  errmax=0.
  do i=1,n
     errmax=max(errmax,abs(yerr(i)/yscal(i)))
  enddo
  errmax=errmax/eps
  if(errmax.gt.1.d0)then
     h=SAFETY*h*(errmax**PSHRNK)
     if(h.lt.0.1d0*h)then
        h=.1d0*h
     endif
     xnew=x+h
     if(xnew.eq.x) write(*,*) 'stepsize underflow in rkqs'
     goto 1
  else
     if(errmax.gt.ERRCON)then
        hnext=SAFETY*h*(errmax**PGROW)
     else
        hnext=5.d0*h
     endif
     hdid=h
     x=x+h
     do i=1,n
        y(i)=ytemp(i)
     enddo
     return
  endif
END SUBROUTINE rkqs
!  (C) Copr. 1986-92 Numerical Recipes Software 530.
!*************************************************************************
    !*****************************************************************************
    subroutine timestamp ( )

    !*****************************************************************************
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














