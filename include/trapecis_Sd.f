! Author: Marián Boguñá
      subroutine pkk(result,d,beta,c,upper_bound)
      implicit none
      double precision Intnew,pi,a,b,Int0,Int1,error,c,beta,upper_bound
      double precision Intfirst,Intsecond,delta,result
      integer n,i,idum,m,d

      delta=1.d-5 ! precision
      pi=4.d0*datan(1.d0)

      a=0.d0 ! lower bound
      b=upper_bound !pi   ! upper bound

      m=50
      error=2.d0*delta
      do while(error.gt.delta)
       m=2*m

       call trapecis1(m,Intnew,a,b,d,c,beta)
       Int0=Intnew
       call trapecis1(2*m,Intnew,a,b,d,c,beta)
       Int1=Intnew
       Intfirst=(4.d0*Int1-Int0)/3.d0

       m=2*m

       call trapecis1(m,Intnew,a,b,d,c,beta)
       Int0=Intnew
       call trapecis1(2*m,Intnew,a,b,d,c,beta)
       Int1=Intnew
       Intsecond=(4.d0*Int1-Int0)/3.d0

       error=dabs(Intsecond-Intfirst)/dabs(Intfirst)

      enddo

       result=Intsecond
       ! Do not multiply by constant every time
       !*dgamma(dble(1+d)/2.d0)/(dsqrt(pi)*dgamma(dble(d)/2.d0))

      return
      end

      subroutine pkk_expected(result,d,beta,c)
            implicit none
            double precision Intnew,pi,a,b,Int0,Int1,error,c,beta
            double precision Intfirst,Intsecond,delta,result
            integer n,i,idum,m,d
      
            delta=1.d-5 ! precision
            pi=4.d0*datan(1.d0)
      
            a=0.d0 ! lower bound
            b=pi   ! upper bound
      
            m=50
            error=2.d0*delta
            do while(error.gt.delta)
             m=2*m
      
             call trapecis2(m,Intnew,a,b,d,c,beta)
             Int0=Intnew
             call trapecis2(2*m,Intnew,a,b,d,c,beta)
             Int1=Intnew
             Intfirst=(4.d0*Int1-Int0)/3.d0
      
             m=2*m
      
             call trapecis2(m,Intnew,a,b,d,c,beta)
             Int0=Intnew
             call trapecis2(2*m,Intnew,a,b,d,c,beta)
             Int1=Intnew
             Intsecond=(4.d0*Int1-Int0)/3.d0
      
             error=dabs(Intsecond-Intfirst)/dabs(Intfirst)
      
            enddo
      
             result=Intsecond
             ! Do not multiply by constant every time
             !*dgamma(dble(1+d)/2.d0)/(dsqrt(pi)*dgamma(dble(d)/2.d0))
            return
            end

      subroutine trapecis1(n,Int,a,b,d,c,beta)
      double precision Int,h,x,a,b,func,c,beta
      integer n,i,d

       x=a
       Int=func(a,d,c,beta)+func(b,d,c,beta)
       h=(b-a)/dble(n)

       do i=1,n-1
        x=x+h
        Int=Int+2.d0*func(x,d,c,beta)
       enddo
       Int=Int*h/2.d0

       return
       end

      subroutine trapecis2(n,Int,a,b,d,c,beta)
      double precision Int,h,x,a,b,func2,c,beta
      integer n,i,d

       x=a
       Int=func2(a,d,c,beta)+func2(b,d,c,beta)
       h=(b-a)/dble(n)

       do i=1,n-1
        x=x+h
        Int=Int+2.d0*func2(x,d,c,beta)
       enddo
       Int=Int*h/2.d0

      return
      end

      double precision function func(x,d,c,beta)
      double precision x,c,beta
      integer d

       func=(dsin(x))**(d-1)/(1.d0+(c*x)**beta)


      return
      end


      double precision function func2(x,d,c,beta)
      double precision x,c,beta
      integer d

       func2=(x*(dsin(x))**(d-1))/(1.d0+(c*x)**beta)

      return
      end
