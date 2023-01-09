# Implementation of steepest descent algorithm
#
# This version has some plotting functions at the end. 
# Run these as: 
# >> plotcontours()
# >> plotdata(xsave,str)
# "str" is a label for the legend of the plot. 
#
# The built-in minimization problem is
#           min   0.5*x'*Q*x - b'*x + 1
#
# Run this as (for example):
# >> include("steep-hw2.jl");
# >> x0 = [10;10];
# >> alpha = 0.1;
# >> xsave = SteepestDescent(x0,alpha);
# >> c1 = 1e-3;
# >> ysave = SteepestDescentArmijo(x0, c1);
#

using Printf
using Plots


# uncomment if you want to do linear algebra calculations
#using LinearAlgebra



# matrix and vector used in quadratic form. 
# defined here, because they are used in both F(x), and DF(x)
Q = [3 -1; -1 1];
b = [1;1];

# set parameters here, for all gradient descent algorithms
tol = 1e-8;     # tolerance on norm of gradient
MaxIter = 10000;  # maximum number of iterations of gradient descent


# define function
function F(x)
   val = 0.5*x'*Q*x - b'*x + 1
   return val
end

# define gradient
function DF(x)
   grad = Q*x-b;
   return grad
end


#
# steepest descent algorithm, with constant step size
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    alpha = step size. Constant, in this algorithm.
# output: 
#    xsave = entire list of points (useful if you want to plot them)
#
function SteepestDescent(x0,alpha)

   # setup for steepest descent
   x = x0;
   successflag = false;
   xsave = zeros(2,MaxIter);
   xsave[:,1] = x0;

   # perform steepest descent iterations
   for iter = 1:MaxIter
       Fval = F(x);
       Fgrad = DF(x);
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, Fval)
          successflag = true;
          xsave = xsave[:,1:iter];
          break;
       end
       # perform steepest descent step
       x = x - alpha*Fgrad;
       
       # save point
       xsave[:,iter+1] = x;
   end
   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end
   return xsave;
end



#
# steepest descent algorithm, with Armijo's rule for backtracking
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    c1 = slope, in Armijo's rule.
# output: 
#    xsave = entire list of points (useful if you want to plot them)
#
function SteepestDescentArmijo(x0, c1)

   # parameters for Armijo's rule
   alpha0 = 10.0;    # initial value of alpha, to try in backtracking
   eta = 0.5;       # factor with which to scale alpha, each time you backtrack
   MaxBacktrack = 20;  # maximum number of backtracking steps

   # setup for steepest descent
   x = x0;
   successflag = false;   
   xsave = zeros(2,MaxIter);
   xsave[:,1] = x0;

   # perform steepest descent iterations
   for iter = 1:MaxIter

      alpha = alpha0;
      Fval = F(x);
      Fgrad = DF(x);

      # check if norm of gradient is small enough
      if sqrt(Fgrad'*Fgrad) < tol
         @printf("Converged after %d iterations, function value %f\n", iter, Fval)
         successflag = true;
         xsave = xsave[:,1:iter];
         break;
      end

      # perform line search
      for k = 1:MaxBacktrack
         x_try = x - alpha*Fgrad;
         Fval_try = F(x_try);
         if (Fval_try > Fval - c1*alpha *Fgrad'Fgrad)
            alpha = alpha * eta;
         else
            Fval = Fval_try;
            x = x_try;
            break;
         end
      end

      # save point
      xsave[:,iter+1] = x;

      # print how we're doing, every 10 iterations
      if (iter%10==0)
         @printf("iter: %d: alpha: %f, %f, %f, %f\n", iter, alpha, x[1], x[2], Fval)
      end
   end

   if successflag == false
       @printf("Failed to converge after %d iterations, function value %f\n", MaxIter, F(x))
   end

   return xsave
end




# here are some visualization functions

# plot contours of function
function plotcontours()
   # limits of the plot. change to whatever you wish. 
   xmin = -1;
   xmax = 4;
   ymin = 0;
   ymax = 5;
   x = xmin:0.02:xmax;
   y = ymin:0.02:ymax;
   Z = zeros(length(y), length(x));
   for i=1:length(x)
      for j=1:length(y)
        Z[j,i] = F([x[i]; y[j]]);
      end
   end
   contourf(x,y,Z,levels=25);
end


# plot data on top of contour plot
function plotdata(xsave,str)
   K= length(xsave[1,:]);
   plot!(xsave[1,1:1:K], xsave[2,1:1:K], lw = 2, marker=2, label = str,legend=:bottomright);
   # uncomment if you want to save the figure. Change the filename to whatever you'd like. 
   #savefig("steepest.png")
end

