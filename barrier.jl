# implements penalty method to solve example explored in Lab 4
# works with multiple inequality constraints
#
# created dec 1, 2021
#
#
# To run this: 
#   set c, eg 
#            c = 5;
#   plot contours using    
#            plotcontours(c);
#   run steepest descent using eg
#            alph = 0.01;
#            x = SteepestDescent(x0,alpha,c);
#   plot data using
#            plotdata(x,c);
#
# Try varying the initial condition, x0
#
#
#
# In this code: replace multiple #### with your own code.
#


using Printf
using LinearAlgebra
using Plots

# set parameters here, for all optimization algorithms
tol = 1e-6;     # tolerance on norm of gradient
MaxIter = 100000;  # maximum number of iterations of gradient descent


# initial condition
x0 = [0.5;0.1];


#============  Function definitions ============#

a = 1;  # parameter used to define objective function

# function we wish to minimize
function f0(x)
    if findmax(g0(x))[1] < 0
        return -(a*x[1]+x[2]);
    else 
        return NaN;    # returns NaN so plotcontours doesn't plot anything outside boundary
    end
end

# function's gradient
function Df0(x)
    return -[a;1];
end

# inequality constraints
function g0(x)
    return [x[1]+x[2]-1;
            -x[1];
            -x[2]];
end

# inequality gradients (each gradient is in a column)
function Dg0(x)
    return [1  1;
            -1  0;
            0  -1]';
end

# barrier function
function B(x) 
    g = g0(x);
    ng = length(g);
    val = 0;
    for i in 1:ng
        if g[i] < 0
            val += -log(-g[i]);
        end
    end
    return val;
end

# barrier's gradient
function DB(x)
    g = g0(x);
    ng = length(g);
    grad = [0;0];
    for i in 1:ng
        grad += -Dg0(x)[:,i]./g[i];
    end
    return grad;
end


# overall function we will minimize (with parameter c)
function Fc(x,c)
    return f0(x) + 1/c * B(x);
end

# overall function's gradient (with parameter c)
function DFc(x,c)
    return Df0(x) + 1/c * DB(x);
end

# overall function we will minimize (commented out)
#=
c = 5;
F = Base.Fix2(Fc,c);
DF = Base.Fix2(DFc,c);
=#



#============  Plotting functions ============#

# plot contours of function
function plotcontours(c)
   off = 0.01;
   xmin = 0-off;
   xmax = 1 +off;
   ymin = xmin;
   ymax = xmax;
   x = xmin:0.01:xmax;
   y = ymin:0.01:ymax;
   Z = zeros(length(y), length(x));
   for i=1:length(x)
      for j=1:length(y)
        Z[j,i] = Fc([x[i]; y[j]],c);
      end
   end

   # plot contours
   contourf(x,y,Z,levels=30, fill=(true,cgrad(rev=true)),aspect_ratio=:equal, size=(500,500));

   # plot inequality constraint on top (specific to this example)
   plot!([0,1,0,0], [0,0,1,0],color=:blue, linewidth=3,label="boundary");
end


# plot data on top of contour plot
function plotdata(xsave,c)
   str = "c="*string(c);
   plot!(xsave[1,:], xsave[2,:], lw = 2, marker=2, label = str);
end



#============  Optimization functions ============#

#
# steepest descent algorithm, with constant step size
# input: 
#    x0 = initial point, a 2-vector (e.g. x0=[1;2])
#    alpha = step size. Constant, in this algorithm.
#    c = penalty parameter
# output: 
#    xsave = list of points
#
function SteepestDescent(x0,alpha,c)

   # function to minimize: fix c parameter to desired value
   F = Base.Fix2(Fc,c);
   DF = Base.Fix2(DFc,c);


   # setup for steepest descent
   x = x0;
   successflag = false;
   xsave = zeros(2,MaxIter+1);
   xsave[:,1] = x0;

   # perform steepest descent iterations
   for iter = 1:MaxIter

       # compute gradient
       Fgrad = DF(x);

       # print info
       @printf("x = %11.10f, %11.10f, F(x) = %10.8f, |grad F| = %10.8f \n", x[1],x[2],F(x),sqrt(Fgrad'*Fgrad));

       # check if gradient is small enough
       if sqrt(Fgrad'*Fgrad) < tol
          @printf("Converged after %d iterations, function value %f\n", iter, F(x))
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





