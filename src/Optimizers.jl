

# """
#     gradient_decent!(UDE, kwargs ...)

# Minimizes the loss funtction of the `UDE` model with the gradent decent algorithm with a step size of `step_size` and a maimum number of itrations of `maxiter`. prints the value fo the loss funciton after each iteration when `maxiter` is true.   
# """
function gradient_decent!(UDE; step_size = 0.05, maxiter = 500, verbose = false, verbos = false)
    
    # set optimization problem 
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
    if verbos
      verbose = true
      @warn ("kwarg: verbos is depricated use verbose")
    end 

    if verbose
        callback = function (p, l; doplot = false)
          print(round(l,digits = 3), " ")
          return false
        end
    else
        callback = function (p, l; doplot = false)
          return false
        end 
    end

    # run optimizer
    sol = Optimization.solve(optprob, OptimizationOptimisers.ADAM(step_size), callback = callback, maxiters = maxiter )
    
    # assign parameters to model 
    UDE.parameters = sol.u
    
    return nothing
end



# """
#     BFGS!(UDE, kwargs ...)

# minimizes the loss function of the `UDE` model using the BFGS algorithm is the inital step norm equal to `initial_step_norm`. The funciton will print the value fo the loss function after each iteration when `verbose` is true.  
# """
function BFGS!(UDE; verbos = false,verbose = false, initial_step_norm = 0.01)
    if verbos
      verbose = true
      @warn ("kwarg: verbos is depricated use verbose")
    end 
    if verbose
        callback = function (p, l; doplot = false)
          print(round(l,digits = 3), " ")
          return false
        end
    else
        callback = function (p, l; doplot = false)
          return false
        end 
    end
    
    
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)

    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = initial_step_norm);
        callback, allow_f_increases = false)

    # assign parameters to model 
    UDE.parameters = sol.u
    
end 