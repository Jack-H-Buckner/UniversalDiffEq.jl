# """
#     gradient_descent!(UDE, kwargs ...)

# Minimizes the loss function of the `UDE` model with the gradient descent algorithm with a step size of `step_size` and a maximum number of iterations of `maxiter`. Prints the value of the loss function after each iteration when `maxiter` is true.   
# """
function gradient_descent!(UDE; step_size = 0.05, maxiter = 500, verbose = false, verbos = false)
    
    # set optimization problem 
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
    if verbos
      verbose = true
      @warn ("kwarg: verbos is deprecated use verbose")
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
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter )
    
    # assign parameters to model 
    UDE.parameters = sol.u
    
    return nothing
end

# adding time steps to skip predictiosn for to accomidate data sets with large gaps
function gradient_descent!(UDE,t_skip; step_size = 0.05, maxiter = 500, verbose = false, verbos = false)
    
  # set optimization problem 
  target = (x,p) -> UDE.loss_function(x,t_skip)
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
  sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter )
  
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

# adding time steps to skip predictiosn for to accomidate data sets with large gaps
function BFGS!(UDE,t_skip; verbos = false,verbose = false, initial_step_norm = 0.01)
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
  
  
  target = (x,p) -> UDE.loss_function(x,t_skip)
  adtype = Optimization.AutoZygote()
  optf = Optimization.OptimizationFunction(target, adtype)
  optprob = Optimization.OptimizationProblem(optf, UDE.parameters)

  sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = initial_step_norm);
      callback, allow_f_increases = false)

  # assign parameters to model 
  UDE.parameters = sol.u
  
end 

# """
#     NUTS!(UDE, kwargs ...)

# performs Bayesian estimation on the parameters of an UDE using the NUTS sampling algorithm
# """
function NUTS!(UDE::BayesianUDE;delta = 0.45,samples = 500, burnin = Int(samples/10), verbose = true)
  UDE.parameters = UDE.parameters[end]

  target = (x,p) -> UDE.loss_function(x) * UDE.times
  l(θ) = sum(abs2,-target(θ,nothing) .- sum(θ .* θ))
  function dldθ(θ)
    x, λ = Zygote.pullback(l,θ)
    grad = first(λ(1))
    return x, grad
  end

  metric = DiagEuclideanMetric(length(UDE.parameters))
  h = Hamiltonian(metric, l, dldθ)
  integrator = Leapfrog(find_good_stepsize(h, UDE.parameters))
  kernel = HMCKernel(Trajectory{MultinomialTS}(integrator,GeneralisedNoUTurn()))
  adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(delta, integrator))
  draws, stats = sample(h, kernel, UDE.parameters, samples, adaptor, burnin; progress = verbose)

  # assign parameters to model 
  UDE.parameters = draws
end

# """
#     SGLD!(UDE, kwargs ...)

# performs Bayesian estimation on the parameters of an UDE using the SGLD sampling algorithm
# """
function SGLD!(UDE::BayesianUDE;samples = 500, burnin = Int(samples/10),a = 10.0, b = 1000, γ = 0.9, verbose = true)
  UDE.parameters = UDE.parameters[end]

  target = (x,p) -> sum(abs2,UDE.loss_function(x) * UDE.times)

  parameters = Vector{typeof(UDE.parameters)}(undef, samples+1)
  parameters[1] = UDE.parameters
  
  for t in 2:(samples+1)
    dL = Zygote.gradient(x -> target(x,nothing), parameters[t-1])
    ϵ = a*(b + t-1)^-γ
    η = ϵ.*(randn(size(parameters[t-1])))

    parameters[t] = parameters[t-1] .- .5ϵ*dL[1] + η
    if verbose
      print(round(target(parameters[t],nothing),digits = 3)," ")
    end
  end

  UDE.parameters = parameters[(end-burnin):end]
end