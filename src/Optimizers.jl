 """
     gradient_descent!(UDE, kwargs ...)

 Minimizes the loss function of the `UDE` model with the gradient descent algorithm with a step size of `step_size` and a maximum number of iterations of `maxiter`. Prints the value of the loss function after each iteration when `maxiter` is true.   

  # kwargs

- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
 """
function gradient_descent!(UDE; step_size = 0.05, maxiter = 500, verbose = false)
    
    # set optimization problem 
    target = (x,p) -> UDE.loss_function(x)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
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


function gradient_descent_permuted!(UDE,B,ϵ; step_size = 0.05, maxiter = 500, verbose = false)
    
  # set optimization problem 
  target = (x,p) -> UDE.permuted_loss_function(x,B,ϵ)
  adtype = Optimization.AutoZygote()
  optf = Optimization.OptimizationFunction(target, adtype)
  optprob = Optimization.OptimizationProblem(optf, UDE.initial_parameters)
  
  # print value of loss function at each time step 
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

  return sol.u
end


# adding time steps to skip predictiosn for to accomidate data sets with large gaps
function gradient_descent!(UDE,t_skip; step_size = 0.05, maxiter = 500, verbose = false)
    
  # set optimization problem 
  target = (x,p) -> UDE.loss_function(x,t_skip)
  adtype = Optimization.AutoZygote()
  optf = Optimization.OptimizationFunction(target, adtype)
  optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
  
  # print value of loss function at each time step 
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

 """
     BFGS!(UDE, kwargs ...)

 minimizes the loss function of the `UDE` model using the BFGS algorithm is the inital step norm equal to `initial_step_norm`. The funciton will print the value fo the loss function after each iteration when `verbose` is true.  

  # kwargs

- `initial_step_norm`: Initial step norm for BFGS algorithm. Default is `0.01`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
 """
function BFGS!(UDE; verbose = false, initial_step_norm = 0.01)
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
function BFGS!(UDE,t_skip; verbose = false, initial_step_norm = 0.01)
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

 """
     NUTS!(UDE, kwargs ...)

 performs Bayesian estimation on the parameters of an UDE using the NUTS sampling algorithm.

 # kwargs

- `delta`: Step size used in NUTS adaptor. Default is `0.45`.
- `samples`: Number of parameters sampled. Default is `500`.
- `burnin`: Number of samples used as burnin of Bayesian algorithm. Default is `samples/10`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
 """
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

 """
     SGLD!(UDE, kwargs ...)

 Performs Bayesian estimation on the parameters of an UDE using the SGLD sampling algorithm. At each step `t`, the stochastic update is provided by a random variable ε with mean 0 and variance:

 ```math
a(b + t-1)^γ
 ```

 # kwargs
 
- `a`: Default is `10.0`.
- `b`: Default is `1000`.
- `γ`: Default is 0.9.
- `samples`: Number of parameters sampled. Default is `500`.
- `burnin`: Number of samples used as burnin of Bayesian algorithm. Default is `samples/10`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
 """
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


 


function default_options(loss_function)
  if loss_function == "conditional likelihood"
    return (step_size = 0.025, maxiter = 500)
  elseif loss_function == "marginal likelihood"
    return (step_size = 0.005, maxiter = 1000)
  elseif loss_function == "derivative matching"
    return (step_size = 0.05, maxiter = 500)
  elseif loss_function == "shooting"
    return(step_size = 0.05, maxiter = 500)
  elseif loss_function == "multiple shooting"
    return (step_size = 0.05, maxiter = 500)
  end
end 




"""
 train!(UDE::UDE;  kwargs...)

This funciton provides access to several training routines for UDE models. The user provides the UDE model object and can then choose between several loss funcitons and optimization algorthms using the key work arguments. 
The training routine will update the UDE object with the trained paramters and return any other useful quantities estimated during the training procedure. 
The options 

Trains the UDE model by minimizing the loss function using the ADAM gradient descent algorithm for `maxiter` step of size `step_size.`
Four loss functions are available using the `loss_function` argument: conditional likelihood, marginal likelihood, derivative matching, shooting, and multiple shooting.
The `options` argument is a named tuple that can be used to pass parameters to the training routine. 


# kwargs
- `loss_function`: Determines the loss function used to train the model and defaults to  "derivative matching."
- `verbose`: If true, the value of the loss function will print between each set of the optimizer.
- `maxiter`: The number of iterations used to run the ADAM gradient descent algorithm.
- `step_size`: The number of steps to run the ADAM algorithm
- `loss_options`: a named tuple with keyword arguments to help construct the loss function.
- `optim_options`: a named tuple with key word arguments to pass to the optiizaiton algorithm. 

# loss function options
Users can choose from one of four loss functions: conditional likelihood, marginal likelihood, derivative matching, and mini batching.


## Conditional likelihood:
To use the conditional likelihood as set the keyword argument  `loss_function = "conditional likelihood"`.

This option trains the UDE model while accounting for imperfect observaitons and process uncertainty by maximizing the conditional likelihood of a state space model where the UDE is used as the process model.
The conditional likelihood is faster to compute, but can be less accurate than the marginal likelihood.


## Marginal likelihood:
To use the marginal likelihood as the keyword argument `loss_function = "marginal likelihood"`.

This option maximizes the marginal likelihood of a state space model, which is approximated using an unscented Kalman filter.
This option is slower than the conditional likelihood but should, in theory, increase the accuracy of the trained model (i.e. reduce bias). 

### loss_options
- `process_error`: an initial estimate of the level of process error. Default is 0.1. 
- `observation_error`: The level of observation error in the data set. No default will throw an error if not provided. 
- `α`: parameter for the knalmand filter algorithm. Defualt is 10^-3.
- `β`: parameter for the knalmand filter algorithm. Defualt is 2.
- `κ`: parameter for the knalmand filter algorithm. Defualt is 2.

## Derivative matching:
To use the derivative matching training routine set  `loss_function = "derivative matching".`
This function trains the UDE model in a two-step process. First, a smoothing function is fit to the the data using a spline regression.
Then, the UDE model is trained by comparing the derivatives of the smoothing functions to the derivatives predicted by the right-hand side of the UDE.
this training routine is much faster than the alternative, but may be less accurate.

### options
- `d`:  the number of degrees of freedom in the curve fitting function defaults to 12.
- `alg`:  the algorithm used to fit the curve to the data set see the DataInterpolations package for details it, defaults to generalized cross validation `:gcv_svd`
- `remove_ends`: The number of data points to leave off of the end of the data set when training the UDE to reduce edge effects from the curve fitting process defaults to 0

## Shooting:
This option calculates the loss by solving the ODE from the initial to the final data point and comparing the observed to the predicted trajectory with MSE.
The initial data point is estimated as a free parameter to reduce the impacts of observaiton error.

## multiple shooting:
This option calculates the loss by breaking the data into blocks of sequential observations. It then uses the UDE to forecast from the initial data point in each block to the first data point in the next block.
The loss is defined as the mean squared error between the forecasts and the data points.
The initial data point in each block is estimated as a free parameter to reduce the impacts of observaiton error.

### options
- `pred_length`: The number of data points in each block, default, is 10.


# Optimization algorithms

The method used to minimize the loss function. Two options are available, ADAM and BFGS. ADAM is a first order gradient descent
algorthim and BFGS uses aproximate second order information. 

The user can specify the maximum number of iteration `maxiter` for each algoritmh using the `optim_options` key word argument. 
For ADAM the `optim_options` can be used to specify the step size and for BFGS you can sepcify the initial step norm. 
"""
function train!(UDE::UDE; 
  loss_function = "derivative matching", 
  optimizer = "ADAM",
  regularization_weight = 0.0, 
  verbose = true, 
  loss_options = NamedTuple(),
  optim_options = NamedTuple())

  # set up loss function 
  loss = x -> 0
  params = UDE.parameters
  uhat = 0
  if loss_function == "conditional likelihood"

    new_options = ComponentArray(loss_options)
    options = ComponentArray((observation_error = 0.025, process_error = 0.025))
    options[keys(new_options)] .= new_options

    loss, params, _  = conditional_likelihood(UDE,regularization_weight,  options.observation_error, options.process_error)

  elseif loss_function == "marginal likelihood"

    L = size(UDE.data)[1]
    Pν = errors_to_matrix(0.1, L)
    if :process_error in keys(loss_options)
      Pν = errors_to_matrix(loss_options.process_error, L)
    end 

    Pη = 0
    if :observation_error in keys(loss_options)
      Pη = errors_to_matrix(loss_options.observation_error, L)
    else 
      throw("Marginal likelihood requires observation errors")
    end 

    new_options = ComponentArray(loss_options)
    options = ComponentArray((α = 10^-3, β = 2,κ = 0))
    inds  = broadcast(i -> !(keys(new_options)[i] in [:process_error,:observation_error]), 1:length(keys(new_options)))
    keys_ = keys(new_options)[inds]
    options[keys_] .= new_options[keys_]

    loss, params, uhat = marginal_likelihood(UDE,regularization_weight,Pν,Pη,options.α,options.β,options.κ)

  elseif loss_function == "derivative matching"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (e.g. CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'. ")
    end 
    new_options = ComponentArray(loss_options)
    options = ComponentArray((d = 12, remove_ends = 0))
    options[keys(new_options)] .= new_options

    loss, params, uhat = derivative_matching_loss(UDE, regularization_weight; d = options.d, alg = :gcv_svd, remove_ends = options.remove_ends)

  elseif loss_function == "shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (e.g. CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'. ")
    end 
    loss, params, _ = shooting_loss(UDE)
    uhat = UDE.data

  elseif loss_function == "multiple shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (e.g. CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'. ")
    end 
    new_options = ComponentArray(loss_options)
    options = ComponentArray((pred_length = 5))
    options[keys(new_options)] .= new_options

    loss, params, _  = multiple_shooting_loss(UDE,regularization_weight,options.pred_length)
    uhat = UDE.data
  else 
    throw("Selelect a valid loss function. Choose from, 'conditional likelihood', 'marginal likelihood', 'derivative matching', 'shooting', or 'multiple shooting' ")

  end

  # optimize loss function
  # set optimization problem 
  target = (x,p) -> loss(x)
  adtype = Optimization.AutoZygote()
  optf = Optimization.OptimizationFunction(target, adtype)
  optprob = Optimization.OptimizationProblem(optf,params)
  
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

  sol = 0 
  Pν = nothing
  if optimizer == "ADAM"

    new_options = ComponentArray(optim_options)
    options = ComponentArray(default_options(loss_function))
    options[keys(new_options)] .= new_options

    # run optimizer
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(options.step_size), callback = callback, maxiters = options.maxiter )
    if loss_function == "marginal likelihood"
      Pν = sol.u.Pν
      UDE.parameters = sol.u.UDE
    else
      UDE.parameters = sol.u
    end
  elseif optimizer == "BFGS"

    new_options = ComponentArray(optim_options)
    options = ComponentArray((initial_step_norm = 0.01, ))
    options[keys(new_options)] .= new_options

    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = initial_step_norm);
        callback, allow_f_increases = false)

    # assign parameters to model 
  else 
    throw("Selelect a valid optimization algorithm. Choose from, 'ADAM', or 'BFGS'. ")

  end 

  # update model parameters 
  

  # states
  out = nothing
  if loss_function == "derivative matching"
      UDE.parameters.uhat .= uhat

    elseif loss_function == "shooting"
      UDE.parameters.uhat .= shooting_states(UDE)

    elseif loss_function == "multiple shooting"
      new_options = ComponentArray(loss_options)
      options = ComponentArray((pred_length = 5))
      options[keys(new_options)] .= new_options

      UDE.parameters.uhat = multiple_shooting_states(UDE,options.pred_length)
    
    elseif loss_function == "marginal likelihood"
    
      H,Pν,Pη,L,α,β,κ = uhat
      Pν = sol.u.Pν*sol.u.Pν'
      f = (u,t,dt,p) -> UDE.process_model.predict(u,t,dt,p)[1]
      x, Px =ukf_smoothing(UDE.data,UDE.times,f,sol.u.UDE.process_model,H,Pν,Pη,L,α,β,κ)
      out = (Pν = Pν, Px = Px)
      UDE.parameters.uhat = x
    
    end

    return out
    
end 





function train!(UDE::MultiUDE; 
  loss_function = "derivative matching", 
  optimizer = "ADAM",
  regularization_weight = 0.0, 
  verbose = true, 
  loss_options = NamedTuple(),
  optim_options = NamedTuple())

  # set up loss function 
  loss = x -> 0
  params = UDE.parameters
  uhat = 0
  if loss_function == "conditional likelihood"

    new_options = ComponentArray(loss_options)
    options = ComponentArray((observation_error = 0.025, process_error = 0.025))
    options[keys(new_options)] .= new_options

    loss, params, _  = conditional_likelihood(UDE,regularization_weight,  options.observation_error, options.process_error)

  elseif loss_function == "marginal likelihood"

    L = size(UDE.data)[1]
    Pν = errors_to_matrix(0.1, L)
    if :process_error in keys(loss_options)
      Pν = errors_to_matrix(loss_options.process_error, L)
    end 

    Pη = 0
    if :observation_error in keys(loss_options)
      Pη = errors_to_matrix(loss_options.observation_error, L)
    else 
      throw("Marginal likelihood requires observation errors. \n Use `loss_options = (observation_error = ...,)` to sepciyg its value.")
    end 

    new_options = ComponentArray(loss_options)
    options = ComponentArray((α = 10^-3, β = 2,κ = 0))
    inds  = broadcast(i -> !(keys(new_options)[i] in [:process_error,:observation_error]), 1:length(keys(new_options)))
    keys_ = keys(new_options)[inds]
    options[keys_] .= new_options[keys_]

    loss, params, uhat = marginal_likelihood(UDE,regularization_weight,Pν,Pη,options.α,options.β,options.κ)

  elseif loss_function == "derivative matching"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (e.g. CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'. ")
    end 
    new_options = ComponentArray(loss_options)
    options = ComponentArray((d = 12, remove_ends = 0))
    options[keys(new_options)] .= new_options

    loss, params, uhat = derivative_matching_loss(UDE, regularization_weight; d = options.d, alg = :gcv_svd, remove_ends = options.remove_ends)

  elseif loss_function == "shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (e.g. CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'. ")
    end 
    loss, params, _ = shooting_loss(UDE,regularization_weight)
    uhat = UDE.data

  elseif loss_function == "multiple shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (e.g. CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'. ")
    end 
    new_options = ComponentArray(loss_options)
    options = ComponentArray((pred_length = 5))
    options[keys(new_options)] .= new_options

    loss, params, _  = multiple_shooting_loss(UDE,regularization_weight,options.pred_length)
    uhat = UDE.data
  else 
    throw("Selelect a valid loss function. Choose from, 'conditional likelihood', 'marginal likelihood', 'derivative matching', 'shooting', or 'multiple shooting'. ")
  end

  # optimize loss function
  # set optimization problem 
  target = (x,p) -> loss(x)
  adtype = Optimization.AutoZygote()
  optf = Optimization.OptimizationFunction(target, adtype)
  optprob = Optimization.OptimizationProblem(optf,params)
  
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

  sol = 0 
  Pν = nothing
  if optimizer == "ADAM"

    new_options = ComponentArray(optim_options)
    options = ComponentArray(default_options(loss_function))
    options[keys(new_options)] .= new_options

    # run optimizer
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(options.step_size), callback = callback, maxiters = options.maxiter )
    if loss_function == "marginal likelihood"
      Pν = sol.u.Pν * sol.u.Pν'
      UDE.parameters = sol.u.UDE
    else
      UDE.parameters = sol.u
    end
  elseif optimizer == "BFGS"

    new_options = ComponentArray(optim_options)
    options = ComponentArray((initial_step_norm = 0.01, ))
    options[keys(new_options)] .= new_options

    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = initial_step_norm);
        callback, allow_f_increases = false)

    # assign parameters to model
  else 
    throw("Selelect a valid optimization algorithm. Choose from, 'ADAM', or 'BFGS'. ") 
  end 

  # update model parameters 
  

  # states
  out = nothing
  if loss_function == "derivative matching"
      UDE.parameters.uhat .= uhat

    elseif loss_function == "shooting"
      UDE.parameters.uhat .= shooting_states(UDE)

    elseif loss_function == "multiple shooting"
      new_options = ComponentArray(loss_options)
      options = ComponentArray((pred_length = 5))
      options[keys(new_options)] .= new_options

      UDE.parameters.uhat = multiple_shooting_states(UDE,options.pred_length)
    
    elseif loss_function == "marginal likelihood"
    
      H,Pη,L,α,β,κ = uhat
      f = (u,t,dt,p) -> UDE.process_model.predict(u,t,dt,p)[1]
      x, Px =ukf_smooth(UDE,sol.u,H,Pη,L,α,β,κ)
      out = (Pν = Pν, Px = Px)
      UDE.parameters.uhat = x
    
    end

  return out
  
end 
