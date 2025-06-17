 """
     gradient_descent!(UDE, kwargs ...)

 Minimizes the loss function of the `UDE` model with the gradient descent algorithm with a step size of `step_size` and a maximum number of iterations of `maxiter`. Prints the value of the loss function after each iteration when `maxiter` is true.

  # kwargs

- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed? Default is `false`.
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
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter)

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
  sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter)

  # assign parameters to model

  return sol.u
end


# adding time steps to skip predictions to accommodate data sets with large gaps
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
  sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter)

  # assign parameters to model
  UDE.parameters = sol.u

  return nothing
end

 """
     BFGS!(UDE, kwargs ...)

 Minimizes the loss function of the `UDE` model using the BFGS algorithm is the initial step norm equal to `initial_step_norm`. The function will print the value of the loss function after each iteration when `verbose` is true.

  # kwargs

- `initial_step_norm`: Initial step norm for BFGS algorithm. Default is `0.01`.
- `verbose`: Should the training loss values be printed? Default is `false`.
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

# adding time steps to skip predictions to accommodate data sets with large gaps
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

 Performs Bayesian estimation on the parameters of a UDE using the No U-Turn Sampler (NUTS) algorithm.

 # kwargs

- `delta`: Step size used in NUTS adaptor. Default is `0.45`.
- `samples`: Number of parameters sampled. Default is `500`.
- `burnin`: Number of samples used as burn-in of Bayesian algorithm. Default is `samples/10`.
- `verbose`: Should the training loss values be printed? Default is `true`.
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

 Performs Bayesian estimation on the parameters of an UDE using the Stochastic Gradient Langevin Dynamics (SGLD) sampling algorithm. At each step `t`, the stochastic update is provided by a random variable ε with mean 0 and noise η.

 ```math
 ϵ = a*(b + t-1)^-γ
 ```

 # kwargs

- `a`: Default is `10.0`.
- `b`: Default is `1000`.
- `γ`: Default is 0.9.
- `samples`: Number of parameters sampled. Default is `500`.
- `burnin`: Number of samples used as burn-in of Bayesian algorithm. Default is `samples/10`.
- `verbose`: Should the training loss values be printed? Default is `true`.
 """
function SGLD!(UDE::BayesianUDE;samples = 500, burnin = Int(samples/10), a = 10.0, b = 1000, γ = 0.9, verbose = true)
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
  elseif loss_function == "neural gradient matching"
    return (step_size = 0.05, maxiter = 500)
  elseif loss_function == "spline gradient matching"
    return (step_size = 0.05, maxiter = 500)
  end
end




"""
 train!(UDE::UDE; kwargs...)

This function provides access to several training routines for UDE models. The user provides the UDE model object and can then choose between several loss functions and optimization algorithms using the keyword arguments.
The training routine will update the UDE object with the trained parameters and return any other useful quantities estimated during the training procedure.
The default optimizer trains the UDE model by minimizing the loss function using the ADAM gradient descent algorithm for `maxiter` steps of size `step_size`.
Five loss functions are available using the `loss_function` argument: conditional likelihood, marginal likelihood, derivative matching, shooting, and multiple shooting.
The `loss_options` and `optim_options` arguments are named tuples that can be used to pass parameters to the training routine.


# kwargs
- `loss_function`: Determines the loss function used to train the model and defaults to "derivative matching" for efficiency.
- `verbose`: If true, the value of the loss function will print between each step of the optimizer.
- `regularization_weight`: Weight given to regularization in the loss function. Default is 0.
- `optimizer`: Determines the optimization algorithm used to train the model and defaults to "ADAM" for gradient descent.
- `loss_options`: Named tuple with keyword arguments to help construct the loss function.
- `optim_options`: Named tuple with keyword arguments to pass to the optimization algorithm. For ADAM, these are `maxiter`, the number of iterations used to run the algorithm, and `step_size`, the size of each iteration.

# Loss Functions
Users can choose from one of five loss functions: conditional likelihood, marginal likelihood, derivative matching, shooting, and multiple shooting.

| Loss Function          | Discrete Model | Continuous Model | Speed    |
|------------------------|----------------|------------------|----------|
| Conditional likelihood | Yes            | Yes              | Moderate |
| Marginal likelihood    | Yes            | Yes              | Slow     |
| Derivative matching    | No             | Yes              | Fast     |
| Shooting               | No             | Yes              | Moderate |
| Multiple shooting      | No             | Yes              | Moderate |


## Conditional likelihood:
To use the conditional likelihood set the keyword argument `loss_function = "conditional likelihood"`.

This option trains the UDE model while accounting for imperfect observations and process uncertainty by maximizing the conditional likelihood of a state-space model where the UDE is used as the process model.
The conditional likelihood is faster to compute, but can be less accurate than the marginal likelihood.

## Marginal likelihood:
To use the marginal likelihood set the keyword argument `loss_function = "marginal likelihood"`.

This option maximizes the marginal likelihood of a state-space model, which is approximated using an unscented Kalman filter.
This option is slower than the conditional likelihood but should, in theory, increase the accuracy of the trained model (i.e., reduce bias).

### loss_options
- `process_error`: An initial estimate of the level of process error. Default is 0.1.
- `observation_error`: The level of observation error in the data set. There is no default, so it will throw an error if not provided.
- `α`: Parameter for the Kalman filter algorithm. Default is 10^-3.
- `β`: Parameter for the Kalman filter algorithm. Default is 2.
- `κ`: Parameter for the Kalman filter algorithm. Default is 0.


## Derivative matching:
To use the derivative matching training routine set `loss_function = "derivative matching"`.

This function trains the UDE model in a two-step process. First, a smoothing function is fit to the data using a spline regression.
Then, the UDE model is trained by comparing the derivatives of the smoothing functions to the derivatives predicted by the right-hand side of the UDE.
This training routine is much faster than the alternatives, but may be less accurate.

### loss_options
- `d`: The number of degrees of freedom in the curve fitting function. Defaults to 12.
- `alg`: The algorithm used to fit the curve to the data set. See the DataInterpolations package for details. Defaults to generalized cross-validation `:gcv_svd`.
- `remove_ends`: The number of data points to leave off of the end of the data set when training the UDE to reduce edge effects from the curve fitting process. Defaults to 0.


## Shooting:
To use the shooting training routine set `loss_function = "shooting"`.

This option calculates the loss by solving the ODE from the initial to the final data point and comparing the observed to the predicted trajectory with mean squared error (MSE).
The initial data point is estimated as a free parameter to reduce the impacts of observation error.


## Multiple shooting:
To use the multiple shooting training routine set `loss_function = "multiple shooting"`.

This option calculates the loss by breaking the data into blocks of sequential observations. It then uses the UDE to forecast from the initial data point in each block to the first data point in the next block.
The loss is defined as the MSE between the forecasts and the data points.
The initial data point in each block is estimated as a free parameter to reduce the impacts of observation error.

### loss_options
- `pred_length`: The number of data points in each block. The default is 10.


# Optimization Algorithms

The method used to minimize the loss function. Two options are available: ADAM and BFGS.
ADAM is a first-order gradient descent algorithm, while BFGS is a quasi-Newton method that uses approximate second-order information.

The user can specify the maximum number of iterations `maxiter` for each algorithm using the `optim_options` keyword argument.
For ADAM the `optim_options` can be used to specify the step size `step_size` and for BFGS you can specify the initial step norm `initial_step_norm`.
"""
function train!(UDE::UDE;
  t_skip = NaN,
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

    # set defualts for process and observaiton errors 
    observation_error = 0.025
    process_error = 0.025

    # if observation errors are provided update with errorts to matrix function 
    L = size(UDE.data)[1]
    if any(keys(loss_options) .== :observation_error)
      observation_error = errors_to_matrix(loss_options.observation_error, L)
    end 

     # if process errors are provided update with errorts to matrix function 
     L = size(UDE.data)[1]
     if any(keys(loss_options) .== :process_error)
       process_error = errors_to_matrix(loss_options.process_error, L)
     end 

    # initialize loss and params so they can be updated with in an if else
    loss, params, _  = (0,0,0)

    if isnan(t_skip)
      loss, params, _  = conditional_likelihood(UDE,regularization_weight,  observation_error, process_error)
    else
      loss, params, _  = conditional_likelihood(UDE,t_skip,regularization_weight, observation_error, process_error)
    end
    
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
      throw("Marginal likelihood requires observation errors.")
    end

    new_options = ComponentArray(loss_options)
    options = ComponentArray((α = 10^-3, β = 2,κ = 0))
    inds  = broadcast(i -> !(keys(new_options)[i] in [:process_error,:observation_error]), 1:length(keys(new_options)))
    keys_ = keys(new_options)[inds]
    options[keys_] .= new_options[keys_]

    
    loss, params, uhat  = (0,0,0)
    
    if isnan(t_skip)
      loss, params, uhat = marginal_likelihood(UDE,regularization_weight,Pν,Pη,options.α,options.β,options.κ)

    else
      loss, params, uhat = marginal_likelihood(UDE,t_skip,regularization_weight,Pν,Pη,options.α,options.β,options.κ)

    end
    
  elseif (loss_function == "derivative matching") | (loss_function == "gradient matching")
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (i.e., CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'.")
    end
    new_options = ComponentArray(loss_options)
    options = ComponentArray((d = 12, remove_ends = 0))
    options[keys(new_options)] .= new_options

    loss, params, uhat = derivative_matching_loss(UDE, regularization_weight; d = options.d, alg = :gcv_svd, remove_ends = options.remove_ends)

  elseif loss_function == "shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (i.e., CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'.")
    end
    loss, params, _ = shooting_loss(UDE)
    uhat = UDE.data

  elseif loss_function == "multiple shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (i.e., CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'.")
    end
    new_options = ComponentArray(loss_options)
    options = ComponentArray((pred_length = 5))
    options[keys(new_options)] .= new_options

    loss, params, _  = multiple_shooting_loss(UDE,regularization_weight,options.pred_length)
    uhat = UDE.data

  elseif loss_function == "neural gradient matching"

    new_options = ComponentArray(loss_options)
    options = ComponentArray((σ=0.1, τ = 0.5, init_weights_layer_1 = 5.0))
    options[keys(new_options)] .= new_options
    loss, params, interp  = neural_gradient_matching_loss(UDE, options.σ, options.τ, regularization_weight, options.init_weights_layer_1)
    interpNN, interp_states = interp

  elseif loss_function == "spline gradient matching"

    new_options = ComponentArray(loss_options)
    options = ComponentArray((σ=0.05^2, τ = 0.025^2,  T = 100))
    options[keys(new_options)] .= new_options
    loss, params, interp  = spline_gradient_matching_loss(UDE, options.σ, options.τ, regularization_weight, options.T)

  else
    throw("Select a valid loss function. Choose from 'conditional likelihood', 'marginal likelihood', 'gradient matching', 'shooting', or 'multiple shooting'.")

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
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(options.step_size), callback = callback, maxiters = options.maxiter)
    if loss_function == "marginal likelihood"
      Pν = sol.u.Pν
      UDE.parameters = sol.u.UDE
    elseif loss_function == "neural gradient matching"
      UDE.parameters = sol.u.UDE
    elseif loss_function == "spline gradient matching"
      UDE.parameters = sol.u.UDE
    else
      UDE.parameters = sol.u
    end
  elseif optimizer == "BFGS"

    new_options = ComponentArray(optim_options)
    options = ComponentArray((initial_step_norm = 0.01, ))
    options[keys(new_options)] .= new_options

    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = options.initial_step_norm);
        callback, allow_f_increases = false)

    # assign parameters to model
  else
    throw("Select a valid optimization algorithm. Choose from 'ADAM' or 'BFGS'.")

  end

  # update model parameters


  # states
  out = nothing
  if (loss_function == "derivative matching") | (loss_function == "gradient matching")
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

    elseif loss_function == "neural gradient matching"
      t = reshape(UDE.times,1,length(UDE.times))
      UDE.parameters.uhat .= Lux.apply(interpNN, t, sol.u.interp, interp_states)[1]

    elseif loss_function == "spline gradient matching"
      UDE.parameters.uhat .= interp_states_spline(interp,sol.u.α)
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
  if loss_function =="conditional likelihood" 

    # new_options = ComponentArray(loss_options)
    # options = ComponentArray((observation_error = 0.025, process_error = 0.025))
    # options[keys(new_options)] .= new_options

    new_options = ComponentArray(loss_options)
    options = ComponentArray((observation_error = 0.025, process_error = 0.025))
    inds  = broadcast(i -> !(keys(new_options)[i] in [:process_error,:observation_error]), 1:length(keys(new_options)))
    keys_ = keys(new_options)[inds]
    options[keys_] .= new_options[keys_]

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
      throw("Marginal likelihood requires observation errors. \n Use `loss_options = (observation_error = ...,)` to specify its value.")
    end

    new_options = ComponentArray(loss_options)
    options = ComponentArray((α = 10^-3, β = 2, κ = 0))
    inds  = broadcast(i -> !(keys(new_options)[i] in [:process_error,:observation_error]), 1:length(keys(new_options)))
    keys_ = keys(new_options)[inds]
    options[keys_] .= new_options[keys_]

    loss, params, uhat = marginal_likelihood(UDE,regularization_weight,Pν,Pη,options.α,options.β,options.κ)

  elseif (loss_function == "derivative matching") | (loss_function == "gradient matching")
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (i.e., CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'.")
    end
    new_options = ComponentArray(loss_options)
    options = ComponentArray((d = 12, remove_ends = 0))
    options[keys(new_options)] .= new_options

    loss, params, uhat = derivative_matching_loss(UDE, regularization_weight; d = options.d, alg = :gcv_svd, remove_ends = options.remove_ends)

  elseif loss_function == "shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (i.e., CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'.")
    end
    loss, params, _ = shooting_loss(UDE,regularization_weight)
    uhat = UDE.data

  elseif loss_function == "multiple shooting"
    if UDE.solvers == nothing
      throw("This method does not work with discrete time models (i.e., CustomDifference), please select from 'conditional likelihood' or 'marginal likelihood'.")
    end
    new_options = ComponentArray(loss_options)
    options = ComponentArray((pred_length = 5))
    options[keys(new_options)] .= new_options

    loss, params, _  = multiple_shooting_loss(UDE,regularization_weight,options.pred_length)
    uhat = UDE.data
  else
    throw("Select a valid loss function. Choose from 'conditional likelihood', 'marginal likelihood', 'gradient matching', 'shooting', or 'multiple shooting'. ")
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
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(options.step_size), callback = callback, maxiters = options.maxiter)
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

    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = options.initial_step_norm);
        callback, allow_f_increases = false)

    # assign parameters to model
  else
    throw("Select a valid optimization algorithm. Choose from 'ADAM' or 'BFGS'.")
  end

  # update model parameters


  # states
  out = nothing
  if (loss_function == "derivative matching") | (loss_function == "gradient matching")
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
      x, Px =ukf_smooth(UDE,sol.u,H,Pη,L,α,β,κ)
      out = (Pν = Pν, Px = Px)
      UDE.parameters.uhat = x

    end

  return out

end
