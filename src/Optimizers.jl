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


"""
  one_step_ahead!(UDE::UDE, kwargs...)

Trains the model `UDE` using a modified version of the loss function where the estimated value of the state variables 
are fixed at the value fo the observations `uhat = y`. The model is then trined to minimized the differnce between the prediced
and observed changes in the data sets using the ADAM gradient descent algorithm.  

# kwargs
- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
"""
function one_step_ahead!(UDE::UDE; verbose = true, maxiter = 500, step_size = 0.05)

  loss = init_one_step_ahead_loss(UDE)
  target = (x,u) -> loss(x)
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
end


function one_step_ahead!(UDE::MultiUDE; verbose = true, maxiter = 500, step_size = 0.05)

  loss = init_one_step_ahead_loss(UDE)
  target = (x,u) -> loss(x)
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
  UDE.parameters.uhat .= UDE.data
end


"""
  mini_batching!(UDE::UDE, kwargs...)

Trains the UDE model using the mini batching algorithm. Mini batching breaks the data set up into blocks of consequtive observaitons of 
length `pred_length`. The model predicts the value at each time point in the block by solving the ODE using the first data point as the 
initial conditions up to the time of the final data point in the next block. The loss is calcualted by comparing the predicted and observed values
using the mean squared error. 

Longer block lengths may increase the speed of training by allowing the ODE solvers to find efficent integation schemes. However. long step sizes can 
create local minima in the loss funciton on data sets with oscilations or orther forms of variability.  

# kwargs
- `pred_length`: Number of observations in each block. Default is 10. 
- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
- `ode_solver`: Algorithm for solving the ODE solver from DiffEqFlux. Default is Tsit5().
- `ad_method`: Automatic differntialion algorithm for the ODE solver from DiffEqFlux. Default is ForwardDiffSensitivity().
"""
function mini_batching!(UDE::UDE; pred_length = 10, verbose = true, maxiter = 500, step_size = 0.05, ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
  
  loss = init_mini_batch_loss(UDE,pred_length, ode_solver, ad_method)
  target = (x,u) -> loss(x)
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
  UDE.parameters.uhat .= UDE.data

end


function mini_batching!(UDE::MultiUDE; pred_length=5,solver=Tsit5(),sensealg = ForwardDiffSensitivity(), verbose = true, maxiter = 500, step_size = 0.05)

  loss = init_mini_batch_loss(UDE,pred_length,solver,sensealg)
  target = (x,u) -> loss(x)
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
  UDE.parameters.uhat .= UDE.data
end


function interpolate_derivs(u,t;d=12,alg = :gcv_svd)
  dudt = zeros(size(u))
  uhat = zeros(size(u))
  for i in 1:size(u)[1]
      A = RegularizationSmooth(Float64.(u[i,:]), t, d; alg = alg)
      uhat[i,:] .= A.û
      dudt_ = t -> DataInterpolations.derivative(A, t)
      dudt[i,:] .= dudt_.(t)
  end 
  return uhat, dudt
end 


function interpolate_derivs_multi(model;d=12,alg = :gcv_svd)

  N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(model.data_frame, model.time_column_name, model.series_column_name)


  ts = [times[starts[series]:(starts[series]+lengths[series]-1)] for series in eachindex(starts)]
  dats = [data[:,starts[series]:(starts[series]+lengths[series]-1)] for series in eachindex(starts)]

  dudts = []
  uhats = []
  for i in eachindex(starts)
    dudt = zeros(size(dats[i]))
    uhat = zeros(size(dats[i]))
    for j in 1:size(dats[i])[1]
        A = RegularizationSmooth(Float64.(dats[i][j,:]), ts[i], d; alg = alg)
        uhat[j,:] .= A.û
        dudt_ = t -> DataInterpolations.derivative(A, t)
        dudt[j,:] .= dudt_.(ts[i])
    end 
    push!(dudts,Float64.(dudt))
    push!(uhats,Float64.(uhat))
  end
  return uhats, dudts, ts, eachindex(starts)
end 


# requires out of place derivative calcualtion 

# @article{Bhagavan2024,
#   doi = {10.21105/joss.06917},
#   url = {https://doi.org/10.21105/joss.06917},
#   year = {2024},
#   publisher = {The Open Journal},
#   volume = {9},
#   number = {101},
#   pages = {6917},
#   author = {Sathvik Bhagavan and Bart de Koning and Shubham Maddhashiya and Christopher Rackauckas},
#   title = {DataInterpolations.jl: Fast Interpolations of 1D data},
#   journal = {Journal of Open Source Software}
# }

"""
  derivative_matching!(UDE::UDE; kwargs ...)

Trains the UDE models using a two step process. First a smooth curve is fit to the data set using funcitons from DataInterpolations.jl (Bhagavan et al. 2024).
The derivatives of the soothing function are then compared to the derivative rpeodicted by the right hand side of the UDE model using the mean squared error.
This approach signifcantly increases the speed of training becuse it does not require the UDE model to be integrated by a ODE solver. Unfortunately, this method
also relies on the accuracy of the smoothing algorithm. We suggest using this method to get close to the optial parameter sets and then applying a differnt more accurate method
to finish the training procedure. 

Note hat some of the smoothing curves will lose accuracy near the begining and end of the time series. The key word arguemnt `remove_ends`
allows the user to specify the number of data point to leave out to remove these edge effects. 

# kwargs
- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
- `d`: the number of grid points used by the data interpolation algorithm. Default is 12.
- `alg`: The algorithm from DataInterpolations.jl used to fit the smoothing curve. Default is :gcv_svd.
- `remove_ends`: Number of data points to leave off to remove edge effects while training. Defualt is 2.
"""
function derivative_matching!(UDE::UDE; verbose = true, maxiter = 500, step_size = 0.05, d = 12, alg = :gcv_svd, remove_ends = 2)

  times = UDE.times
  uhat, dudt = interpolate_derivs(UDE.data,UDE.times;d=d,alg = alg)
  uhat = Float64.(uhat)
  dudt = Float64.(dudt)
  function loss(parameters)
      L = 0
        for t in (remove_ends+1):(size(uhat)[2]-remove_ends)
            dudt_hat = UDE.process_model.rhs(uhat[:,t],parameters.process_model,times[t])
            L += sum((dudt[:,t] .- dudt_hat).^2)
        end
      return L
  end 

  target = (x,u) -> loss(x)
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
  UDE.parameters.uhat = uhat

end 



function derivative_matching!(UDE::MultiUDE; verbose = true, maxiter = 500, step_size = 0.05, d = 12, alg = :gcv_svd, remove_ends = 2)


  uhats, dudts, times, inds  = interpolate_derivs_multi(UDE;d=d,alg = alg)


  function loss(parameters)
      L = 0
      for i in inds
        for t in (remove_ends+1):(size(uhats[i])[2]- remove_ends)
            dudt_hat = UDE.process_model.rhs(uhats[i][:,t], i, parameters.process_model, times[i][t])
            L += sum((dudts[i][:,t] .- dudt_hat).^2)
        end
      end
      return L
  end 

  target = (x,u) -> loss(x)
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

  N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)
  for i in eachindex(starts) 
    UDE.parameters.uhat[:,starts[i]:(starts[i]+lengths[i]-1)] .= uhats[i]
  end 
  
end 
