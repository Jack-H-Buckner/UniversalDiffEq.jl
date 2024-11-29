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

function mini_batching!(UDE::UDE; pred_length, verbose = true, maxiter = 500, step_size = 0.05, ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
  
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


# function interpolate_derivs(u,t;d=12,alg = :gcv_svd)
#   dudt = zeros(size(u))
#   uhat = zeros(size(u))
#   for i in 1:size(u)[1]
#       A = RegularizationSmooth(Float64.(u[i,:]), t, d; alg = alg)
#       uhat[i,:] .= A.û
#       dudt_ = t -> DataInterpolations.derivative(A, t)
#       dudt[i,:] .= dudt_.(t)
#   end 
#   return uhat, dudt
# end 


# requires out of place derivative calcualtion 
# ##, RegularizationTools, DataInterpolations
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
# function derivative_matching!(model; verbose = true, maxiter = 500, step_size = 0.05, d = 12, alg = :gcv_svd)

#   times = model.times
#   uhat, dudt = interpolate_derivs(model.data,model.times;d=d,alg = alg)
#   uhat = Float64.(uhat)
#   dudt = Float64.(dudt)
#   function loss(parameters)
#       L = 0
#         for t in 1:size(uhat)[2]
#             dudt_hat = model.process_model.right_hand_side(uhat[:,t],parameters.process_model,times[t])
#             L += sum((dudt[:,t] .- dudt_hat).^2)
#         end
#       return L
#   end 

#   target = (x,u) -> loss(x)
#   adtype = Optimization.AutoEnzyme()
#   optf = Optimization.OptimizationFunction(target, adtype)
#   optprob = Optimization.OptimizationProblem(optf, model.parameters)
  
#   # print value of loss function at each time step 
#   if verbose
#       callback = function (p, l; doplot = false)
#         print(round(l,digits = 3), " ")
#         return false
#       end
#   else
#       callback = function (p, l; doplot = false)
#         return false
#       end 
#   end

#   # run optimizer
#   sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter )
  
#   # assign parameters to model 

#   model.parameters = sol.u
#   model.parameters.uhat = uhat

# end 

