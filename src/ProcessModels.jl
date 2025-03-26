
function init_forecast(predict,l,extrap_rho)
    
    function forecast(u,t,dt,parameters,umax,umin,umeans)
        
        # eval network
        du = predict(u,t,dt,parameters)[1] .- u
        
        # max extrapolation 
        ind_max = u .> umax
        
        omega = exp.( -0.5/l^2 * ((u[ind_max] .- umax[ind_max])./ (umax[ind_max].-umin[ind_max])).^2)
        du[ind_max] .= omega .* du[ind_max] .- extrap_rho .*(1 .- omega) .* (u[ind_max] .- umeans[ind_max] )
        
        # min extrapolation 
        ind_min = u .< umin
        omega = exp.( -0.5/l^2 * ((u[ind_min] .- umin[ind_min])./ (umax[ind_min].-umin[ind_min])).^2)
        du[ind_min] .= omega .* du[ind_min] .- (1 .- omega) .*extrap_rho .* (u[ind_min] .- umeans[ind_min] )
        
        return u .+ du
    end
    
    return forecast
    
end 

"""
    ProcessModel

A Julia mutable struct that stores the functions and parameters for the process model. 
...
# Elements
- parameters: ComponentArray
- predict: Function the predict one time step ahead
- forecast: Function, a modified version of predict to improve performance when extrapolating
- covariates: Function that returns the value of the covariates at each point in time. 
...
"""
mutable struct ProcessModel
    parameters
    predict
    forecast
    covariates
    right_hand_side #(u,x,p,t)
    rhs #(u,p,t)
    IVP
end


function check_arguments(derivs)

    if any([method.nargs for method in methods(derivs)] .== 4)

        function dudt!(du,u,p,t) 
             du .= derivs(u,p,t)
        end 
 
        return dudt!, derivs

     else
 
        function rhs(u,parameters,t)
                du = zeros(length(u))
                derivs(du,u,parameters,t)
            return du
        end
        return derivs, rhs
     end

end 

function ContinuousProcessModel(derivs,parameters, dims, l ,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())


    derivs!, right_hand_side= check_arguments(derivs)
    
    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,t,dt,parameters) 
        tspan =  (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, ode_solver, u0 = u, p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    return ProcessModel(parameters,predict, forecast,0,right_hand_side,right_hand_side,IVP)
end 


function check_arguments_X(derivs)
    
    if any([method.nargs for method in methods(derivs)] .== 5)

        function dudt!(du,u,X,p,t) 
             du .= derivs(u,X,p,t)
        end 

        return dudt!, derivs

     else

 
        function rhs(u,x,parameters,t)
                du = zeros(length(u))
                derivs(du,u,x,parameters,t)
            return du
        end
        return derivs, rhs
     end
end

function ContinuousProcessModel(derivs,parameters,covariates,dims,l,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
   
    derivs!, right_hand_side = check_arguments_X(derivs)
    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    derivs_t! = (du,u,p,t) -> derivs!(du,u,covariates(t),p,t)
    rhs_ = (u,p,t) -> right_hand_side(u,covariates(t),p,t)
    IVP = ODEProblem(derivs_t!, u0, tspan, parameters)
    
    function predict(u,t,dt,parameters) 
        tspan =  (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, ode_solver, u0 = u, p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)

    # function right_hand_side(u,x,parameters,t)
    #     du = zeros(length(u))
    #     derivs!(du,u,x,parameters,t)
    #     return du
    # end 
    
    return ProcessModel(parameters,predict, forecast,covariates, right_hand_side,rhs_, IVP)
end 



function DiscreteProcessModel(difference, parameters, covariates, dims, l, extrap_rho)
    
    function predict(u,t,dt,parameters) 
        u = difference(u,covariates(t),t,parameters)
        return (u, 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    function right_hand_side(u,x,parameters,t)
        return difference(u,x,t,parameters) .- u
    end 

    rhs = (u,p,t) ->predict(u,t,1,p) 

    return ProcessModel(parameters,predict, forecast, covariates,right_hand_side,rhs,nothing)
end 

function DiscreteProcessModel(difference, parameters, dims, l, extrap_rho)
    
    function predict(u,t,dt,parameters) 
        u = difference(u,t,parameters)
        return (u, 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    function right_hand_side(u,parameters,t)
        return difference(u,t,parameters) .- u
    end 

    rhs = (u,p,t) ->predict(u,t,1,p) 

    return ProcessModel(parameters,predict, forecast,0,right_hand_side,rhs,nothing)
end 




function NODEWithARD(dims,covariates; hidden = 20, nonlinearity = soft_plus)
   
    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    X0 = covariates(0)
    input_dims = dims + length(X0)

    NN, parameters = ARD(input_dims,dims;hidden = hidden, nonlinearity = nonlinearity)

    function derivs!(du,u,p,t)
        dudt = NN(vcat(u[1:dims],covariates(t)),p)
        du[1:dims] .= dudt 
        du[(dims+1):end] .= 0.5*(dudt ./ abs.(p.α)).^2
    end

    IVP = ODEProblem(derivs!, vcat(u0,zeros(dims)), tspan, parameters)
    
    function predict(u,t,dt,parameters) 
        tspan = (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, Tsit5(), u0 = vcat(u,zeros(dims)), p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[1:dims,end], X[(dims+1):end,end])
    end 
    
    function forecast(u,t,dt,parameters,u1,u2,u3) 
        tspan =  (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, Tsit5(), u0 = vcat(u,zeros(dims)), p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return X[1:dims,end]
    end 

    function right_hand_side(u,X,p,t)
        dudt = NN(vcat(u,X),p)
    end 
    
    return ProcessModel(parameters,predict, forecast,covariates, right_hand_side,IVP)
end 





mutable struct GPProcessModel
    parameters
    predict
    forecast
    covariates
    right_hand_side
    GP
end


function GP_process_model(dims,inducing_points,covariates)
   
    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)

    GP, parameters = initMvGaussianProcess(dims, inducing_points)

    function derivs!(du,u,p,t)
        GP(vcat(u,covariates(t)),p)
    end

    IVP = ODEProblem(derivs!, vcat(u0,zeros(dims)), tspan, parameters)
    
    function predict(u,t,dt,parameters) 
        tspan = (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, Tsit5(), u0 = u, p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    function forecast(u,t,dt,parameters,u1,u2,u3) 
        tspan =  (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, Tsit5(), u0 = u, p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return X[:,end]
    end 

    function right_hand_side(u,X,p,t)
        GP(vcat(u,X),p)
    end 
    
    return GPProcessModel(parameters,predict, forecast,covariates, right_hand_side,GP)
end 





mutable struct NeuralNetwork
    dims::Int # number of state variables
    NN # lux neural network object 
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
    forecast
    right_hand_side
end 


function NeuralNetwork(dims,hidden,seed,extrap_rho,l)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, known_dynamics = NamedTuple())
    function predict(u,t,dt,parameters) 
        tspan =  1:dt
        for t in tspan
            u = u .+  NN(u,parameters.NN,NN_states)[1]
        end 
        return (u, 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)

    function right_hand_side(u,parameters,t)
        return NN(u,parameters.NN,NN_states)[1]
    end 
    
    return NeuralNetwork(dims,NN,parameters,predict,forecast,right_hand_side)
    
end 


mutable struct DiscreteModelErrors
    dims::Int # number of state variables
    error_function # lux neural network object 
    known_dynamics
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network
    forecast
end 

function DiscreteModelErrors(dims,known_dynamics,init_known_dynamics_parameters,hidden,seed,extrap_rho,l)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (known_dynamics = init_known_dynamics_parameters, 
                    NN = NN_parameters)
    # prediction 
    error_function = (u,parameters) -> NN(u,parameters.NN,NN_states)[1]
    known_dynamics_function = (u,t,parameters) -> known_dynamics(u,t,parameters.known_dynamics)
    
    
    function predict(u0,t,dt,parameters) 
        tspan =  1:dt
        u = u0
        for t in tspan
            u = known_dynamics_function(u,t,parameters) .+ error_function(u,parameters)
        end 
        return (u, 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    return DiscreteModelErrors(dims,error_function,known_dynamics_function,parameters,predict,forecast)
    
end 



mutable struct NODE_process
    dims::Int # number of state variables
    IVP
    derivs!
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
    forecast
    covariates    
    right_hand_side
    rhs
end 


function NODE_process(dims,hidden,covariates,seed,l,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims+length(covariates(0)),hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, )

    function derivs!(du,u,parameters,t)
        du .= NN(vcat(u,covariates(t)),parameters.NN,states)[1]
        return du
    end 

    function rhs(u,p,t)
        NN(vcat(u,covariates(t)),p.NN,states)[1]
    end 

    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,t,dt,parameters) 
        tspan =  (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, ode_solver , u0 = u, p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method )
        X = Array(sol)
        return (X[:,end], 0)
    end 

    function derivs_forecast!(du,u,parameters,t, covariates)
        du .= NN(vcat(u,covariates(t)),parameters.NN,states)[1]
        return du
    end 
    
    
   forecast = init_forecast(predict,l,extrap_rho)

   function right_hand_side(u,x,parameters,t)
        return NN(vcat(u,x),parameters.NN,states)[1]
    end 
    
    return NODE_process(dims,IVP,derivs!,parameters,predict,forecast,covariates,right_hand_side,rhs)
    
end 


function NODE_process(dims,hidden,seed,l,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, )
    function derivs!(du,u,parameters,t)
        du .= NN(u,parameters.NN,states)[1]
        return du
    end 

    function rhs(u,p,t)
        NN(u,p.NN,states)[1]
    end 

    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,t,dt,parameters) 
        tspan =  (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, ode_solver, u0 = u, p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method)
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
   forecast = init_forecast(predict,l,extrap_rho)

   function right_hand_side(u,parameters,t)
        return NN(u,parameters.NN,states)[1]
    end

    return NODE_process(dims,IVP,derivs!,parameters,predict,forecast,[],right_hand_side,rhs)
    
end 


