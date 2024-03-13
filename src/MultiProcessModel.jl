
function init_Multiforecast(predict,l,extrap_rho)
    
    function forecast(u,i,t,dt,parameters,umax,umin,umeans)
        
        # eval network
        du = predict(u,i,t,dt,parameters)[1] .- u
        
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

# """
#     MultiProcessModel

# A Julia mutable struct that stores the functions and parameters for the process model. 
# ...
# # Elements
# - parameters: ComponentArray
# - predict: Function the predict one time step ahead
# - forecast: Function, a modified version of rpedict to imporve performace when extrapolating
# - covariates: Function that returns the value of the covariates at each point in time. 
# ...
# """
mutable struct MultiProcessModel
    parameters
    predict
    forecast
    covariates
end

function MultiContinuousProcessModel(derivs!,parameters, dims, l ,extrap_rho)
   
    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    derivs_t! = (du,u,p,t) -> derivs!(du,u,p.series,p,t)
    IVP = ODEProblem(derivs_t!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = solve(IVP, Tsit5(), u0 = u, p=params,tspan = tspan, saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    forecast = init_Multiforecast(predict,l,extrap_rho)
    
    return MultiProcessModel(parameters,predict, forecast,0)
end 


function MultiContinuousProcessModel(derivs!,parameters,covariates,dims,l,extrap_rho)
   
    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    derivs_t! = (du,u,p,t) -> derivs!(du,u,p.series,covariates(t,p.series),p,t)
    IVP = ODEProblem(derivs_t!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = solve(IVP, Tsit5(), u0 = u, p=params,tspan = tspan, saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    forecast = init_Multiforecast(predict,l,extrap_rho)
    
    return MultiProcessModel(parameters,predict, forecast,0)
end 




mutable struct MultiNODE_process
    dims::Int # number of state variables
    IVP
    derivs!
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
    forecast
    covariates
end 


function MultiNODE_process(dims,hidden,covariates,seed,l,extrap_rho)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims+length(covariates(0,1)),hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, )

    function derivs!(du,u,p,t)
        du .= NN(vcat(u,covariates(t,p.series)),p.NN,states)[1]
        return du
    end 

    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = solve(IVP, Tsit5(), u0 = u, p=params,tspan = tspan,saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end], 0)
    end 

    
   forecast = init_Multiforecast(predict,l,extrap_rho)
    
    return MultiNODE_process(dims,IVP,derivs!,parameters,predict,forecast,covariates)
    
end 


function MultiNODE_process(dims,hidden,seed,l,extrap_rho)
    
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

    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        sol = solve(IVP, Tsit5(), u0 = u, p=parameters,tspan = tspan, saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
   forecast = init_Multiforecast(predict,l,extrap_rho)
    
    return MultiNODE_process(dims,IVP,derivs!,parameters,predict,forecast,x->0)
    
end 

