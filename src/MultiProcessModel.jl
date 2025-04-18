
function init_Multiforecast(predict,l,extrap_rho)
    
    function forecast(u,i,t,dt,parameters,umax,umin,umeans)
        
        # evaluate network
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

 """
     MultiProcessModel

 A Julia mutable struct that stores the functions and parameters for the process model. 
 ...
 # Elements
 - `parameters`: ComponentArray
 - `predict`: Function that predicts one time step ahead
 - `forecast`: Function that is a modified version of predict to improve performace when extrapolating
 - `covariates`: Function that returns the values of the covariates at each point in time
 - `right_hand_side`: Function that returns the right-hand side of a differential equation (i.e., the relationships between state variables and parameters)
 ...
 """
mutable struct MultiProcessModel
    parameters
    predict
    forecast
    covariates
    right_hand_side #(i,i,X,p,t)
    rhs # (u,i,p,t)
    IVP
end


function check_arguments_multi(derivs)
    if any([method.nargs for method in methods(derivs)] .== 5)


        function dudt!(du,u,i,p,t) 
             du .= derivs(u,i,p,t)
        end 
 
        return dudt!, derivs

     else
 

        function right_hand_side(u,i,parameters,t)
                du = zeros(length(u))
                derivs(du,u,i,parameters,t)
            return du
        end
        return derivs, right_hand_side
     end
end 


function MultiContinuousProcessModel(derivs,parameters, dims, l ,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
   
    derivs!, right_hand_side = check_arguments_multi(derivs)
    u0 = zeros(dims); tspan = (0.0,1000.0) # assessing value for the initial conditions and time span (these are arbitrary)
    derivs_t! = (du,u,p,t) -> derivs!(du,u,p.series,p,t)
    IVP = ODEProblem(derivs_t!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = OrdinaryDiffEq.solve(IVP,ode_solver, u0 = u, p=params,tspan = tspan, saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method  )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    forecast = init_Multiforecast(predict,l,extrap_rho)

    # function right_hand_side(u,i,t,parameters)
    #     du = zeros(length(u))
    #     derivs!(du,u,i,parameters,t)
    #     return du
    # end 
    
    return MultiProcessModel(parameters,predict, forecast, 0, right_hand_side,right_hand_side, IVP)
end 


function check_arguments_multi_X(derivs)
    if any([method.nargs for method in methods(derivs)] .== 6)
     
        function dudt!(du,u,i,X,p,t) 
             du .= derivs(u,i,X,p,t)
        end 
 
        return dudt!, derivs

     else
 
        function right_hand_side(u,i,X,parameters,t)
                du = zeros(length(u))
                derivs(du,u,i,X,parameters,t)
            return du
        end
        return derivs, right_hand_side
     end
end 

function MultiContinuousProcessModel(derivs,parameters,covariates,dims,l,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
   
    derivs!, rhs = check_arguments_multi_X(derivs)

    u0 = zeros(dims); tspan = (0.0,1000.0) # assessing value for the initial conditions and time span (these are arbitrary)
    derivs_t! = (du,u,p,t) -> derivs!(du,u,p.series,covariates(t,p.series),p,t)
    rhs_(u,i,p,t) = rhs(u,i,covariates(t,i),p,t)

    IVP = ODEProblem(derivs_t!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = OrdinaryDiffEq.solve(IVP, ode_solver, u0 = u, p=params,tspan = tspan, saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    forecast = init_Multiforecast(predict,l,extrap_rho)
    
    # function right_hand_side(u,i,X,t,parameters)
    #     du = zeros(length(u))
    #     derivs!(du,u,i,X,parameters,t)
    #     return du
    # end 
    

    return MultiProcessModel(parameters,predict, forecast,covariates,rhs,rhs_,IVP)
end 



function MultiDiscreteProcessModel(difference,parameters,covariates,dims,l,extrap_rho)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  t:(t+dt)
        for t in tspan
            u = difference(u,i,covariates(t,i),parameters,t)
        end 
        return (u, 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    function right_hand_side(u,i,X,parameters,t)
        return difference(u,i,X,parameters,t) .- u
    end 

    return MultiProcessModel(parameters,predict,forecast,covariates,right_hand_side,nothing,nothing)
end 

function MultiDiscreteProcessModel(difference,parameters,dims,l,extrap_rho)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  t:(t+dt)
        for t in tspan
            u = difference(u,i,parameters,t)
        end 
        return (u, 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    function right_hand_side(u,i,parameters,t)
        return difference(u,i,parameters,t) .- u
    end 

    return MultiProcessModel(parameters,predict,forecast,0,right_hand_side,nothing,nothing)
end 

mutable struct MultiNODE_process
    dims::Int # number of state variables
    IVP
    derivs!
    parameters #::ComponentArray of neural network parameters
    predict::Function # neural network 
    forecast
    covariates
    right_hand_side
end 


function MultiNODE_process(dims,hidden,covariates,seed,l,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    
    # initial neural network
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

    u0 = zeros(dims); tspan = (0.0,1000.0) # assessing value for the initial conditions and time span (these are arbitrary)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = OrdinaryDiffEq.solve(IVP, ode_solver, u0 = u, p=params,tspan = tspan,saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method)
        X = Array(sol)
        return (X[:,end], 0)
    end 

    
   forecast = init_Multiforecast(predict,l,extrap_rho)

   function right_hand_side(u,i,X,t,parameters)
        du = NN(vcat(u,X),parameters.NN,states)[1]
        return du
    end 

    
    return MultiNODE_process(dims,IVP,derivs!,parameters,predict,forecast,covariates,right_hand_side)
    
end 


function MultiNODE_process(dims,hidden,seed,l,extrap_rho; ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    
    # initial neural network
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

    u0 = zeros(dims); tspan = (0.0,1000.0) # assessing value for the initial conditions and time span (these are arbitrary)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,i,t,dt,parameters) 
        tspan =  (t,t+dt) 
        sol = OrdinaryDiffEq.solve(IVP, ode_solver, u0 = u, p=parameters,tspan = tspan, saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ad_method )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
   forecast = init_Multiforecast(predict,l,extrap_rho)


   function right_hand_side(u,i,t,parameters)
        du = NN(u,parameters.NN,states)[1]
        return du
    end 
    
    return MultiNODE_process(dims,IVP,derivs!,parameters,predict,forecast,0,right_hand_side)
    
end 

