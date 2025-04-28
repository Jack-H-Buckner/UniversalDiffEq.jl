
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
        tspan =  t:(t+dt-1)
        for t in tspan
            u = difference(u,covariates(t),t,parameters)
        end 
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
        tspan =  t:(t+dt-1)
        for t in tspan
            u = difference(u,t,parameters)
        end 
        return (u, 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    function right_hand_side(u,parameters,t)
        return difference(u,t,parameters) .- u
    end 

    rhs = (u,p,t) ->predict(u,t,1,p) 

    return ProcessModel(parameters,predict, forecast,0,right_hand_side,rhs,nothing)
end 

