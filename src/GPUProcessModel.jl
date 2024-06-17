function ContinuousProcessModel_GPU(derivs!,parameters, dims, l ,extrap_rho, gpu_device)
   
    u0 = zeros(dims) |> gpu_device; tspan = (0.0f0,1000.0f0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters |> gpu_device)
    
    function predict(u,t,dt,parameters) 
        tspan =  (t,t+dt)
        sol = solve(IVP, Tsit5(), u0 = u, p=parameters,tspan = tspan, 
                    saveat = (t,t+dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol) |> gpu_device
        return (X[:,end], 0)
    end 
    
    forecast = init_forecast(predict,l,extrap_rho)
    
    function right_hand_side(u,parameters,t)
        du = zeros(length(u))
        derivs!(du,u,parameters,t)
        return du
    end 

    return ProcessModel(parameters,predict, forecast,0,right_hand_side)
end 
