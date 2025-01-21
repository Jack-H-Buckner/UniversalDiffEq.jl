
# Funtions that build loss functions for differnt training routines?

function errors_to_matrix(error, dims)

    if any(typeof(error) .== [Int64, Float64, Int32, Float32])
        return error * Matrix(I,dims,dims)
    elseif length(error) == dims 
        return  Matrix(I,dims,dims) .* error
    elseif all(size(error) .== dims)
        return error
    else 
        print("Error: error term needs to be a float, vector or square matrix")
    end 
end 

function conditional_likelihood(UDE::UDE, regularization_weight, observation_error, process_error)

    # loss function 
    dims = size(UDE.data)[1]
    Σobs = errors_to_matrix(observation_error, dims)
    Σproc = errors_to_matrix(process_error, dims)
    Σobsinv = inv(Σobs)
    Σprocinv = inv(Σproc)
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(UDE.data)[2])
            yt = UDE.data[:,t]
            yhat = UDE.observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += (yt .- yhat)' * Σobsinv * (yt .- yhat)
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(UDE.data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt =UDE.times[t]-UDE.times[t-1]
            u1hat, epsilon =UDE.process_model.predict(u0,UDE.times[t-1],dt,parameters.process_model) 
            L_proc += (u1 .- u1hat)' * Σprocinv * (u1 .- u1hat)
        end
        
        # regularization
        L_reg = regularization_weight *UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return L_obs + L_proc + L_reg
    end

    return loss_function,UDE.parameters, []
end 



function marginal_likelihood(UDE::UDE,regularization_weight,Pν,Pη,α,β,κ)


    # observaiton model is identity function
    L = size(UDE.data)[1]
    H = Matrix(I,L,L)

    y = UDE.data
    times = UDE.times
    f = (u,t,dt,p) -> UDE.process_model.predict(u,t,dt,p)[1]

    function loss(parameters)
        Pν = parameters.Pν * parameters.Pν'
        ll = -1*ukf_likeihood(y,times,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
        ll += regularization_weight *UDE.process_regularization.loss(parameters.UDE.process_model,parameters.UDE.process_regularization)
        return ll
    end
    
    Pνchol = Matrix(cholesky(Pν).L)

    params = ComponentArray((UDE = UDE.parameters, Pν = Pνchol))

    return loss, params, (H,Pν,Pη,L,α,β,κ)
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
  
  

function derivative_matching_loss(UDE::UDE, regularization_weight; d = 12, alg = :gcv_svd, remove_ends = 2)

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
        L += regularization_weight *UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)
        return L
    end 

    return loss, UDE.parameters ,uhat
end 


function shooting_loss(UDE::UDE)
    
    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict(u,t0,tsteps,parameters)
        tspan =  (t0,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=parameters,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg  )
        X = Array(sol)
        return X
    end 

    function loss_function(parameters)

        # dynamics loss 
        L_proc = 0
        u0 = parameters.uhat[:,1]
        t0 =UDE.times[1]
        times =UDE.times[2:end]
        u1 =UDE.data[:,2:end]
        u1hat = predict(u0,t0,times,parameters.process_model)
        dt = 1
        for i in 2:length(times) 
            L_proc +=UDE.process_loss.loss(u1[:,i],u1hat[:,i],dt,parameters.process_loss)
        end 
        
        # regularization
        L_reg =UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg +=UDE.observation_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_proc + L_reg
    end

    return loss_function,UDE.parameters, []
end 


function shooting_states(UDE::UDE)

    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict(u,t0,tsteps,parameters)
        tspan =  (t0,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=parameters,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg  )
        X = Array(sol)
        return X
    end 

    uhat = zeros(size(UDE.data))
    u0 = UDE.parameters.uhat[:,1]
    t0 =UDE.times[1]
    times =UDE.times[2:end]
    uhat[:,2:end] .= predict(u0,t0,times,UDE.parameters.process_model)
    uhat[:,1] .= u0
    return uhat
end 


function multiple_shooting_loss(UDE::UDE,pred_length)

    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict_mini(u,t0,tsteps,parameters)
        tspan =  (t0,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=parameters,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg  )
        X = Array(sol)
        return X
    end 

    function loss_function(parameters)

        # dynamics loss 
        L_proc = 0
        inds = 1:pred_length
        tspan = 1:pred_length
        for t in 1:pred_length:(size(UDE.data)[2])
            
            if (t - size(UDE.data)[2]) >=pred_length
                inds = (t+1):(t+pred_length)
                tspan =UDE.times[inds]
            else
                inds = (t+1):size(UDE.data)[2]
                tspan =UDE.times[inds]
            end 

            u0 = parameters.uhat[:,t]
            u1 =UDE.data[:,inds]
            u1hat = predict_mini(u0,UDE.times[t],tspan,parameters.process_model)
            #L_proc += sum( (u1 .- u1hat).^2 )/length(model.data)
            dt = 1
            for i in 1:length(inds) 
                L_proc += sum((u1[:,i].-u1hat[:,i]).^2)/length(UDE.data)
            end 
        end
        
        # regularization
        L_reg = UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg +=UDE.observation_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_proc + L_reg
    end

    return loss_function,UDE.parameters, []
end 



function multiple_shooting_states(UDE::UDE,pred_length)

    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict(u,t0,tsteps,parameters)
        tspan =  (t0,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=parameters,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg  )
        X = Array(sol)
        return X
    end 

    # dynamics loss 
    inds = 1:pred_length
    tspan = 1:pred_length
    uhat = zeros(size(UDE.data))
    for t in 1:pred_length:(size(UDE.data)[2])
        if (t - size(UDE.data)[2]) >=pred_length
            inds = (t+1):(t+pred_length)
            tspan =UDE.times[inds]
        else
            inds = (t+1):size(UDE.data)[2]
            tspan =UDE.times[inds]
        end 
        u0 = UDE.parameters.uhat[:,t]
        uhat[:,t] .= u0
        uhat[:,inds] .= predict(u0,UDE.times[t],tspan,UDE.parameters.process_model)
    end

    return uhat
end 