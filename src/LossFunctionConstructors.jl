
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




function conditional_likelihood(UDE::UDE, t_skip, regularization_weight, observation_error, process_error)

    # process training dat
    # set up error matrices
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
            if t != t_skip
                u0 = parameters.uhat[:,t-1]
                u1 = parameters.uhat[:,t]
                dt = UDE.times[t]-UDE.times[t-1]
                u1hat, epsilon =UDE.process_model.predict(u0,UDE.times[t-1],dt,parameters.process_model) 
                L_proc += (u1 .- u1hat)' * Σprocinv * (u1 .- u1hat)
            end 
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
        ll = -1*ukf_likelihood(y,times,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
        ll += regularization_weight *UDE.process_regularization.loss(parameters.UDE.process_model,parameters.UDE.process_regularization)
        return ll
    end
    
    Pνchol = Matrix(cholesky(Pν).L)

    params = ComponentArray((UDE = UDE.parameters, Pν = Pνchol))

    return loss, params, (H,Pν,Pη,L,α,β,κ)
end




function marginal_likelihood(UDE::UDE,t_skip,regularization_weight,Pν,Pη,α,β,κ)

    # observaiton model is identity function
    L = size(UDE.data)[1]
    H = Matrix(I,L,L)

    y1 = UDE.data[:,1:t_skip]
    y2 = UDE.data[:,(t_skip+1):end]
    times1 = UDE.times[:,1:t_skip]
    times2 = UDE.times[:,(t_skip+1):end]
    f = (u,t,dt,p) -> UDE.process_model.predict(u,t,dt,p)[1]

    function loss(parameters)
        Pν = parameters.Pν * parameters.Pν'
        ll = -1*ukf_likelihood(y1,times1,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
        ll = -1*ukf_likelihood(y2,times2,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
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


function multiple_shooting_loss(UDE::UDE,regularization_weight,pred_length)

    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict_mini(u,tsteps,parameters)
        tspan =  (tsteps[1]-10^-6,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
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
        
            if ( size(UDE.data)[2]-t) >= pred_length
                inds = t:(t+pred_length)
                tspan = UDE.times[inds]
            else
                inds = t:size(UDE.data)[2]
                tspan =UDE.times[inds]
            end 
            u0 = parameters.uhat[:,t]
            u1 = UDE.data[:,inds]
            u1hat = predict_mini(u0,tspan,parameters.process_model)
            for i in 1:length(inds) 
                L_proc += sum((u1[:,i].-u1hat[:,i]).^2)/length(UDE.data)
            end 
        end
        
        # regularization
        L_reg = regularization_weight * UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return L_proc + L_reg
    end

    return loss_function,UDE.parameters, []
end 




function multiple_shooting_states(UDE::UDE,  pred_length)

    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict(u,tsteps,parameters)
        tspan =  (tsteps[1]-10^-6,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
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
        if (size(UDE.data)[2]-t) >=pred_length
            inds = t:(t+pred_length)
            tspan =UDE.times[inds]
        else
            inds = t:size(UDE.data)[2]
            tspan =UDE.times[inds]
        end 
        u0 = UDE.parameters.uhat[:,t]
        uhat[:,inds] .= predict(u0,tspan,UDE.parameters.process_model)
    end

    return uhat
end 


## multiple time seires model constructors


function init_single_conditional_likelihood(UDE::MultiUDE,  observation_error, process_error)
    
    dims = size(UDE.data)[1]
    Σobs = errors_to_matrix(observation_error, dims)
    Σproc = errors_to_matrix(process_error, dims)
    Σobsinv = inv(Σobs)
    Σprocinv = inv(Σproc)

    function loss(parameters,series,starts,lengths)
        
        # observation loss
        L_obs = 0.0 
        uhat = parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
        time = UDE.times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = UDE.data[:,starts[series]:(starts[series]+lengths[series]-1)]
        for t in 1:(size(dat)[2])
            yt = dat[:,t]
            yhat = UDE.observation_model.link(uhat[:,t],parameters.observation_model)
            L_obs += (yt .- yhat)' * Σobsinv * (yt .- yhat)
        end
    
        # process loss 
        L_proc = 0
        for t in 2:(size(dat)[2])
            u0 = uhat[:,t-1]
            u1 = uhat[:,t]
            dt = time[t]-time[t-1]
            u1hat, epsilon = UDE.process_model.predict(u0,series,time[t-1],dt,parameters.process_model) 
            L_proc += (u1 .- u1hat)' * Σprocinv * (u1 .- u1hat)
        end
        
        
        return L_obs + L_proc
    end
    return loss
end




function conditional_likelihood(UDE::MultiUDE, regularization_weight, observation_error, process_error)

    single_loss = init_single_conditional_likelihood(UDE, observation_error, process_error)
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)

    function loss(parameters)
        L = 0
        for i in eachindex(starts)
            L+= single_loss(parameters,i,starts,lengths)
        end
        # regularization
        L += regularization_weight *UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return L
    end   

    return loss,UDE.parameters, []
end



function init_single_marginal_likelihood(UDE::MultiUDE,H,Pη,L,α,β,κ)
    
    Imat = Matrix(I,L,L)
    function loss(parameters,series,starts,lengths)
        
        times = UDE.times[starts[series]:(starts[series]+lengths[series]-1)]
        y = UDE.data[:,starts[series]:(starts[series]+lengths[series]-1)]
        f = (u,t,dt,p) -> UDE.process_model.predict(u,series,t,dt,p)[1]

        Pν = Imat .* parameters.Pν.^2

        nll = -1*ukf_likelihood(y,times,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
        return nll + L_reg
    end

  
    return loss
end

function single_marginal_likelihood(UDE::MultiUDE,regularization_weight,Pη,α,β,κ)


    # observaiton model is identity function
    L = size(UDE.data)[1]
    H = Matrix(I,L,L)

    y = UDE.data
    times = UDE.times
    f = (u,i,t,dt,p) -> UDE.process_model.predict(u,i,t,dt,p)[1]

    function loss(parameters,series,starts,lengths)
        Pν = parameters.Pν * parameters.Pν'

        times = UDE.times[starts[series]:(starts[series]+lengths[series]-1)]
        y = UDE.data[:,starts[series]:(starts[series]+lengths[series]-1)]

        ll = -1*ukf_likelihood(y,times,(u,t,dt,p) -> f(u,series,t,dt,p),parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)
        ll += regularization_weight * UDE.process_regularization.loss(parameters.UDE.process_model,parameters.UDE.process_regularization)
        return ll
    end
    

    return loss
end

function marginal_likelihood(UDE::MultiUDE,regularization_weight,Pν,Pη,α,β,κ)
    
    single_loss = single_marginal_likelihood(UDE,regularization_weight,Pη,α,β,κ)
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)

    function loss(parameters)
        ll = 0
        for i in eachindex(starts)
            ll+= single_loss(parameters,i,starts,lengths)
        end
        ll += regularization_weight *UDE.process_regularization.loss(parameters.UDE.process_model,parameters.UDE.process_regularization)

        return ll
    end   

    L = size(UDE.data)[1]
    H = Matrix(I,L,L)

    Pνchol = Matrix(cholesky(Pν).L)

    params = ComponentArray((UDE = UDE.parameters, Pν = Pνchol))

    return loss, params, (H,Pη,L,α,β,κ)
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

  
function derivative_matching_loss(UDE::MultiUDE, regularization_weight; d = 12, alg = :gcv_svd, remove_ends = 2)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)
  
    ts = [times[starts[series]:(starts[series]+lengths[series]-1)] for series in eachindex(starts)]
    dats = [data[:,starts[series]:(starts[series]+lengths[series]-1)] for series in eachindex(starts)]
  
    dudts = []
    uhats = []
    uhatpreds = zeros(size(data))
    for i in eachindex(starts)
        dudt = zeros(size(dats[i]))
        uhat = zeros(size(dats[i]))
        for j in 1:size(dats[i])[1]
            A = RegularizationSmooth(Float64.(dats[i][j,:]), ts[i], d; alg = alg)
            uhat[j,:] .= A.û
            uhatpreds[j,starts[i]:(starts[i]+lengths[i]-1)] .= A.û
            dudt_ = t -> DataInterpolations.derivative(A, t)
            dudt[j,:] .= dudt_.(ts[i])
        end 
        push!(dudts,Float64.(dudt))
        push!(uhats,Float64.(uhat))
    end


    function loss(parameters)
        L = 0
        for i in eachindex(uhats)
            for t in (remove_ends+1):(size(uhats[i])[2]- remove_ends)
                dudt_hat = UDE.process_model.rhs(uhats[i][:,t], i, parameters.process_model, ts[i][t])
                L += sum((dudts[i][:,t] .- dudt_hat).^2)
            end
        end
        L += regularization_weight *UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return L
    end 

    return loss, UDE.parameters, uhatpreds

end 




function single_shooting_loss(UDE::MultiUDE)
    
    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad


    # loss function 
    function predict(u,i,t0,tsteps,parameters)
        tspan =  (t0,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=params,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg )
        X = Array(sol)
        return X
    end 

    function loss_function(parameters,series,starts,lengths)

        # dynamics loss 
        L = 0
        u0 = UDE.data[:,starts[series]]
        t0 = UDE.times[starts[series]]

        times = UDE.times[(starts[series]+1):(starts[series]+lengths[series]-1)]
        u1 = UDE.data[:,(starts[series]+1):(starts[series]+lengths[series]-1)]

        u1hat = predict(u0,series,t0,times,parameters.process_model)
        for i in 2:length(times) 
            L += sum( (u1[:,i] .- u1hat[:,i]).^2)
        end 
        
        return L
    end

    return loss_function
end 



function shooting_loss(UDE::MultiUDE,regularization_weight)
    
    single_loss = single_shooting_loss(UDE)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)

    function loss(parameters)
        ll = 0
        for i in eachindex(starts)
            ll+= single_loss(parameters,i, starts, lengths)
        end
        ll += regularization_weight *UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return ll
    end   

    return loss, UDE.parameters, []
end 


function shooting_states(UDE::MultiUDE)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)

    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict(u,i,t0,tsteps,parameters)
        tspan =  (t0,tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=params,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg )
        X = Array(sol)
        return X
    end 

    uhat = zeros(size(UDE.data))

    for series in eachindex(starts)
        t0 = times = UDE.times[starts[series]]
        times = UDE.times[starts[series]:(starts[series]+lengths[series]-1)]
        u0 = data[:,starts[series]]
        uhat[:,starts[series]:(starts[series]+lengths[series]-1)] .= predict(u0,series,t0,times,UDE.parameters.process_model)
    end
   
    return uhat
end 







function single_multiple_shooting_loss(UDE::MultiUDE,pred_length)

    solver = UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict_mini(u,i,tsteps,parameters)
        tspan =  (tsteps[1],tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=params,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg )
        X = Array(sol)
        return X
    end 

    function loss_function(parameters,series,starts,lengths)

        # dynamics loss 
        times = UDE.times[(starts[series]):(starts[series]+lengths[series]-1)]
        dat = UDE.data[:,(starts[series]):(starts[series]+lengths[series]-1)]
        uhats = parameters.uhat[:,(starts[series]):(starts[series]+lengths[series]-1)]

        # dynamics loss 
        L_proc = 0
        inds = 1:pred_length
        tspan = 1:pred_length
        for t in 1:pred_length:(size(dat)[2])
            
            if (size(dat)[2] - t) >=pred_length
                inds = t:(t+pred_length)
                tspan = times[inds]
            else
                inds = t:size(dat)[2]
                tspan = times[inds]
            end 

            u0 = uhats[:,t]
            u1 = dat[:,inds]
            u1hat = predict_mini(u0,series,tspan,parameters.process_model)
            for i in eachindex(inds) 
                L_proc += sum((u1[:,i].-u1hat[:,i]).^2)/length(dat)
            end 
        end
        
        return L_proc
    end

    return loss_function
end 


# L_reg = UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)

function multiple_shooting_loss(UDE::MultiUDE, regularization_weight, pred_length)
    
    single_loss = single_multiple_shooting_loss(UDE,pred_length)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)
    function loss(parameters)
        ll = 0
        for i in eachindex(starts)
            ll+= single_loss(parameters,i, starts, lengths)
        end
        ll += regularization_weight *UDE.process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return ll
    end   

    return loss, UDE.parameters, []
end 


function multiple_shooting_states(UDE::MultiUDE,  pred_length)

    solver =UDE.solvers.ode
    sensealg =UDE.solvers.ad

    # loss function 
    function predict(u,i,tsteps,parameters)
        tspan =  (tsteps[1],tsteps[end])  # Tsit5()#ForwardDiffSensitivity()
        params = vcat(parameters,ComponentArray((series =i, )))
        sol = OrdinaryDiffEq.solve(UDE.process_model.IVP, solver, u0 = u, p=params,tspan = tspan, 
                    saveat = tsteps,abstol=1e-6, reltol=1e-6, sensealg = sensealg )
        X = Array(sol)
        return X
    end 

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(UDE.data_frame, UDE.time_column_name, UDE.series_column_name)

    inds = 1:pred_length
    tspan = 1:pred_length
    uhats = zeros(size(UDE.data))
    
    for series in eachindex(starts)

        times = UDE.times[starts[series]:(starts[series]+lengths[series]-1)]
        uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]

        for t in 1:pred_length:length(times)
            inds = 0
            tspan = 0
            if (length(times)-t) >=pred_length
                inds = t:(t+pred_length)
                tspan = times[inds]
            else
                inds = t:length(times)
                tspan = times[inds]
            end 

            u0 = uhat[:,t]
            u = predict(u0,series,tspan,UDE.parameters.process_model)
            uhats[:, starts[series] .+ inds .-1] = u

        end
    end

    return uhats
end 


# define neural gradient matching loss
function interp_loss_2(p, states, t, u, NN)
    uhat =  Lux.apply(NN,t,p,states)[1]
    L = sum((u .- uhat).^2)
    return L
end


function initialize_nn_interpolations(u,t,init_weights_layer_1)

    # initialise neural network and parameters
    rng = Random.default_rng() 
    logit(x) = exp(x)/(1+exp(x))
    NN = Lux.Chain(Lux.Dense(1,length(t),tanh),Lux.Dense(length(t),size(u)[1]))

    pars, states = Lux.setup(rng,NN)
    pars.layer_1.weight .= init_weights_layer_1 # initlaize weights so one neuron in lined up with each time step 
    pars.layer_1.bias .= -1*t[1,:].*init_weights_layer_1
   
    # Target function 
    function target(x,_)
        interp_loss_2(x, states, t, u, NN)
    end

    # Optimization rutine 
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf,ComponentArray(pars))
    callback = function (p, l; doplot = false)
      return false
    end

    # Run optimizer twice with two differnt stpe sizes 
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.05), callback = callback, maxiters = 250)
    optprob = Optimization.OptimizationProblem(optf,ComponentArray(sol.u))
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(0.025), callback = callback, maxiters = 500)

    # return parameters for interpolation
    interp_params = sol.u
    return interp_params, NN, states
end 

function dNNdt(t,pars)
    derivs(x) = 1 - tanh(x)^2
    m1 = pars.layer_1.weight .* pars.layer_2.weight' 
    M = derivs.(pars.layer_1.bias.+pars.layer_1.weight*t )'
    du = M*m1[:,1]
    for i in 2:size(m1)[2]
        du = hcat(du, M*m1[:,i])
    end
    return du
end

function neural_gradient_matching_loss(UDE::UDE, σ, τ, regularization_weight, init_weights_layer_1)

    t = reshape(UDE.times,1,length(UDE.times))
    interp_params, interpNN, states = initialize_nn_interpolations(UDE.data,t,init_weights_layer_1)
   
    parameters = ComponentArray((UDE = UDE.parameters, interp = interp_params))
    u = UDE.data

    function loss(parameters, σ, τ,regularization_weight) #t,u,params,states,interpNN,gradModel,σ,τ,δ

        # get dimensions 
        T = size(t)[2]
        
        # calculate loss for first dimesion
        û = Lux.apply(interpNN,t, parameters.interp, states)[1]
        L = 1/σ^2*sum((u .- û).^2) 

        # calcualte gradients 
        Δû= dNNdt(t, parameters.interp)'

        for t_ in 1:T
            Δ̂u = UDE.process_model.rhs(û[:,t_], parameters.UDE.process_model, t[t_])
            L += 1/τ^2*sum((Δû[:,t_].- Δ̂u).^2 )
        end
        L += L2(0.0,regularization_weight,parameters.UDE.process_model)

        return L 
    end

    loss_ = parameters -> loss(parameters,σ,τ,regularization_weight)

    return loss_, parameters, (interpNN,states)

end 



############################################
### Method for training UDEs by matching ###
### the gradients of the UDE to a linear ###
### interpolation of the time series.    ###
### This approximates inference for the  ###
### stochastic ODEs.                     ###
############################################    
struct linear_interp
    t
    ts 
    inds_1
    inds_2
    weights_1
    weights_2
end 

function linear_interp(t,ts) 
    println(argmin(abs.(ts .- maximum(ts[ts.<t[end]]))))
    inds_1 = broadcast(u -> argmin(abs.(ts .- maximum(ts[ts.<u]))), t)
    inds_2 = broadcast(u -> argmin(abs.(ts .- minimum(ts[ts.>u]))), t)
    weights_1 = (t.-ts[inds_1])./(ts[inds_2]-ts[inds_1])
    weights_2 = 1 .- weights_1
    return linear_interp(t,ts,inds_1,inds_2,weights_1,weights_2)
end 

function interp_derivs(interp,α)
    dt = (α[2:(end),:] .- α[1:(end-1),:])./(interp.ts[2:(end)].-interp.ts[1:(end-1)])
    return interp.ts[1:(end-1)], dt
end 

function interp_states_spline(uapprox,α)
    w1, w2 = uapprox.weights_1, uapprox.weights_2
    α1 = α[uapprox.inds_1,:]
    α2 = α[uapprox.inds_2,:]
    û = (α1 .* w1 .+ α2 .* w2)
    return û[1,:,:]'
end 


function spline_gradient_matching_loss(UDE::UDE, σ, τ, regularization_weight, T)

    # Get time points for interpolations
    t = reshape(UDE.times,1,length(UDE.times))
    dt = (t[1,end]-t[1,1]) / T
    ts = collect((t[1,1]-2*dt):dt:(t[1,end]+2*dt))

    # Get time step of interpolation reletive to the data set
    average_interval = (t[1,end]-t[1,1])/length(t)
    Δt = dt/average_interval

    # initialize interpolcation 
    uapprox = linear_interp(t,ts) 
    u = UDE.data
    α = zeros(length(ts),size(u)[1])
    parameters = ComponentArray((α=α,UDE = UDE.parameters))
    
    function loss(u,parameters,uapprox,σ,τ)

        # get dimensions 
        T = length(uapprox.t)

        # calculate loss for first dimesion
        û = interp_states_spline(uapprox,parameters.α)
        Lobs = sum( ((u .- û)./σ).^2 ) 

        # calcualte gradients 
        û = parameters.α[1:(end-1),:]
        ts, Δû = interp_derivs(uapprox,parameters.α)
        T̂ = length(ts)
        Ldyn = 0
        for t_ in eachindex(ts)[1:(end-1)]
            Δ̂u = UDE.process_model.rhs(û[t_,:], parameters.UDE.process_model, ts[t_])
            Ldyn += sum( ((Δû[t_,:] .- Δ̂u)./(sqrt(Δt)*τ)).^2 )
        end
        Lreg = L2(0.0,regularization_weight,parameters.UDE.process_model)
        return Lobs + Lreg + T/T̂ * Ldyn
    end

    loss_ = parameters -> loss(u,parameters,uapprox,σ,τ)

    return loss_, parameters, uapprox

end 