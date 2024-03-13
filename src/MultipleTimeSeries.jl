include("MultiProcessModel.jl")

mutable struct MultiUDE
    times
    data
    X
    data_frame
    parameters
    loss_function
    process_model
    process_loss 
    observation_model
    observation_loss 
    process_regularization
    observation_regularization
    constructor
end

function init_single_loss(process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    function loss(parameters,times,data,series,starts,lengths)
        
        # observation loss
        L_obs = 0.0 
        uhat = parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        for t in 1:(size(dat)[2])
            yt = dat[:,t]
            yhat = observation_model.link(uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(dat)[2])
            u0 = uhat[:,t-1]
            u1 = uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,series,times[t-1],dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model.NN,parameters.process_regularization)

        return L_obs + L_proc + L_reg
    end
    return loss
end

function init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    single_loss = init_single_loss(process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    function loss(parameters)
        L = 0
        for i in eachindex(starts)
            L+= single_loss(parameters,times,data,i,starts,lengths)
        end
        return L
    end   
    
    return loss
        
end
    


"""
    MultiNODE(data;kwargs...)

builds a NODE model to fit to the data. `data` is a DatFrame object with time arguments placed in a colum labed `t` and a second colum with a unique index for each time series. The remaining columns have observation of the state variables at each point in time and for each time series.
"""
function MultiNODE(data;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)
    
    # convert data
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)

    process_model =  MultiNODE_process(dims,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(T,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    # parameters
    parameters = (uhat = zeros(size(data)), 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    constructor = (data) -> MultiNODE(data; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)
    
    return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)
    
end 


function MultiNODE(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)
    covariates = interpolate_covariates_multi(X)

    process_model = MultiNODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(T,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    # parameters
    parameters = (uhat = zeros(size(data)), 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    constructor = (data) -> MultiNODE(data,X; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)
    
    return MultiUDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)
    
end 


function MultiNODESimplex(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)
    

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)
    covariates = interpolate_covariates_multi(X)

    process_model = MultiNODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = softmax()
    observation_loss = softmaxMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()

    parameters = (uhat = zeros(size(data)), 
            process_model = process_model.parameters,
            process_loss = process_loss.parameters,
            observation_model = observation_model.parameters,
            observation_loss = observation_loss.parameters,
            process_regularization = process_regularization.reg_parameters, 
            observation_regularization = observation_regularization.reg_parameters)

    parameters = ComponentArray(parameters)

    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = (data) -> MultiNODESimplex(data,X; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)

    return MultiUDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)

end 


function MultiNODESimplex(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)

    process_model = MultiNODE_process(dims,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = softmax()
    observation_loss = softmaxMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()

    parameters = (uhat = zeros(size(data)), 
            process_model = process_model.parameters,
            process_loss = process_loss.parameters,
            observation_model = observation_model.parameters,
            observation_loss = observation_loss.parameters,
            process_regularization = process_regularization.reg_parameters, 
            observation_regularization = observation_regularization.reg_parameters)

    parameters = ComponentArray(parameters)

    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = (data) -> MultiNODE(data; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)

    return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)

end 


function MultiCustomDerivatives(data,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    # convert data
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)

    # generate submodels 
    process_model = MultiContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    # model constructor
    constructor = (data) -> MultiCustomDerivatives(data,derivs!,initial_parameters;
                    proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
    return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end


function MultiCustomDerivatives(data,X,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    # convert data
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)
    covariates = interpolate_covariates_multi(X)

    # generate submodels 
    process_model = MultiContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    # model constructor
    constructor = (data) -> MultiCustomDerivatives(data,X,derivs!,initial_parameters;
                    proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
    return MultiUDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end

