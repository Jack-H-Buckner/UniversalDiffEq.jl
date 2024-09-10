"""
    MultiUDE

Data structure used to the model structure, parameters and data for UDE and NODE models in the case multiple time series are used for fitting. 
...
# Elements
- times: a vector of times for each observation
- data: a matrix of observaitons at each time point
- X: a DataFrame with any covariates used by the model
- data_frame: a DataFrame with columns for the time of each observation and values of the state variables
- parameters: a ComponentArray that stores model parameters
- loss_function: the loss function used to fit the model
- process_model: a Julia mutable struct used to define model predictions 
- process_loss: a Julia mutable struct used to measure the performance of model predictions
- observation_model: a Julia mutable struct used to predict observaitons given state variable estimates
- observaiton_loss: a Julia mutable struct used to measure the performance of the observation model
- process_regularization: a Julia mutable struct used to store data needed for process model regularization
- observation_regularization: a Julia mutable struct used to store data needed for observation model regularization
- constructor: A function that initializes a UDE model with identical structure.
- time_column_name: Name of the column used to identify time steps.
- series_column_name: Name of the column used to identify different time series.
- series_labels: Labels used to identify different time series.
- varnames: Names of variables.
...
"""

mutable struct MultiUDE
    times
    data
    X
    data_frame
    X_data_frame
    parameters
    loss_function
    process_model
    process_loss 
    observation_model
    observation_loss 
    process_regularization
    observation_regularization
    constructor
    time_column_name
    series_column_name
    variable_column_name 
    value_column_name
    series_labels
    varnames
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
            dt = time[t]-time[t-1]
            u1hat, epsilon = process_model.predict(u0,series,time[t-1],dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return L_obs + L_proc + L_reg
    end
    return loss
end



function init_single_loss_skip(process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    function loss(parameters,times,data,series,starts,lengths,t_skip)
        
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
            dt = time[t]-time[t-1]
            if !(isapprox(time[t-1], t_skip))
                u1hat, epsilon = process_model.predict(u0,series,time[t-1],dt,parameters.process_model) 
                L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
            end
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)

        return L_obs + L_proc + L_reg
    end
    return loss
end

function init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,time_column_name,series_column_name,labels_df)
    
    single_loss = init_single_loss(process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    function loss(parameters)
        L = 0
        for i in eachindex(starts)
            L+= single_loss(parameters,times,data,i,starts,lengths)
        end
        return L
    end   

    single_loss_skip = init_single_loss_skip(process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    function loss(parameters,skips::DataFrame)
        L = 0
        for i in eachindex(starts)
            t_skip = skips[skips[:,series_column_name] .== labels_df.label[i],time_column_name] # assumes series are given IDs listed 1:n 
            if length(t_skip) > 0
                L+= single_loss_skip(parameters,times,data,i,starts,lengths,t_skip[1])
            end
        end
        return L
    end 

    function loss(parameters,inds::Vector)
        L = 0
        for i in inds
            L+= single_loss(parameters,times,data,i,starts,lengths)
        end
        return L
    end 
    
    
    return loss
        
end
    


"""
    MultiNODE(data;kwargs...)

builds a NODE model to fit to the data. `data` is a DataFrame object with time arguments placed in a column labed `t` and a second column with a unique index for each time series. The remaining columns have observations of the state variables at each point in time and for each time series.
"""
function MultiNODE(data;time_column_name = "time", series_column_name = "series",hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)

    time_column_name, series_column_name = check_column_names(data, time_column_name = time_column_name, series_column_name = series_column_name)
    # convert data
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(data, time_column_name, series_column_name)

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
    
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,time_column_name,series_column_name,labels_df )
    
    constructor = (data) -> MultiNODE(data;time_column_name = time_column_name , series_column_name = time_column_name ,hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)
    
    return MultiUDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor,time_column_name, series_column_name,nothing,nothing,labels_df,varnames)
    
end 


"""
    MultiNODE(data,X;kwargs...)

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame. 
"""
function MultiNODE(data,X;time_column_name = "time", series_column_name = "series", variable_column_name = nothing, value_column_name = nothing,hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name, series_column_name = series_column_name, value_column_name = value_column_name, variable_column_name = variable_column_name)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths,varnames, labels_df = process_multi_data(data, time_column_name, series_column_name)
    covariates, variables = interpolate_covariates(X, time_column_name, series_column_name,  variable_column_name, value_column_name)


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
    
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,time_column_name,series_column_name,labels_df )
    
    constructor = (data,X) -> MultiNODE(data,X;time_column_name = time_column_name , series_column_name =  series_column_name, variable_column_name = variable_column_name, value_column_name = value_column_name,
                                        hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)
    
    return MultiUDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor,time_column_name, series_column_name, variable_column_name, value_column_name,labels_df,varnames)
    
end 


function MultiCustomDerivatives(data,derivs!,initial_parameters;time_column_name = "time", series_column_name = "series",proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    time_column_name, series_column_name = check_column_names(data, time_column_name = time_column_name, series_column_name = series_column_name)

    # convert data
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(data, time_column_name, series_column_name)
    
    # generate submodels 
    process_model = MultiContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,time_column_name,series_column_name,labels_df )

    # model constructor
    constructor = (data) -> MultiCustomDerivatives(data,derivs!,initial_parameters;time_column_name = time_column_name , series_column_name =  series_column_name,
                    proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
    return MultiUDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, series_column_name,nothing,nothing,labels_df,varnames)

end


function MultiCustomDerivatives(data,X,derivs!,initial_parameters;time_column_name = "time", series_column_name = "series", variable_column_name = nothing, value_column_name = nothing,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name, series_column_name = series_column_name, value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(data, time_column_name, series_column_name)
    covariates, variables = interpolate_covariates(X, time_column_name, series_column_name,  variable_column_name, value_column_name)

    # generate submodels 
    process_model = MultiContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,time_column_name,series_column_name,labels_df )

    # model constructor
    constructor = (data,X) -> MultiCustomDerivatives(data,X,derivs!,initial_parameters;time_column_name = time_column_name , series_column_name =  series_column_name,
                        variable_column_name = variable_column_name, value_column_name = value_column_name, proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
    return MultiUDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, series_column_name,variable_column_name,value_column_name, labels_df,varnames)

end

