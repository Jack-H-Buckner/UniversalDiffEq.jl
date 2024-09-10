"""
    UDE

Basic data structure used to the model structure, parameters and data for UDE and NODE models. 
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
...
"""
mutable struct UDE
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
    variable_column_name 
    value_column_name
end

"""
    BayesianUDE

Basic data structure used to the model structure, parameters and data for Bayesian UDE and NODE models. 
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
...
"""
mutable struct BayesianUDE
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
    variable_column_name 
    value_column_name 
end


"""
    CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )

Constructs a UDE model for the data set `data`  based on user defined derivatives `derivs`. An initial guess of model parameters are supplied with the `initial_parameters` argument. 

...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- derivs: a Function of the form `derivs!(du,u,p,t)` where `u` is the value of the state variables, `p` are the model parameters, `t` is time, and du is updated with the value of the derivatives
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
...
"""
function CustomDerivatives(data,derivs!,initial_parameters;time_column_name = "time",proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                nothing, nothing)

end


"""
    CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )

When a function's priors is supplied its value will be added to the loss function as a penalty term for user specified parameters. It should take the a single NamedTuple `p` as an argument penalties for each paramter should be calculated by accessing `p` with the period operator.
    
The prior function can be used to nudge the fitted model toward prior expectations for a parameter value. For example, the following function increases the loss when a parameter `p.r` has a value other than 1.5, nad a second parameter `p.beta` is greater than zeros. 

```julia 
function priors(p)
    l = 0.01*(p.r - 1.5)^2
    l += 0.01*(p.beta)^2
    return l
end 
```

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;time_column_name = "time",proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name )
    
    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    function loss_function(parameters)
        loss_(parameters) + priors(parameters.process_model)
    end
    function loss_function(parameters,tskip)
        loss_(parameters,tskip) + priors(parameters.process_model)
    end
    # model constructor
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters,priors;time_column_name = time_column_name ,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                nothing, nothing )

end 
    

"""
    CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 

When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`. 

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `variable_column_name`: Name of column in `data` that corresponds to the variables. Default is `"variable"`.
- `value_column_name`: Name of column in `data` that corresponds to the covariates. Default is `"value"`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function CustomDerivatives(data::DataFrame,X::DataFrame,derivs!::Function,initial_parameters;time_column_name = "time",variable_column_name = nothing ,value_column_name = nothing,proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters;time_column_name = time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name, proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name,value_column_name)

end


function CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters,priors::Function;time_column_name = "time",variable_column_name = nothing ,value_column_name = nothing, proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    function loss_function(parameters)
        loss_(parameters) + priors(parameters.process_model)
    end
    function loss_function(parameters,tskip)
        loss_(parameters,tskip) + priors(parameters.process_model)
    end
    # model constructor
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters,priors;time_column_name=time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name, proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name,value_column_name)
   
end 

"""
    CustomDifference(data,step,initial_parameters;kwrags...)

Constructs a UDE model for the data set `data` based on user defined difference equation `step`. An initial guess of model parameters are supplied with the initial_parameters argument.
...
# Arguments
- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- step: a Function of the form `step(u,t,p)` where `u` is the value of the state variables, `p` are the model parameters.
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
...
"""
function CustomDifference(data,step,initial_parameters;time_column_name = "time",proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type="L2")
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # generate submodels 
    process_model = DiscreteProcessModel(step,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = data -> CustomDifference(data,step,initial_parameters;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                nothing, nothing)
    
end

"""
    CustomDifference(data::DataFrame,step,initial_parameters,priors::Function;kwargs ... )

When a function's priors is supplied its value will be added to the loss function as a penalty term for user specified paramters. It should take the a single NamedTuple `p` as an argument penalties for each parameter should be calcualted by accessing `p` with the period operator. 

```julia 
function priors(p)
    l = 0.01*(p.r - 1.5)^2
    l += 0.01*(p.beta)^2
    return l
end 
```

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function CustomDifference(data::DataFrame,step,initial_parameters,priors::Function;time_column_name = "time",proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type="L2")
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # generate submodels 
    process_model = DiscreteProcessModel(step,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = data -> CustomDifference(data,step,initial_parameters,priors;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, 
                nothing, nothing)
    
end

"""
    CustomDifference(data::DataFrame,X,step,initial_parameters;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 

When `X` is provided the step function must have the form `step(u,x,t,p)` where `x` is a vector with the value of the covariates at time `t`. 

    # kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `variable_column_name`: Name of column in `data` that corresponds to the variables. Default is `"variable"`.
- `value_column_name`: Name of column in `data` that corresponds to the covariates. Default is `"value"`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function CustomDifference(data::DataFrame,X,step,initial_parameters;time_column_name = "time", variable_column_name = nothing ,value_column_name = nothing, proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type = "L2")
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels 
    process_model = DiscreteProcessModel(step,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data,X) -> CustomDifference(data,X,step,initial_parameters;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name,value_column_name)
    
end


function CustomDifference(data::DataFrame,X,step,initial_parameters,priors::Function;time_column_name = "time",variable_column_name = nothing ,value_column_name = nothing,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type = "L2")
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels 
    process_model = DiscreteProcessModel(step,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = (data,X) -> CustomDifference(data,X,step,initial_parameters,priors;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name,value_column_name)
    
end



"""
    NNDE(data;kwargs ...)

Constructs a nonparametric discrete time model for the data set `data` using a single layer neural network to represent the systems dynamics. 

    # kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function NNDE(data;time_column_name = "time",hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # submodels
    process_model = NeuralNetwork(dims,hidden_units,seed,extrap_rho,l)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    constructor = data -> NNDE(data;time_column_name=time_column_name,hidden_units=hidden_units,seed = seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho = 0.1,l = 0.25)
    
    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                                nothing, nothing)
    
end 



function NNDE(data, X;time_column_name = "time",variable_column_name = nothing ,value_column_name = nothing,hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # submodels
    process_model = NeuralNetwork(dims,hidden_units,seed,extrap_rho,l)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    constructor = (data, X) -> NNDE(data, X; time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,hidden_units=hidden_units,seed = seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho = 0.1,l = 0.25)
    
    return UDE(times,data,0,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name, value_column_name)
    
end 


"""
    NODE(data;kwargs ... )


Constructs a nonparametric continuous time model for the data set `data` using a single layer neural network to represent the systems dynamics. 

    # kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.    
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function NODE(data;time_column_name = "time",hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0 )
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # submodels
    process_model = NODE_process(dims,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    
    constructor = (data) -> NODE(data;time_column_name=time_column_name,hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                nothing, nothing)
end 


"""
    NODE(data,X;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame. 

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `variable_column_name`: Name of column in `data` that corresponds to the variables. Default is `"variable"`.
- `value_column_name`: Name of column in `data` that corresponds to the covariates. Default is `"value"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function NODE(data,X;time_column_name = "time", variable_column_name = nothing ,value_column_name = nothing, hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0 )
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # submodels
    process_model = NODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    
    constructor = (data,X) -> NODE(data,X;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,l=l,extrap_rho=extrap_rho)
    
    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name, value_column_name)
end 

"""
    EasyNODE(data;kwargs ... )
Constructs a pretrained continuous time model for the data set `data` using a single layer neural network to represent the systems dynamics. 

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
"""
function EasyNODE(data;time_column_name = "time",hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0, step_size = 0.05, maxiter = 500, verbose = false)
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # submodels
    process_model = NODE_process(dims,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    
    constructor = (data) -> NODE(data;time_column_name=time_column_name,hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    untrainedNODE = UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
        observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
        nothing, nothing)
    
    return gradient_descent!(untrainedNODE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end 

"""
    EasyNODE(data,X;kwargs ... )
When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame. 

# kwargs 

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `variable_column_name`: Name of column in `data` that corresponds to the variables. Default is `"variable"`.
- `value_column_name`: Name of column in `data` that corresponds to the covariates. Default is `"value"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
"""
function EasyNODE(data,X;time_column_name = "time",variable_column_name = nothing ,value_column_name = nothing,hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0, step_size = 0.05, maxiter = 500, verbose = false)
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)
    # submodels
    process_model = NODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    
    constructor = (data,X) -> NODE(data,X;time_column_name=time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name,hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,l=l,extrap_rho=extrap_rho)
    
    untrainedNODE = UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
        observation_loss,process_regularization,observation_regularization,constructor,time_column_name,variable_column_name,value_column_name)
        return gradient_descent!(untrainedNODE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end 

"""
    EasyUDE(data,derivs!,initial_parameters;kwargs ... )
Constructs a pretrained UDE model for the data set `data`  based on user defined derivatives `derivs`. An initial guess of model parameters are supplied with the `initial_parameters` argument. 

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
"""
function EasyUDE(data,known_dynamics!,initial_parameters;time_column_name = "time",hidden_units = 10, seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2", step_size = 0.05, maxiter = 500, verbose = false)
    
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # generate submodels 
    NN = Lux.Chain(Lux.Dense(dims,hidden_units,tanh), Lux.Dense(hidden_units,dims))
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, known = initial_parameters,)
    function nn!(du,u,parameters,t)
        du .= NN(u,parameters,states)[1]
        return du
    end 
    function derivs!(du,u,parameters,t)
        NNcomp = nn!(du,u,parameters.NN,t)
        knowncomp = known_dynamics!(du,u,parameters.known,t)
        du .= NNcomp .+ knowncomp
    end
    process_model = ContinuousProcessModel(derivs!,ComponentArray(parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = data -> CustomDerivatives(data,derivs!,parameters;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    untrainedUDE = UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,nothing,nothing)
    return gradient_descent!(untrainedUDE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end

"""
    EasyUDE(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )
When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 
When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`. 

    # kwargs 

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `variable_column_name`: Name of column in `data` that corresponds to the variables. Default is `"variable"`.
- `value_column_name`: Name of column in `data` that corresponds to the covariates. Default is `"value"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
- `step_size`: Step size for ADAM optimizer. Default is `0.05`.
- `maxiter`: Maximum number of iterations in gradient descent algorithm. Default is `500`.
- `verbose`: Should the training loss values be printed?. Default is `false`.
"""
function EasyUDE(data::DataFrame,X,known_dynamics!::Function,initial_parameters;time_column_name = "time",variable_column_name = "variable",value_column_name = "value",proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)
    # generate submodels 
    NN = Lux.Chain(Lux.Dense(dims+length(covariates(0)),hidden,tanh), Lux.Dense(hidden,dims))
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    parameters = (NN = parameters, known = initial_parameters,)
    function nn!(du,u,parameters,t)
        du .= NN(u,parameters,states)[1]
        return du
    end 
    function derivs!(du,u,parameters,t)
        NNcomp = NN(vcat(u,covariates(t)),parameters.NN,states)[1]
        knowncomp = known_dynamics!(du,u,covariates(t),parameters.known,t)
        du .= NNcomp .+ knowncomp
    end
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters;time_column_name=time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    untrainedUDE = UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,variable_column_name,value_column_name)
    return gradient_descent!(untrainedUDE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end


"""
    BayesianNODE(data;kwargs ... )
Constructs a Bayesian continuous time model for the data set `data` using a single layer neural network to represent the systems dynamics. 

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function BayesianNODE(data;time_column_name = "time",hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0)
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # submodels
    process_model = NODE_process(dims,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = no_reg()
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    parameters_vector = Vector{typeof(parameters)}(undef,1)
    parameters_vector[1] = parameters
    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    
    constructor = (data) -> BayesianNODE(data;time_column_name=time_column_name,hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,l=l,extrap_rho=extrap_rho)
    
    return BayesianUDE(times,data,0,dataframe,0,parameters_vector,loss_function,process_model,process_loss,observation_model,observation_loss,
            process_regularization,observation_regularization,constructor,time_column_name,nothing,nothing)
end 
"""
    BayesianNODE(data,X;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame. 

# kwargs 

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `variable_column_name`: Name of column in `data` that corresponds to the variables. Default is `"variable"`.
- `value_column_name`: Name of column in `data` that corresponds to the covariates. Default is `"value"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function BayesianNODE(data,X;time_column_name = "time",variable_column_name = nothing,value_column_name = nothing,hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0 )
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # submodels
    process_model = NODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = no_reg()
    if reg_type == "L1"
        process_regularization = no_reg()
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    parameters_vector = Vector{typeof(parameters)}(undef,1)
    parameters_vector[1] = parameters

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    
    constructor = (data,X) -> BayesianNODE(data,X;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,l=l,extrap_rho=extrap_rho)
    
    return BayesianUDE(times,data,X,dataframe,X_data_frame,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name, value_column_name)
end 

"""
    BayesianCustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters;kwargs ... )

Constructs a Bayesian UDE model for the data set `data`  based on user defined derivatives `derivs`. An initial guess of model parameters are supplied with the `initial_parameters` argument. 

...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- derivs: a Function of the form `derivs!(du,u,p,t)` where `u` is the value of the state variables, `p` are the model parameters, `t` is time, and du is updated with the value of the derivatives
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.

# kwargs 

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
...
"""
function BayesianCustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters;time_column_name = "time",proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    
    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = no_reg()
    if reg_type == "L1"
        process_regularization = no_reg()
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    parameters_vector = Vector{typeof(parameters)}(undef,1)
    parameters_vector[1] = parameters

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = data -> BayesianCustomDerivatives(data,derivs!,initial_parameters;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    return BayesianUDE(times,data,X,dataframe,0,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,nothing,nothing)
   

end
    

"""
    BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 

When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`. 

# kwargs 

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.  
- `variable_column_name`: Name of column in `data` that corresponds to the variables. Default is `"variable"`.
- `value_column_name`: Name of column in `data` that corresponds to the covariates. Default is `"value"`.  
- `hidden_units`: Number of neurons in hidden layer. Default is `10`.
- `seed`: Fixed random seed for repeatable results. Default is `1`.
- `proc_weight`: Weight of process error `omega_{proc}`. Default is `1.0`.
- `obs_weight`: Weight of observation error `omega_{obs}`. Default is `1.0`.
- `reg_weight`: Weight of regularization error `omega_{reg}`. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
"""
function BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;time_column_name = "time",variable_column_name = nothing,value_column_name = nothing,proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = no_reg()
    if reg_type == "L1"
        process_regularization = no_reg()
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    parameters_vector = Vector{typeof(parameters)}(undef,1)
    parameters_vector[1] = parameters

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data,X) -> BayesianCustomDerivatives(data,X,derivs!,initial_parameters;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return BayesianUDE(times,data,X,dataframe,X_data_frame,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                variable_column_name, value_column_name)
   
end


function BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters,priors::Function;time_column_name = "time",variable_column_name = nothing,value_column_name = nothing,proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = no_reg()
    if reg_type == "L1"
        process_regularization = no_reg()
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    parameters_vector = Vector{typeof(parameters)}(undef,1)
    parameters_vector[1] = parameters

    # loss function 
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = (data,X) -> BayesianCustomDerivatives(data,X,derivs!,initial_parameters,priors;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return BayesianUDE(times,data,X,dataframe,X_data_frame,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,variable_column_name,value_column_name)
   
end 