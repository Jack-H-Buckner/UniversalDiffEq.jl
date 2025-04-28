"""
    UDE

Basic data structure used to the model structure, parameters and data for UDE and NODE models.
...
# Elements
- times: a vector of times for each observation
- data: a matrix of observations at each time point
- X: a DataFrame with any covariates used by the model
- data_frame: a DataFrame with columns for the time of each observation and values of the state variables
- parameters: a ComponentArray that stores model parameters
- loss_function: the loss function used to fit the model
- process_model: a Julia mutable struct used to define model predictions
- process_loss: a Julia mutable struct used to measure the performance of model predictions
- observation_model: a Julia mutable struct used to predict observations given state variable estimates
- observation_loss: a Julia mutable struct used to measure the performance of the observation model
- process_regularization: a Julia mutable struct used to store data needed for process model regularization
- observation_regularization: a Julia mutable struct used to store data needed for observation model regularization
- constructor: A function that initializes a UDE model with identical structure.
- time_column_name: A string with the name of the column used for
- weights
- variable_column_name
- value_column_name
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
    weights
    variable_column_name
    value_column_name
    solvers
end




"""
    CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )

Constructs a UDE model for the data set `data`  based on user-defined derivatives `derivs`. An initial guess of model parameters are supplied with the `initial_parameters` argument.

...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point.
- derivs: a Function of the form `derivs!(du,u,p,t)` where `u` is the value of the state variables, `p` are the model parameters, `t` is time, and du is updated with the value of the derivatives
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `proc_weight`: Weight of process error ``omega_{proc}``. Default is `1.0`.
- `obs_weight`: Weight of observation error ``omega_{obs}``. Default is `1.0`.
- `reg_weight`: Weight of regularization error ``omega_{reg}``. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
- `ode_solver`: method to aproximate solutions to the differntail equation. Defaul is `Tsit5()`
- `ad_method`:method to evalaute derivatives of the ODE solver. Default is `ForwardDiffSensitivity()`
...
"""
function CustomDerivatives(data,derivs!,initial_parameters;time_column_name = "time",proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2", 
                            ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())

    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)

    # generate submodels
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho; ode_solver = ode_solver, ad_method = ad_method)
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

    # parameters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # model constructor
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,ode_solver = ode_solver, ad_method = ad_method)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,weights,
                nothing, nothing, (ode = ode_solver, ad = ad_method))

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
- `proc_weight`: Weight of process error ``omega_{proc}``. Default is `1.0`.
- `obs_weight`: Weight of observation error ``omega_{obs}``. Default is `1.0`.
- `reg_weight`: Weight of regularization error ``omega_{reg}``. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
- `ode_solver`: method to aproximate solutions to the differntail equation. Defaul is `Tsit5()`
- `ad_method`:method to evalaute derivatives of the ODE solver. Default is `ForwardDiffSensitivity()`
"""
function CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;
                                time_column_name = "time",proc_weight=1.0,obs_weight=1.0,
                                reg_weight=10^-6,extrap_rho=0.0,l=10.0^6,reg_type = "L2",
                                ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name )

    # generate submodels
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho; ode_solver = ode_solver, ad_method=ad_method)
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

    # parameters vector
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
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters,priors;time_column_name = time_column_name ,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,ode_solver = ode_solver, ad_method=ad_method)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, weights,
                nothing, nothing,(ode = ode_solver, ad = ad_method) )

end


"""
    CustomDerivatives(data::DataFrame,X::DataFrame,derivs!::Function,initial_parameters;kwargs ... )

When a data frame `X` is supplied the model will run with covariates. The argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame.

When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`.

# kwargs

- `time_column_name`: Name of column in `data` and `X` that corresponds to time. Default is `"time"`.
- `variable_column_name`: Name of column in `X` that corresponds to the variables. Default is `nothing`.
- `value_column_name`: Name of column in `X` that corresponds to the covariates. Default is `nothing`.
- `proc_weight`: Weight of process error ``omega_{proc}``. Default is `1.0`.
- `obs_weight`: Weight of observation error ``omega_{obs}``. Default is `1.0`.
- `reg_weight`: Weight of regularization error ``omega_{reg}``. Default is `10^-6`.
- `reg_type`: Type of regularization, whether `"L1"` or `"L2"` regularization. Default is `"L2"`.
- `l`: Extrapolation parameter for forecasting. Default is `0.25`.
- `extrap_rho`: Extrapolation parameter for forecasting. Default is `0.0`.
- `ode_solver`: method to aproximate solutions to the differntail equation. Defaul is `Tsit5()`.
- `ad_method`:method to evalaute derivatives of the ODE solver. Default is `ForwardDiffSensitivity()`.
"""
function CustomDerivatives(data::DataFrame,X::DataFrame,derivs!::Function,initial_parameters;time_column_name = "time",
                            variable_column_name = nothing ,value_column_name = nothing,
                            proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,
                            reg_type = "L2",ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())

    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho; ode_solver = ode_solver , ad_method = ad_method)
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

    # parameters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # model constructor
    constructor = data -> CustomDerivatives(data,X,derivs!,initial_parameters;time_column_name = time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name, proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type,ode_solver = ode_solver , ad_method = ad_method)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,
                weights,variable_column_name,value_column_name,(ode = ode_solver, ad = ad_method))

end


function CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters,priors::Function;time_column_name = "time",variable_column_name = nothing ,value_column_name = nothing, 
                                proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,
                                extrap_rho=0.1,l=0.25,reg_type = "L2",
                                ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho; ode_solver = ode_solver, ad_method = ad_method)
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

    # parameters vector
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
    constructor = data -> CustomDerivatives(data,X,derivs!,initial_parameters,priors;time_column_name=time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name, proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type, ode_solver = ode_solver, ad_method = ad_method)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, weights,
                variable_column_name,value_column_name,(ode = ode_solver, ad = ad_method))

end


function CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters,observation_loss::Function, observation_parameters::NamedTuple,priors::Function;time_column_name = "time",
                            variable_column_name = nothing ,value_column_name = nothing,
                             proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,
                             l=0.25,reg_type = "L2",
                             ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho;ode_solver = ode_solver, ad_method = ad_method)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = LossFunction(observation_loss, observation_parameters)
    process_regularization = L2(initial_parameters,weight=reg_weight)

    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()

    # parameters vector
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
    constructor = data -> CustomDerivatives(data,X,derivs!,initial_parameters,priors;time_column_name=time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name, proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type, ode_solver = ode_solver, ad_method = ad_method)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, weights,
                variable_column_name,value_column_name,(ode = ode_solver, ad = ad_method))

end



function CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters,observation_loss::Function, 
                            observation_parameters::NamedTuple; time_column_name = "time",
                            variable_column_name = nothing ,value_column_name = nothing, 
                            proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.0,l=10000000000.0,
                            reg_type = "L2",ode_solver = Tsit5(), ad_method = ForwardDiffSensitivity())
    
    X_data_frame = X
    time_column_name, series_column_name, value_column_name, variable_column_name = check_column_names(data, X, time_column_name = time_column_name,value_column_name = value_column_name, variable_column_name = variable_column_name)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name)
    covariates, vars = interpolate_covariates(X,time_column_name,variable_column_name,value_column_name)

    # generate submodels
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho; ode_solver = ode_solver, ad_method = ad_method)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = LossFunction(observation_loss, observation_parameters)
    process_regularization = L2(initial_parameters,weight=reg_weight)

    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()

    # parameters vector
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
    constructor = data -> CustomDerivatives(data,X,derivs!,initial_parameters,priors;time_column_name=time_column_name, variable_column_name=variable_column_name, value_column_name=value_column_name, proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type,ode_solver = ode_solver, ad_method = ad_method)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, weights,
                variable_column_name,value_column_name,(ode = ode_solver, ad = ad_method))

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
- `proc_weight`: Weight of process error ``omega_{proc}``. Default is `1.0`.
- `obs_weight`: Weight of observation error ``omega_{obs}``. Default is `1.0`.
- `reg_weight`: Weight of regularization error ``omega_{reg}``. Default is `10^-6`.
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

    # parameters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # model constructor
    constructor = data -> CustomDifference(data,step,initial_parameters;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,weights,
                nothing, nothing, nothing)

end

"""
    CustomDifference(data::DataFrame,step,initial_parameters,priors::Function;kwargs ... )

When a function's priors is supplied its value will be added to the loss function as a penalty term for user-specified parameters. It should take the a single NamedTuple `p` as an argument penalties for each parameter should be calcualted by accessing `p` with the period operator.

```julia
function priors(p)
    l = 0.01*(p.r - 1.5)^2
    l += 0.01*(p.beta)^2
    return l
end
```

# kwargs

- `time_column_name`: Name of column in `data` that corresponds to time. Default is `"time"`.
- `proc_weight`: Weight of process error ``omega_{proc}``. Default is `1.0`.
- `obs_weight`: Weight of observation error ``omega_{obs}``. Default is `1.0`.
- `reg_weight`: Weight of regularization error ``omega_{reg}``. Default is `10^-6`.
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

    # parameters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = data -> CustomDifference(data,step,initial_parameters,priors;time_column_name=time_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,0,dataframe,0,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name,weights,
                nothing, nothing, nothing)

end

"""
    CustomDifference(data::DataFrame,X,step,initial_parameters;kwargs ... )

When a data frame `X` is supplied the model will run with covariates. The argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame.

When `X` is provided the step function must have the form `step(u,x,t,p)` where `x` is a vector with the value of the covariates at time `t`.

    # kwargs

- `time_column_name`: Name of column in `data` and `X` that corresponds to time. Default is `"time"`.
- `variable_column_name`: Name of column in `X` that corresponds to the variables. Default is `nothing`.
- `value_column_name`: Name of column in `X` that corresponds to the covariates. Default is `nothing`.
- `proc_weight`: Weight of process error ``omega_{proc}``. Default is `1.0`.
- `obs_weight`: Weight of observation error ``omega_{obs}``. Default is `1.0`.
- `reg_weight`: Weight of regularization error ``omega_{reg}``. Default is `10^-6`.
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

    # parameters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # model constructor
    constructor = data -> CustomDifference(data,X,step,initial_parameters;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, weights,
                variable_column_name,value_column_name, nothing)

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

    # parameters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = data -> CustomDifference(data,X,step,initial_parameters,priors;time_column_name=time_column_name,variable_column_name=variable_column_name,value_column_name=value_column_name,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)

    weights = (regularization =  reg_weight, process = proc_weight, observation = obs_weight)

    return UDE(times,data,X,dataframe,X_data_frame,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor,time_column_name, weights,
                variable_column_name,value_column_name, nothing)

end

