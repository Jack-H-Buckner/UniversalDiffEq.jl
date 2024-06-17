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

include("Optimizers.jl")

"""
    CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )

Constructs a UDE model for the data set `data`  based on user defined derivatives `derivs`. An initial guess of model parameters are supplied with the `initial_parameters` argument. 

...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- derivs: a Function of the form `derivs!(du,u,p,t)` where `u` is the value of the state variables, `p` are the model parameters, `t` is time, and du is updated with the value of the derivatives
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.
...
"""
function CustomDerivatives(data,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end

function CustomDerivs(data,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25)
    CustomDerivatives(data,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
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
"""
function CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters,priors;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)

end 
    

"""
    CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 

When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`. 
"""
function CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end


function CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters,priors::Function;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters,priors;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
   
end 

"""
    CustomDifference(data,step,initial_parameters;kwrags...)

Constructs a UDE model for the data set `data` based on user defined difference equation `step`. An initial guess of model parameters are supplied with the initial_parameters argument.
...
# Arguments
- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- step: a Function of the form `step(u,t,p)` where `u` is the value of the state variables, `p` are the model parameters.
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.
...
"""
function CustomDifference(data,step,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type="L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    constructor = data -> CustomDifference(data,step,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
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
"""
function CustomDifference(data::DataFrame,step,initial_parameters,priors::Function;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type="L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    constructor = data -> CustomDifference(data,step,initial_parameters,priors;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end

"""
    CustomDifference(data::DataFrame,X,step,initial_parameters;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 

When `X` is provided the step function must have the form `step(u,x,t,p)` where `x` is a vector with the value of the covariates at time `t`. 
"""
function CustomDifference(data::DataFrame,X,step,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    constructor = (data,X) -> CustomDifference(data,X,step,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end


function CustomDifference(data::DataFrame,X,step,initial_parameters,priors::Function;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    constructor = (data,X) -> CustomDifference(data,X,step,initial_parameters,priors;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type=reg_type)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end



"""
    NNDE(data;kwargs ...)

Constructs a nonparametric discrete time model for the data set `data` using a single layer neural network to represent the systems dynamics. 
"""
function NNDE(data;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    
    constructor = data -> NNDE(data;hidden_units=hidden_units,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                                observation_loss,process_regularization,observation_regularization,constructor)
    
end 




"""
    DiscreteUDE(data,step,init_parameters;kwargs ...)

Constructs an additive `UDE` model with user supplied difference equations `step` and a single layer neural network. When `init_parameters` are provided for the user supplied function their values will be estimated in the training process.  

# Model equaitons
```math
x_{t+1} = f(x_t;\theta) + NN(x_t;w,b)
```

...
# Key word arguments

- proc_weight=1.0 : Weight given to the model predictions in loss funciton
- obs_weight=1.0 : Weight given to the state estimates in loss function 
- reg_weight=10^-6 : Weight given to regularization in the loss function 
- extrap_rho=0.0 : Asymptotic value of derivatives when extrapolating (negative when extrapolating higher than past observations, positive when extrapolating lower)
- l=0.25 : rate at which extrapolations converge on asymptotic behavior
...
"""
function DiscreteUDE(data,step,init_parameters;
                        hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 0.0,extrap_rho = 0.1,l = 0.25)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
    # submodels
    process_model = DiscreteModelErrors(dims,step,init_parameters,hidden_units,seed,extrap_rho,l)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    DiscreteUDE(data,step,init_parameters;
                        hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 0.0,extrap_rho = 0.1,l = 0.25)
    
    constructor = data -> DiscreteUDE(data,step,init_parameters; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 



function DiscreteUDE(data,known_dynamics;hidden_units=10,seed = 1,errors_weight=0.1,MSE_weight=1.0,obs_weight=1.0,reg_weight = 0.00)
    
    init_parameters = NamedTuple()
    step = (x,p) -> known_dynamics(x)
    
    return DiscreteUDE(data,step,init_parameters; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
end 


"""
    NODE(data;kwargs ... )


Constructs a nonparametric continuous time model for the data set `data` using a single layer neural network to represent the systems dynamics. 
"""
function NODE(data;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0 )
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    
    
    constructor = (data) -> NODE(data;hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 


"""
    NODE(data,X;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame. 

"""
function NODE(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0 )
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    
    
    constructor = (data,X) -> NODE(data,X;hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,l=l,extrap_rho=extrap_rho)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 

"""
    EasyNODE(data;kwargs ... )
Constructs a pretrained continuous time model for the data set `data` using a single layer neural network to represent the systems dynamics. 
"""
function EasyNODE(data;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0, step_size = 0.05, maxiter = 500, verbose = false)
   
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    
    
    constructor = (data) -> NODE(data;hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    untrainedNODE = UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
        observation_loss,process_regularization,observation_regularization,constructor)
    
    return gradient_descent!(untrainedNODE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end 

"""
    EasyNODE(data,X;kwargs ... )
When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame. 
"""
function EasyNODE(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0, step_size = 0.05, maxiter = 500, verbose = false)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)
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
    
    
    constructor = (data,X) -> NODE(data,X;hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,l=l,extrap_rho=extrap_rho)
    
    untrainedNODE = UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
        observation_loss,process_regularization,observation_regularization,constructor)
        return gradient_descent!(untrainedNODE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end 

"""
    EasyUDE(data,derivs!,initial_parameters;kwargs ... )
Constructs a pretrained UDE model for the data set `data`  based on user defined derivatives `derivs`. An initial guess of model parameters are supplied with the `initial_parameters` argument. 
"""
function EasyUDE(data,known_dynamics!,initial_parameters;hidden_units = 10, seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2", step_size = 0.05, maxiter = 500, verbose = false)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
        NNcomp = NN(du,u,parameters.NN,t)
        knowncomp = known_dynamics!(du,u,parameters.known,t)
        du .= NNcomp .+ knowncomp
    end
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
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    untrainedUDE = UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    return gradient_descent!(untrainedUDE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end

"""
    EasyUDE(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )
When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 
When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`. 
"""
function EasyUDE(data::DataFrame,X,known_dynamics!::Function,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)
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
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    untrainedUDE = UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    return gradient_descent!(untrainedUDE, step_size = step_size, maxiter = maxiter, verbose = verbose)
end


"""
    BayesianNODE(data;kwargs ... )
Constructs a Bayesian continuous time model for the data set `data` using a single layer neural network to represent the systems dynamics. 
"""
function BayesianNODE(data;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0, step_size = 0.05, maxiter = 500, verbose = false)
   
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    
    
    constructor = (data) -> NODE(data;hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return BayesianUDE(times,data,0,dataframe,parameters_vector,loss_function,process_model,process_loss,observation_model,observation_loss,
            process_regularization,observation_regularization,constructor)
end 
"""
    BayesianNODE(data,X;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for values of time not included in the data frame. 

"""
function BayesianNODE(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6, reg_type = "L2", l = 0.25,extrap_rho = 0.0 )
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    
    
    constructor = (data,X) -> NODE(data,X;hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type,l=l,extrap_rho=extrap_rho)
    
    return BayesianUDE(times,data,X,dataframe,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 

"""
    BayesianCustomDerivatives(data,derivs!,initial_parameters;kwargs ... )

Constructs a Bayesian UDE model for the data set `data`  based on user defined derivatives `derivs`. An initial guess of model parameters are supplied with the `initial_parameters` argument. 

...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- derivs: a Function of the form `derivs!(du,u,p,t)` where `u` is the value of the state variables, `p` are the model parameters, `t` is time, and du is updated with the value of the derivatives
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.
...
"""
function BayesianCustomDerivatives(data,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    return BayesianUDE(times,data,X,dataframe,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
   

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
"""
function BayesianCustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
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
    loss_ = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    loss_function = parameters -> loss_(parameters) + priors(parameters.process_model)
    # model constructor
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters,priors;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    return BayesianUDE(times,data,X,dataframe,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
   
end 
    

"""
    CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 

When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`. 
"""
function BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    return BayesianUDE(times,data,X,dataframe,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
   
end


function BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters,priors::Function;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

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
    constructor = (data,X) -> CustomDerivatives(data,X,derivs!,initial_parameters,priors;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l,reg_type = reg_type)
    
    return BayesianUDE(times,data,X,dataframe,parameters_vector,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
   
end 