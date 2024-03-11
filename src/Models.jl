
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
CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )

Constructs a UDE model for the data set `data`  based on user defined derivitivs `derivs`. An initial guess of model parameters are supplied with the `initia_parameters` argument. 

...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- derivs: a Function of the form `derivs!(du,u,p,t)` where `u` is the value of the state variables, `p` are the model parameters, `t` is time, and du is updated with the value of the derivitives
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.

# Key word arguments

- proc_weight=1.0 : Weight given to the model predictiosn in loss funciton
- obs_weight=1.0 : Weight given to the state estiamtes in loss function 
- reg_weight=10^-6 : Weight given to regularization in the loss function 
- extrap_rho=0.0 : Asymthotic value of derivitives when extrapolating (negative when extrapolating higher than past observaitons, postive when extrapolating lower)
- l=0.25 : rate at which extrapolations converge on asymthotic behavior
...
"""
function CustomDerivatives(data,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data) -> CustomDerivs(data,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end

function CustomDerivatives(data,X,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data,X) -> CustomDerivs(data,X,derivs!,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end



"""
    CustomDiffernce(data,derivs,initial_parameters;kwrags...)

Constructs a UDE model for the data set `data` based on user defined difference equation `step`. An initial guess of model parameters are supplied with the initia_parameters argument.

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- step: a Function of the form `step(u,p)` where `u` is the value of the state variables, `p` are the model parameters.
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.
...
# Key word arguments

- proc_weight=1.0 : Weight given to the model predictiosn in loss funciton
- obs_weight=1.0 : Weight given to the state estiamtes in loss function 
- reg_weight=10^-6 : Weight given to regularization in the loss function 
- extrap_rho=0.0 : Asymthotic value of derivitives when extrapolating (negative when extrapolating higher than past observaitons, postive when extrapolating lower)
- l=0.25 : rate at which extrapolations converge on asymthotic behavior
...
"""
function CustomDiffernce(data,step,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
    # generate submodels 
    process_model = DiscreteProcessModel(step,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data) -> CustomDerivs(data,derivs,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end

function CustomDiffernce(data,X,step,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    covariates = interpolate_covariates(X)

    # generate submodels 
    process_model = DiscreteProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    # model constructor
    constructor = (data,X) -> CustomDerivs(data,X,derivs,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end




"""
NNDE(data;kwargs ...)

Constructs a nonparametric discrete time model for the data set `data` using a single layer neural network to reporesent the systems dynamics. 

# Model equations 
```math
x_{t+1} = NN(x_t;w,b)
```
...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 

# Key word arguments

- proc_weight=1.0 : Weight given to the model predictiosn in loss funciton
- obs_weight=1.0 : Weight given to the state estiamtes in loss function 
- reg_weight=10^-6 : Weight given to regularization in the loss function 
- extrap_rho=0.0 : Asymthotic value of derivitives when extrapolating (negative when extrapolating higher than past observaitons, postive when extrapolating lower)
- l=0.25 : rate at which extrapolations converge on asymthotic behavior
...
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
    
    constructor = data -> NNDE(data;hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                                observation_loss,process_regularization,observation_regularization,constructor)
    
end 




"""
    DiscreteUDE(data,step,init_parameters;kwargs ...)

Constructs an additive `UDE` model with user supplied difference equations `step` and a single layer neural network. When `init_parameters` are provided for the use supplied function their values will be estiated in the training process.  

# Model equaitons
```math
x_{t+1} = f(x_t;\theta) + NN(x_t;w,b)
```

...
# Key word arguments

- proc_weight=1.0 : Weight given to the model predictiosn in loss funciton
- obs_weight=1.0 : Weight given to the state estiamtes in loss function 
- reg_weight=10^-6 : Weight given to regularization in the loss function 
- extrap_rho=0.0 : Asymthotic value of derivitives when extrapolating (negative when extrapolating higher than past observaitons, postive when extrapolating lower)
- l=0.25 : rate at which extrapolations converge on asymthotic behavior
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
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
  
    constructor = data -> DiscreteUDE(data,step,init_parameters; hidden_units=hidden_units,seed=seed,errors_weight=errors_weight,MSE_weight=MSE_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 



function DiscreteUDE(data,known_dynamics;hidden_units=10,seed = 1,errors_weight=0.1,MSE_weight=1.0,obs_weight=1.0,reg_weight = 0.00)
    
    init_known_dynamics_parameters = NamedTuple()
    known_dynamics_function = (x,p) -> known_dynamics(x)
    
    return DiscreteUDE(data,known_dynamics_function,init_known_dynamics_parameters; hidden_units=hidden_units,seed=seed,errors_weight=errors_weight,MSE_weight=MSE_weight,obs_weight=obs_weight,reg_weight = reg_weight)
end 


"""
    NODE(data;kwargs ... )


Constructs a nonparametric continuous time model for the data set `data` using a single layer neural network to represent the systems dynamics. 

# Model equations 
```math
    \frac{dx}{dt} = NN(x;w,b)
```

...
# Arguments

- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point.

# Key word arguments

- proc_weight=1.0 : Weight given to the model predictiosn in loss funciton
- obs_weight=1.0 : Weight given to the state estiamtes in loss function 
- reg_weight=10^-6 : Weight given to regularization in the loss function 
- extrap_rho=0.0 : Asymthotic value of derivitives when extrapolating (negative when extrapolating higher than past observaitons, postive when extrapolating lower)
- l=0.25 : rate at which extrapolations converge on asymthotic behavior
...
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
    
    
    constructor = (data,X) -> NODE(data;hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 

"""
UDE(data,derivs,init_parameters;kwargs...)

Constructs an additive continuous time `UDE` model with user supplied derivitives `step` and a single layer neural network. When `init_parameters` are provided for the user supplied function their values will be estiated during model training. 

# Model equaitons
```math
    \frac{dx}{dt} = f(x;\theta) + NN(x;w,b)
```

...
# Key word arguments

- proc_weight=1.0 : Weight given to the model predictiosn in loss funciton
- obs_weight=1.0 : Weight given to the state estiamtes in loss function 
- reg_weight=10^-6 : Weight given to regularization in the loss function 
- extrap_rho=0.0 : Asymthotic value of derivitives when extrapolating (negative when extrapolating higher than past observaitons, postive when extrapolating lower)
- l=0.25 : rate at which extrapolations converge on asymthotic behavior
...
"""
function UDE(data,known_dynamics,init_known_dynamics_parameters;hidden_units=10,seed=1,proc_weight=1.0,obs_weight=1.0,reg_weight=10.0^-6,l=0.25,extrap_rho = 0.1)
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    
    # submodels
    process_model = ContinuousModelErrors(dims,known_dynamics,init_known_dynamics_parameters,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    
    constructor = data -> UDE(data,known_dynamics,init_known_dynamics_parameters;
                                    hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return UDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end 


function UDE(data,known_dynamics;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 0.0)
    
    known_dynamics1 = (u,p) -> known_dynamics(u)
    init_known_dynamics_parameters = NamedTuple()
    
    return ContinuousModelErrors(data,known_dynamics1,init_known_dynamics_parameters;
                                    hidden_units=hidden_units,seed = seed,proc_weight=proc_weight,
                                    obs_weight=obs_weight,reg_weight = reg_weight)
end 



