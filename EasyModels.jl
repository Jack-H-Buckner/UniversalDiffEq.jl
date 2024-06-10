include("Models.jl")
include("Optimizers.jl")

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
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end

"""
    EasyUDE(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )

When a dataframe `X` is supplied the model will run with covariates. the argument `X` should have a column for time `t` with the value for time in the remaining columns. The values in `X` will be interpolated with a linear spline for value of time not included in the data frame. 

When `X` is provided the derivs function must have the form `derivs!(du,u,x,p,t)` where `x` is a vector with the value of the covariates at time `t`. 
"""
function CustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
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
    
    return UDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end
