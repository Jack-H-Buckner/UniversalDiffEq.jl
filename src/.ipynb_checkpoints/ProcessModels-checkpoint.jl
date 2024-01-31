"""
The module defines the process model classes. 

We declare an abstact type AbstractProcessModel of which all the process model classes are subtypes. Instances of AbstractProcessModel sub types are expected to have a field named parameters that stores the model parameter, predict, that takes the state variables u_t, parameters and length of time step Dt and makes a prediction and loss that takes the oberved state u_{t+1}, the predicted state uhat_{t+1} and the model paramters and calcualtes the performance. The parameers object will be assumed to be a Component Array from `ComponentArrays.jl` with field sub fields `parameters.predict` and `parameters.loss`

Each AbstractProcessModel subtype can then have additional fields that store data specific to that model. For example, a simple neural network based process model migth store the neural network object adn the number of input dimensions.
"""
module ProcessModels

using Lux, Random, DifferentialEquations, DiffEqFlux

mutable struct ProcessModel
    parameters
    predict
end

function ProcessModel(derivs!,parameters, dims)

    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,dt,parameters) 
        tspan =  (0.0,dt) 
        sol = solve(IVP, Tsit5(), u0 = u, p=parameters,tspan = tspan, 
                    saveat = (0.0,dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    return ProcessModel(parameters,predict)
end 


mutable struct NeuralNetwork #<: AbstractProcessModel
    dims::Int # number of state variables
    NN # lux neural network object 
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
    loss::Function
end 


function NeuralNetwork(dims;hidden = 10, seed = 1)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    loss_params = NamedTuple()
    
    # parameters
    parameters = (loss = loss_params,predict = NN_parameters)
    #parameters = ComponentArray(parameters)
    
    # Loss
    loss = (u,uhat,parameters) -> sum((u.-uhat).^2)
    
    # prediction 
    predict = (u,parameters) -> u .+ NN(u,parameters.predict,NN_states)[1]
    
    return NeuralNetwork(dims,NN,parameters,predict,loss)
    
end 

function NeuralNetworkErrors(dims;hidden = 10, seed = 1)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    loss_params = (omega=1.0,)
    
    # parameters
    parameters = (loss = loss_params,predict = NN_parameters)
    #parameters = ComponentArray(parameters)
    
    # Loss
    loss = (u,uhat,parameters) -> sum(parameters.loss.omega.^2*(u.-uhat).^2 .-log(abs(parameters.loss.omega)))
    
    # prediction 
    predict = (u,parameters) -> u .+ NN(u,parameters.predict,NN_states)[1]
    
    return NeuralNetwork(dims,NN,parameters,predict,loss)
    
end 



mutable struct NeuralNetwork2
    dims::Int # number of state variables
    NN # lux neural network object 
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
end 


function NeuralNetwork2(dims;hidden = 10, seed = 1)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, NN_states = Lux.setup(rng,NN) 

    # prediction 
    predict = (u,dt,parameters) -> (u .+ NN(u,parameters,NN_states)[1], 0)
    
    return NeuralNetwork2(dims,NN,parameters,predict)
    
end 


mutable struct DiscreteModelErrors
    dims::Int # number of state variables
    error_function # lux neural network object 
    known_dynamics
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
end 

function DiscreteModelErrors(dims,known_dynamics,init_known_dynamics_parameters;hidden = 10, seed = 1)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (known_dynamics = init_known_dynamics_parameters, 
                    NN = NN_parameters)
    # prediction 
    error_function = (u,parameters) -> NN(u,parameters.NN,NN_states)[1]
    known_dynamics_function = (u,parameters) -> known_dynamics(u,parameters.known_dynamics)
    
    function predict(u,dt,parameters) 
        epsilon = NN(u,parameters.NN,NN_states)[1]
        return known_dynamics_function(u,parameters) .+ epsilon, epsilon
    end 
    
    return DiscreteModelErrors(dims,error_function,known_dynamics_function,parameters,predict)
    
end 



mutable struct NODE_process
    dims::Int # number of state variables
    IVP
    derivs!
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
end 


function NODE_process(dims;hidden = 10, seed = 1)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 

    function derivs!(du,u,parameters,t)
        du .= NN(u,parameters,states)[1]
        return du
    end 

    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    function predict(u,dt,parameters) 
        tspan =  (0.0,dt) 
        sol = solve(IVP, Tsit5(), u0 = u, p=parameters,tspan = tspan, 
                    saveat = (0.0,dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end], 0)
    end 
    
    
    return NODE_process(dims,IVP,derivs!,parameters,predict)
    
end 

mutable struct ContinuousModelErrors
    dims::Int # number of state variables
    error_function # lux neural network object 
    known_dynamics
    IVP
    derivs!
    parameters #::ComponentArray # nerual network paramters
    predict::Function # neural network 
end 

function ContinuousModelErrors(dims,known_dynamics,init_known_dynamics_parameters;hidden = 10, seed = 1)
        
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, states = Lux.setup(rng,NN) 
    
    parameters = (NN=NN_parameters,known_dynamics=init_known_dynamics_parameters)
    
    # derivs
    error_function = (u,p) -> NN(u,p.NN,states)[1]
    known_dynamics_function = (u,p) -> known_dynamics(u,p.known_dynamics)
    function derivs!(du,u,p,t)
        du .= known_dynamics_function(u,p) .+ error_function(u,p)
        return du
    end 
    
    # predictions
    u0 = zeros(dims); tspan = (0.0,1000.0) # assing value for the inital conditions and time span (these dont matter, will be replaced below)
    IVP = ODEProblem(derivs!, u0, tspan, parameters)
    
    
    function predict(u,dt,parameters) 
        tspan =  (0.0,dt) 
        sol = solve(IVP, Tsit5(), u0 = u, p=parameters,tspan = tspan, 
                    saveat = (0.0,dt),abstol=1e-6, reltol=1e-6, sensealg = ForwardDiffSensitivity() )
        X = Array(sol)
        return (X[:,end],0)
    end 
    
    
    return ContinuousModelErrors(dims,error_function,known_dynamics_function,IVP,derivs!,parameters,predict)
    
end 


mutable struct NeuralNetworkTimeDelays
    dims::Int # number of state variables
    lags::Int
    NN
    parameters #::ComponentArrasy # nerual network paramters
    predict::Function # neural network 
end 



function NeuralNetworkTimeDelays(dims,lags;hidden = 10, seed = 1)
    
    # initial neurla Network
    NN = Lux.Chain(Lux.Dense(dims+lags*dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN) 
    parameters = (NN = NN_parameters, aux0 = zeros(dims*lags))
    
    function predict(u,parameters) 
        return u .+ NN(vcat(u,parameters.aux0),parameters.NN,NN_states)[1], vcat(u,parameters.aux0[1:(end-dims)])
    end 
    
    function predict(u,aux,parameters) 
        return u .+ NN(vcat(u,aux),parameters.NN,NN_states)[1], vcat(u,aux[1:(end-dims)])
    end 
    
    return NeuralNetworkTimeDelays(dims,lags,NN,parameters,predict)
    
end 


mutable struct ModelErrorsTimeDelays
    dims::Int # number of state variables
    lags::Int
    errors
    known_dynamics
    parameters #::ComponentArrasy # nerual network paramters
    predict::Function # neural network 
end 



function ModelErrorsTimeDelays(dims,lags,known_dynamics,init_known_dynamics_parameters;hidden = 10, seed = 1)
    
    # initial neural Network
    NN = Lux.Chain(Lux.Dense(dims+lags*dims,hidden,tanh), Lux.Dense(hidden,dims))
    
    # parameters 
    Random.seed!(1)  # set seed for reproducibility 
    rng = Random.default_rng() 
    NN_parameters, NN_states = Lux.setup(rng,NN)     
    parameters = (known_dynamics = init_known_dynamics_parameters, 
                    NN = NN_parameters,aux0 = zeros(dims*lags))
    # prediction 
    error_function = (u,parameters) -> NN(u,parameters.NN,NN_states)[1]
    known_dynamics_function = (u,parameters) -> known_dynamics(u,parameters.known_dynamics)
    
    function predict(u,aux,parameters) 
        epsilon = error_function(vcat(u,aux),parameters)
        return known_dynamics_function(u,parameters) .+ epsilon, vcat(u,aux[1:(end-dims)]), epsilon
    end 
    
    function predict(u,parameters) 
        epsilon = error_function(vcat(u,parameters.aux0),parameters)
        return known_dynamics_function(u,parameters) .+ epsilon , vcat(u,parameters.aux0[1:(end-dims)]), epsilon 
    end 

    return ModelErrorsTimeDelays(dims,lags,error_function,known_dynamics_function,parameters,predict)
    
end 



mutable struct ModelErrorsLSTM
    dims::Int # number of state variables
    lags::Int
    error_function
    known_dynamics
    parameters #::ComponentArrasy # nerual network paramters
    #init_prediction
    predict
end 



function ModelErrorsLSTM(dims,known_dynamics,init_known_dynamics_parameters;cell_dim = 2, seed = 1)
    
    # initial neurla Network
    LSTM = Lux.LSTMCell(dims=>cell_dim)
    DenseLayer = Lux.Dense(cell_dim=>dims,tanh)
    
    # parameters 
    Random.seed!(seed)  # set seed for reproducibility 
    rng = Random.default_rng() 
    LSTM_parameters, LSTM_states = Lux.setup(rng,LSTM) 
    dense_parameters, dense_states = Lux.setup(rng, DenseLayer)
    rng = Random.default_rng()  
    parameters = (known_dynamics = init_known_dynamics_parameters, 
                    Dense = dense_parameters,LSTM = LSTM_parameters)
    
    known_dynamics_function = (u,parameters) -> known_dynamics(u,parameters.known_dynamics)
    
    function error_function(u,parameters)
        x = reshape(u,dims,1)
        (y, c), st_lstm = LSTM(x,parameters.LSTM, LSTM_states)
        error_term, states = DenseLayer(y,parameters.Dense,dense_states)
        return reshape(error_term,length(error_term)), (c, st_lstm)
    end 
    
    function error_function(u, aux, parameters)
        x = reshape(u,dims,1)
        c, st_lstm = aux
        (y, c), st_lstm = LSTM((x,c),parameters.LSTM, st_lstm)
        error_term, states = DenseLayer(y,parameters.Dense,dense_states)
        return reshape(error_term,length(error_term)), (c, st_lstm)
    end 
    
    
    function predict(u,parameters)
        error_term, (c,st_lstm) = error_function(u,parameters)
        known_term = known_dynamics_function(u,parameters)
        return known_term .+ error_term, (c, st_lstm), error_term
    end 
    
    function predict(u,aux,parameters) 
        error_term, (c, st_lstm) = error_function(u, aux, parameters)
        known_term = known_dynamics_function(u,parameters)
        return  known_term .+ error_term, (c, st_lstm), error_term
    end 

    return ModelErrorsLSTM(dims,cell_dim,error_function,known_dynamics_function,parameters,predict)
    
end 


end