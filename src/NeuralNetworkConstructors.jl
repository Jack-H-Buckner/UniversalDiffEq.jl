<<<<<<< HEAD
mutable struct ARD
=======
mutable struct SimpleNeuralNetwork
>>>>>>> main
    inputs
    outputs
    hidden
    NN
    states
end 

<<<<<<< HEAD
function soft_plus(x)
    x/(1-exp(-5*x))
  end 

function ARD(inputs,outputs;hidden = 20, nonlinearity = soft_plus)


    NN_1 = Lux.Chain(Lux.Dense(inputs,hidden,nonlinearity),Lux.Dense(hidden,1)) 
    rng = Random.default_rng() 
    parameters_1, states_1 = Lux.setup(rng,NN_1) 
    nets = (NN_1 = NN_1, ); parameters = (NN_1 = parameters_1, ); states = [states_1]


    for i in 2:outputs
        NN_i = Lux.Chain(Lux.Dense(inputs,hidden,nonlinearity),Lux.Dense(hidden,1)) 
        rng = Random.default_rng() 
        parameters_i, states_i = Lux.setup(rng,NN_i) 
        pars_dict = Dict(string("NN_", i) => parameters_i,)
        pars_tuple = (; (Symbol(k) => v for (k,v) in pars_dict)...)
        parameters  = merge(parameters , pars_tuple)

        NN_dict = Dict(string("NN_", i) => NN_i,)
        NN_tuple = (; (Symbol(k) => v for (k,v) in NN_dict)...)
        nets  = merge(nets , NN_tuple)
        push!(states, states_i)
    end

    parameters = (NN = parameters, α = zeros(outputs) .+ 1.0, scale = zeros(inputs,outputs) .+ 1.0)
    return ARD(inputs, outputs, hidden, nets, states), parameters
end 

function (model::ARD)(x,parameters)
    f = d -> abs(parameters.α[d])*model.NN[Symbol(string("NN_",d))](parameters.scale[:,d] .* x,parameters.NN[Symbol(string("NN_",d))],model.states[d])[1][1]
    broadcast(f, 1:model.outputs)
end 


mutable struct feedforward
    inputs
    outputs
    hidden
    NN
    states
end 


function feedforward(inputs,outputs; hidden = 20, nonlinearity = tanh)
    NN = Lux.Chain(Lux.Dense(inputs,hidden,nonlinearity),Lux.Dense(hidden,outputs))
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    return feedforward(inputs,outputs,hidden,NN,states), parameters
end 

function (model::feedforward)(x,parameters)
    model.NN(x,parameters.NN,model.states)[1]
end 
=======
"""
    SimpleNeuralNetwork(inputs,outputs; kwargs ...)

Builds a neural network object and returns randomly initialized parameter values. The neural network object can be evaluated like a function it takes two arguments: a vector  `x` and named tuple with the neural network weights and biases `parameters`.  

    # kwargs 

- `hidden`: the number of neurons in the hidden layer. The default is 10
- `nonlinearity`: the activation funciton used in the neurla network. The default is the hyperbolic tangent function `tanh`.
- `seed` : random number generator seed for initializing the neural network weights
"""
function SimpleNeuralNetwork(inputs,outputs; hidden = 10, nonlinearity = tanh, seed = 123)
    NN = Lux.Chain(Lux.Dense(inputs,hidden,nonlinearity),Lux.Dense(hidden,outputs))
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    return SimpleNeuralNetwork(inputs,outputs,hidden,NN,states), parameters
end 

function (model::SimpleNeuralNetwork)(x,parameters)
    model.NN(x,parameters,model.states)[1]
end 






mutable struct MultiTimeSeriesNetwork
    inputs
    series
    outputs
    hidden
    NN
    states
    distance
    one_hot
end 

"""
    MultiTimeSeriesNetwork(inputs,series,outputs; kwargs...)

Builds a neural network object and returns randomly initialized parameter values. The neural network object can be evaluated like a function it takes three  arguments: a vector  `x`, an integer `i` and named tuple with the neural network weights and biases `parameters`.  

    # kwargs 

- `hidden`: the number of neurons in the hidden layer. The default is 10
- `nonlinearity`: the activation funciton used in the neurla network. The default is the hyperbolic tangent function `tanh`.
- `seed` : random number generator seed for initializing the neural network weights
- `distance` : deterines the level of differnce between the functions estimated for each time series. the defualt is 1.0.
"""
function MultiTimeSeriesNetwork(inputs,series,outputs; hidden = 10, nonlinearity = tanh, seed = 123, distance = 1.0)
    NN = Lux.Chain(Lux.Dense(inputs+series,hidden,nonlinearity),Lux.Dense(hidden,outputs))
    rng = Random.default_rng() 
    parameters, states = Lux.setup(rng,NN) 
    one_hot = zeros(series)
    return MultiTimeSeriesNetwork(inputs,series,outputs,hidden,NN,states,distance,one_hot), parameters
end 

function (model::MultiTimeSeriesNetwork)(x,i,parameters)
    if i == "average"
        model.one_hot .= model.distance/sqrt(model.series) 
        inputs = vcat(x,model.one_hot)
        out = model.NN(inputs,parameters,model.states)[1]
        model.one_hot .= 0
        return out
    end 
    model.one_hot[round(Int,i)] = model.distance 
    inputs = vcat(x,model.one_hot)
    out = model.NN(inputs,parameters,model.states)[1]
    model.one_hot[round(Int,i)] = 0
    return out
end 


>>>>>>> main
