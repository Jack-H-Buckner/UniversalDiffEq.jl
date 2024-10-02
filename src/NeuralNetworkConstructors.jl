mutable struct ARD
    inputs
    outputs
    hidden
    NN
    states
end 

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