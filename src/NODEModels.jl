"""
General purpose functions go here
"""
function loadNeuralNetwork(fileName::String)
    return deserialize(fileName)
end

"""
Neural Ordinary Differential Equations (NODEs) are the deterministic core of the tools implemented here
"""

struct NODE{M,P}
    neuralNetwork::M
    parameters::P
end

function train(model::NODE,trainingData::Matrix{Float64},timeSpans = nothing;
    solver = nothing,
    tol = 1e-6,
    lossFunction = :MAE,
    learningRate = 0.005,
    maxIters = 300)
    rng = Random.default_rng()

    ps, st = Lux.setup(rng, model.neuralNetwork)
    ps64 = Float64.(ComponentArray(ps))
    neuralode = NeuralODE(model.neuralNetwork, (timeSpans[1],timeSpans[end]), solver,saveat=timeSpans,reltol=tol,abstol=tol)

    #Determine loss function from a library
    function predict(p)
        Array(neuralode(trainingData[:,1], p,st)[1])
    end
    if lossFunction == :MSE
        function loss(p)
            pred = predict(p)
            ℓ = sum(abs, trainingData .- pred)
            return ℓ, pred
        end
    elseif lossFunction == :RMSE
        function loss(p)
            pred = predict(p)
            ℓ = sum(abs2, trainingData .- pred)
            return ℓ, pred
        end
    end

    #Train
    pinit = ComponentVector(ps64)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> lossFunction(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)

    optimizedParameters = Optimization.solve(optprob,
                                           ADAM(learningRate),
                                           maxiters = maxIters)
    return NODE(model.neuralNetwork,optimizedParameters)
end

function test(model::NODE,x0::Vector{Float64},T::Float64;
    solver = nothing,
    tol = 1e-6)
#If solver is not provided, choose an appropiate automatic solver
solver = isnothing(solver) ? (tol > 1e-8 ? AutoTsit5(Rodas4P()) : AutoAutoVern9(RadauIIA5())) : solver

    p, st = Lux.setup(rng,model.neuralNetwork)
    neuralode = NeuralODE(neuralNetwork,(0.,T),solver,saveat=1)
    prediction = Array(neuralode(x0,params,st)[1])
    return prediction
end

function saveNeuralNetwork(model::NODE,fileName = "fit_neural_network")
    serialize(fileName*".jls",model)
end

"""
Universal Differential Equations (UDEs) expand on NODEs to include known dynamics as an additional input
"""

struct UDE{M,P,F}
    neuralNetwork::M
    parameters::P
    knownDynamics::F
    neededParameters::Int
    givenParameters::Vector{Float32}
end

function train(model::UDE,trainingData::Matrix{Float64},timeSpans = nothing;
    solver = nothing,
    tol = 1e-6,
    lossFunction = :MAE,
    learningRate = 0.005,
    maxIters = 300)
    rng = Random.default_rng()
    ps, st = Lux.setup(rng, model.neuralNetwork)
    ps64 = Float64.(ComponentArray(ps))
    psDynamics = ComponentArray((predefined_params = rand(Float64, model.neededParameters), model_params = ps64))

    function ude!(du,u,p,t,q)
        knownPred = model.knownDynamics(u,p.predefined_params,q)
        nnPred = Array(model.neuralNetwork(u,p.model_params,st)[1])

        for i in 1:length(u)
            du[i] = knownPred[i]+nnPred[i]
        end
    end

    # Closure with the known parameter
    nn_dynamics!(du,u,p,t) = ude!(du,u,p,t,model.givenParameters)
    prob_nn = ODEProblem(nn_dynamics!,trainingData[:, 1], (timeSpans[1],timeSpans[end]), psDynamics)

    #Determine loss function from a library
    function predict(p)
        _prob = remake(prob_nn, u0 = trainingData[:, 1], tspan = (timeSpans[1], timeSpans[end]), p = p)
        Array(solve(_prob,solver,saveat = timeSpans,
            abstol = tol, reltol = tol))
    end
    if lossFunction == :MSE
        function loss(p)
            pred = predict(p)
            ℓ = sum(abs, trainingData .- pred)
            return ℓ, pred
        end
    elseif lossFunction == :RMSE
        function loss(p)
            pred = predict(p)
            ℓ = sum(abs2, trainingData .- pred)
            return ℓ, pred
        end
    end

    #Train
    pinit = ComponentVector(ps_dynamics)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction((x, p) -> lossFunction(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, pinit)

    optimizedParameters = Optimization.solve(optprob,
                                           ADAM(learningRate),
                                           maxiters = maxIters)
    return UDE(model.neuralNetwork,optimizedParameters,model.knownDynamics,model.neededParameters,model.givenParameters)
end

function test(model::UDE,x0::Vector{Float64},T::Float64;
    solver = nothing,
    tol = 1e-6)    
    #If solver is not provided, choose an appropiate automatic solver
    solver = isnothing(solver) ? (tol > 1e-8 ? AutoTsit5(Rodas4P()) : AutoAutoVern9(RadauIIA5())) : solver
    
    ps, st = Lux.setup(rng, model.neuralNetwork)

    function ude!(du,u,p,t,q)
        knownPred = model.knownDynamics(u,p.predefined_params,q)
        nnPred = Array(model.neuralNetwork(u,p.model_params,st)[1])

        for i in 1:length(u)
            du[i] = knownPred[i]+nnPred[i]
        end
    end
    # Closure with the known parameter
    nn_dynamics!(du,u,p,t) = ude!(du,u,p,t,model.givenParameters)
    # Define the problem
    prob_nn = ODEProblem(nn_dynamics!,x0, (0.,T), model.parameters)
    prediction = Array(solve(prob_nn, solver, saveat = 1,
                abstol=tol, reltol=tol
                ))
    return prediction
end

function saveNeuralNetwork(model::UDE,fileName = "fit_neural_network")
    methodToSave = methods(model.knownDynamics)[1]
    modelToSave = UDE(model.neuralNetwork,model.parameters,methodToSave,model.neededParameters,model.givenParameters)
    serialize(fileName*".jls",modelToSave)
end