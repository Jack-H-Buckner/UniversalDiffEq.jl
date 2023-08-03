include("NODEModels.jl")

function denseLayersLux(inputSize,hiddenSize;functions=nothing)
    nn = Lux.Chain(Lux.Dense(inputSize,hiddenSize[1],functions[1]),
                    Lux.Dense(hiddenSize[1],inputSize))
    return nn
end

function createModel(path::String;knownDynamics = nothing,hiddenLayerSize = 0,modelType = :NODE,solver = nothing,
    tol = 1e-6,
    hasTime = false,
    lossFunction = :MAE,
    learningRate = 0.005,
    maxIters = 300,
    neededParameters = 0,
    givenParameters = Float64[])

    trainingData = readdlm(path)
    #Determine we have more observations than inputs
    if size(trainingData,1) > size(trainingData,2)
        trainingData = permutedims(trainingData)
    end
    
    #If string headers are there, remove them
    if typeof(trainingData)==Matrix{Any}
        trainingData = convert(Array{Float64},trainingData[:,2:end])
    end
    
    return createModel(trainingData,knownDynamics=knownDynamics,hiddenLayerSize = hiddenLayerSize,modelType = modelType,solver = solver,
    tol = tol,
    hasTime = hasTime,
    lossFunction = lossFunction,
    learningRate = learningRate,
    maxIters = maxIters,
    neededParameters = neededParameters,
    givenParameters = givenParameters)
end

function createModel(trainingData::Matrix{Float64};knownDynamics = nothing,hiddenLayerSize = 0,modelType = :NODE,solver = nothing,
    tol = 1e-6,
    hasTime = false,
    lossFunction = :MAE,
    learningRate = 0.005,
    maxIters = 300,
    neededParameters = 0,
    givenParameters = Float64[])
    #Check provided model type is valid
    if !(modelType in (:NODE,:UDE))
        throw(ArgumentError("Provided model type is not currently supported. Currently supported model types are Neural Ordinary Differential Equations (:NODE) or Universal Differential Equations (:UDE)"))
    end
    #Check provided loss function is valid
    if !(lossFunction in (:MAE,:RMSE))
        throw(ArgumentError("Invalid error loss function. Available options are Maximum Absolute Error (:MAE) or Root Mean Squared Error (:RMSE)."))
    end

    #If known dynamics are provided, test them first
    if !isnothing(knownDynamics)
        try
            p = rand(neededParameters)
            knownDynamics(trainingData[:,1],p,givenParameters)
        catch
            throw(ArgumentError("Provided dynamics do not match required format. Is your function of the form f(x,p,q)? Did you provide enough parameters to the function?"))
        end
        #If known dynamics are provided, it's a UDE
        modelType == :UDE
    end
    
    #If time is provided, choose first row as time 
    if hasTime
        timeSpans = trainingData[1,:]
        trainingData = trainingData[2:end,:]
    else
        timeSpans = 1.:1:size(trainingData,2)
    end
    #If solver is not provided, choose an appropiate automatic solver
    solver = isnothing(solver) ? (tol > 1e-8 ? AutoTsit5(Rodas4P()) : AutoAutoVern9(RadauIIA5())) : solver

    #If hidden layer size is not provided, give a default
    hiddenLayerSize = hiddenLayerSize == 0 ? 2*minimum(size(trainingData)) : hiddenLayerSize

    #Create neural network
    if minimum(trainingData) < 0
        neuralNetwork = denseLayersLux(size(trainingData,1),hiddenLayerSize;functions=[tanh])
    else
        neuralNetwork = denseLayersLux(size(trainingData,1),hiddenLayerSize;functions=[relu])
    end

    #Create model
    untrainedModel = (modelType == :NODE) ? NODE(neuralNetwork,nothing) :
     modelType == :UDE ? UDE(neuralNetwork,nothing,knownDynamics,neededParameters,givenParameters) :
     nothing

    #Train model
    trainedModel, trainedFit = train(untrainedModel,trainingData,timeSpans,solver = solver,
    tol = tol,
    lossFunction = lossFunction,
    learningRate = learningRate,
    maxIters = maxIters)
    return trainedModel, trainedFit
end