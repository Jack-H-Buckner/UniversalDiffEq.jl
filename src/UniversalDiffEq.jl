module UniversalDiffEq

using DiffEqFlux, OrdinaryDiffEq, StochasticDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL, ComponentArrays, Lux, Random, StatsBase, DelimitedFiles, Plots, DataFrames, Distributions, Roots, LaTeXStrings, NLsolve, FiniteDiff, LinearAlgebra, AdvancedHMC, MCMCChains, Interpolations, Optim, CategoricalArrays, Zygote, RegularizationTools, DataInterpolations, JSON


include("covariates.jl")
include("SimulateData.jl")
include("ObservationModels.jl")
include("ProcessModels.jl")
include("LossFunctions.jl")
include("Regularization.jl")
include("Models.jl")
include("MultiProcessModel.jl")
include("MultipleTimeSeries.jl")
include("ModelTesting.jl")
include("ODEAnalysis.jl")
include("MultiModelTests.jl")
include("helpers.jl")
include("Optimizers.jl")
include("CrossValidation.jl")
include("NeuralNetworkConstructors.jl")
include("UnscentedKalmanFilter.jl")
include("LossFunctionConstructors.jl")


export UDE, MultiUDE, CustomDerivatives, CustomModel, NNDE, NODE, MultiNODE, MultiCustomDerivatives ,train!, plot_predictions, plot_forecast, leave_future_out, vectorfield_and_nullclines, get_right_hand_side, CustomDifference, plot_state_estimates, LotkaVolterra, LorenzLotkaVolterra, gradient_descent!, equilibrium_and_stability, EasyNODE, EasyUDE, BayesianNODE, BayesianUDE, NUTS!, SGLD!, predict, get_parameters, plot_bifurcation_diagram, bifurcation_data, cross_validation_kfold, simulate_coral_data, SimpleNeuralNetwork, save_model, load_model

end
