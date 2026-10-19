module UniversalDiffEqPlots

using UniversalDiffEq, Plots, DiffEqFlux, OrdinaryDiffEq, StochasticDiffEq, Optimization, OptimizationOptimisers, OptimizationOptimJL, ComponentArrays, Lux, Random, StatsBase, DelimitedFiles, DataFrames, Distributions, Roots, LaTeXStrings, NLsolve, FiniteDiff, LinearAlgebra, AdvancedHMC, MCMCChains, Interpolations, Optim, CategoricalArrays, Zygote, RegularizationTools, DataInterpolations, JSON, CSV


include("ModelPlots.jl")
include("MultiModelPlots.jl")
include("ODEAnalysisPlots.jl")
include("SimulatedDataPlots.jl")

#
export plot_predictions, plot_state_estimates, plot_forecast, phase_plane, phase_plane_3d, observation_error_correlations, vectorfield_and_nullclines, plot_bifurcation_diagram, PlotLotkaVolterra, PlotLorenzLotkaVolterra, PlotStochasticLotkaVolterra, PlotLogisticLorenz, PlotLogisticMap

end