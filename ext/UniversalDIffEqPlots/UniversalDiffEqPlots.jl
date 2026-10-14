module UniversalDiffEqPlots

using UniversalDiffEq, Plots

include("ModelPlots.jl")
include("MultiModelPlots.jl")
include("ODEAnalysisPlots.jl")
include("SimulatedDataPlots.jl")

#
export plot_predictions, plot_state_estimates, plot_forecast, phase_plane, phase_plane_3d, observation_error_correlations, vectorfield_and_nullclines, plot_bifurcation_diagram, LotkaVolterra, LorenzLotkaVolterra, StochasticLotkaVolterra, LogisticLorenz, LogisticMap

end