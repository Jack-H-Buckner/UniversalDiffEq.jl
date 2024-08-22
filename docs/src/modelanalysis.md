# Model analysis 

UniversalDiffEq.jl provides several functions to analyze the characteristics of the fitted models. The most basic of these is the `get_right_hand_side` function. This function takes a UDE model as an argument and returns the right-hand side of the fitted differential or difference equation. This function can then be treated like any dynamic model and analyzed for equilibria, stability, tipping points, and other phenomena of interest.  

```@docs
UniversalDiffEq.get_right_hand_side(UDE::UDE)
```

The library also has functions to evaluate model predictions. The `forecast` function will run a simulation of the model starting at the initial point `u0` and returning the value of the state variables at each point in the `times` vector.  


The function `phase_plane` plots forecasted trajectories of state variables for a given number of timesteps `T`. All phase plane functions also work with the `MultiUDE` model type, and plot phase planes for each series in the data.
```@docs
UniversalDiffEq.phase_plane(UDE::UDE; idx=[1,2], u1s=-5.0,0.25,5.0, u2s=-5:0.25:5,u3s = 0,T = 100)
UniversalDiffEq.phase_plane(UDE::UDE, u0s::AbstractArray; idx=[1,2],T = 100)
UniversalDiffEq.phase_plane_3d(UDE::UDE; idx=[1,2,3], u1s=-5.0,0.25,5.0, u2s=-5:0.25:5,u3s=-5:0.25:5,T = 100)
```

```@docs
UniversalDiffEq.forecast(UDE::UDE, u0::AbstractVector, times::AbstractVector)
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
UniversalDiffEq.get_NN_parameters(UDE::UDE)
UniversalDiffEq.get_parameters(UDE::UDE)
```
