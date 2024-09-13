# Model analysis

UniversalDiffEq.jl provides several functions to analyze the characteristics of the fitted models. The most basic of these is the `get_right_hand_side` function. This function takes a UDE model as an argument and returns the right-hand side of the fitted differential or difference equation. This function can then be treated like any dynamic model and analyzed for equilibria, stability, tipping points, and other phenomena of interest.

```@docs; canonical=false
UniversalDiffEq.get_right_hand_side(UDE::UDE)
```

The function `phase_plane` plots forecasted trajectories of state variables for a given number of timesteps `T`. All phase plane functions also work with the `MultiUDE` model type, and plot phase planes for each series in the data.

```@docs; canonical=false
UniversalDiffEq.phase_plane(UDE::UDE; idx=[1,2], u1s=-5:0.25:5, u2s=-5:0.25:5, T = 100)
UniversalDiffEq.phase_plane(UDE::UDE, u0s::AbstractArray; idx=[1,2],T = 100)
UniversalDiffEq.phase_plane_3d(UDE::UDE; idx=[1,2,3], u1s=-5:0.25:5, u2s=-5:0.25:5, u3s=-5:0.25:5, T = 100)
```

The library also has functions to evaluate model predictions. The `forecast` function will run a simulation of the model starting at the initial point `u0` and returning the value of the state variables at each point in the `times` vector. The results of these forecasts can be displayed using the `plot_forecast` function.

```@docs; canonical=false
UniversalDiffEq.forecast(UDE::UDE, u0::AbstractVector, times::AbstractVector)
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
```

Finally, the parameters of the process model, the weights and biases of the neural network, and the estimated parameter values of the known dynamics can be displayed using the functions `get_parameters`, `get_NN_parameters`, and `print_parameter_estimates`, respectively.

```@docs; canonical=false
UniversalDiffEq.get_parameters(UDE::UDE)
UniversalDiffEq.get_NN_parameters(UDE::UDE)
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
```
