# Model analysis 

UniversalDiffEq provides several functions to analyze the characteristics of the fitted models. The most basic of these is the `get_right_hand_side` funciton. This function takes a `UDE model as an argument and returns the right hand side of the fitted differntial or differnce equation. This function can then be treated like any dynamic model and analyzed for equilibria, stability, tippinging pionts and other phenomina of interest.  

```@docs
UniversalDiffEq.get_right_hand_side(UDE::UDE)
```

The library also has functions to evaluate model predictions. The `forecast` function will run a simulation of the model starting at the initial point `u0` and returning the value of the state variables at each point in the `times` vector.  

```@docs
UniversalDiffEq.forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})
```



```@docs
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
UniversalDiffEq.get_NN_parameters(UDE::UDE)
UniversalDiffEq.get_parameters(UDE::UDE)
```