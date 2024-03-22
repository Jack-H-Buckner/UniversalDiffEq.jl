# Model analysis 

UniversalDiffEq provides several functions to analyze the characteristics of the fitted models. The most basic of these is the `get_right_hand_side` funciton. This function takes a `UDE model as an arguemtn and returns the right hand side of the fitted differntial or differnce equation. This function cna then be treated like any dynamical model and analyzed for equilibria, stability, tippinging pionts and other phenominoa of interest.  

```@docs
UniversalDiffEq.get_right_hand_side(UDE::UDE)
```


```@docs
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
UniversalDiffEq.forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
UniversalDiffEq.get_NN_parameters(UDE::UDE)
UniversalDiffEq.predictions(UDE::UDE,test_data::DataFrame)
UniversalDiffEq.get_parameters(UDE::UDE)
```