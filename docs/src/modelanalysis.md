# Model analysis 

UniversalDiffEq provides several fuctions to analyze the characteristics of the fitted models. 

```@docs
UniversalDiffEq.get_right_hand_side(UDE::UDE)
```


```@docs
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
UniversalDiffEq.forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
UniversalDiffEq.leave_future_out_cv(model; forecast_length = 10,  K = 10, spacing = 1, step_size = 0.05, maxiter = 500)
UniversalDiffEq.get_NN_parameters(UDE::UDE)
UniversalDiffEq.predictions(UDE::UDE,test_data::DataFrame)
UniversalDiffEq.get_parameters(UDE::UDE)
```