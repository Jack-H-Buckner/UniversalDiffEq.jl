# Test models

```@docs
UniversalDiffEq.plot_state_estiamtes(UDE::UDE)
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
UniversalDiffEq.plot_predictions(UDE::UDE)
UniversalDiffEq.plot_predictions(UDE::UDE, test_data::DataFrame)
UniversalDiffEq.forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
UniversalDiffEq.plot_forecast(UDE::UDE, test_data::DataFrame)
UniversalDiffEq.leave_future_out_cv(model; forecast_length = 10,  K = 10, spacing = 1, step_size = 0.05, maxiter = 500)
get_NN_parameters(UDE::UDE)
get_right_hand_side(UDE::UDE)
predictions(UDE::UDE,test_data::DataFrame)
get_parameters(UDE::UDE)
```