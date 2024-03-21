# Test models

UniversalDiffEq.jl provides a number of functions to test the performance of NODE and UDE models, with both in sample and out of sample data. 

## Evaluating model fits

There are two primary functions to test model fits `plot_state_estimates` and `plot_predictions`. The model fitting procedure estimates the value of the state variables ``\hat{u}`` at each time point in the data set and the parameters of the NODE or UDE model that predicts chagnes in the state variables. The `plot_state_estimates` function compares the esitmated states to the data to check the quality of the state estimates and `plot_predictions` compares the predictions of UDE model one step into the future to estiamted sequence of statate variables. Both functions take a UDE as an input and return a plot showing the correspondence between model predictions and observations. 

```@docs
UniversalDiffEq.plot_state_estiamtes(UDE::UDE)
UniversalDiffEq.plot_predictions(UDE::UDE)
```

In addition to 

```@docs
UniversalDiffEq.print_parameter_estimates(UDE::UDE)
UniversalDiffEq.plot_predictions(UDE::UDE, test_data::DataFrame)
UniversalDiffEq.forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})
UniversalDiffEq.plot_forecast(UDE::UDE, T::Int)
UniversalDiffEq.plot_forecast(UDE::UDE, test_data::DataFrame)
UniversalDiffEq.leave_future_out_cv(model; forecast_length = 10,  K = 10, spacing = 1, step_size = 0.05, maxiter = 500)
UniversalDiffEq.get_NN_parameters(UDE::UDE)
UniversalDiffEq.get_right_hand_side(UDE::UDE)
UniversalDiffEq.predictions(UDE::UDE,test_data::DataFrame)
UniversalDiffEq.get_parameters(UDE::UDE)
```