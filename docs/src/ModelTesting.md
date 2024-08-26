# Test models

UniversalDiffEq.jl provides a number of functions to test the performance of NODE and UDE models, with both in-sample and out-of-sample data. 

## Evaluating model fits

There are two primary functions to test model fits: `plot_state_estimates` and `plot_predictions`. The model fitting procedure estimates the value of the state variables ``\hat{u}`` at each time point in the data set as well as the parameters of the NODE or UDE model that predicts changes in the state variables. The `plot_state_estimates` function compares the estimated states to the data to check the quality of the state estimates, and `plot_predictions` compares the predictions of UDE model one step into the future to the estimated sequence of state variables. Both functions take a UDE as an input and return a plot showing the correspondence between model predictions and observations. 

```@docs; canonical=false
UniversalDiffEq.plot_state_estimates(UDE::UDE)
UniversalDiffEq.plot_predictions(UDE::UDE)
```

There are also functions to compare the model predictions to out-of-sample data. The simplest is `plot_forecast`, which compares the observations in the test data set to a deterministic simulation from the data set, which starts at the first observation and runs to the end of the test data. 

```@docs; canonical=false
UniversalDiffEq.plot_forecast(UDE::UDE, test_data::DataFrame)
```

It is also possible to test the performance of the models one time step into the future using the `plot_predictions` function. When a test data set is supplied to the `plot_predictions` function, it will run a series of forecasts starting at each point in the data set, predicting one time step into the future. The function returns a plot comparing the predicted and observed changes.

```@docs; canonical=false
UniversalDiffEq.plot_predictions(UDE::UDE, test_data::DataFrame)
```

## Cross-validation

Cross-validation is important for model comparison and hyper-parameter tuning. The `leave_future_out_cv` function breaks the data set into training and test data sets by leaving off the final observations in the data set. The model is then trained on the beginning of the data set and the performance is calculated by comparing a forecast to the test data. The user can specify the time horizon for the forecast ``T_{Forecast}`` and the number of tests ``K``. The first test trains the model on the full data set only omitting the final ``T_{forecast}`` years as the test set. The remaining tests each generate a new test data set by iteratively removing more of the observations from the end of the data set. The number removed between each test can be controlled by changing the spacing argument. The `kfold_cv` function performs a k-fold version of `leave_future_out_cv`.


```@docs; canonical=false
UniversalDiffEq.leave_future_out_cv(model::UDE)
UniversalDiffEq.kfold_cv(model::UDE)
```
