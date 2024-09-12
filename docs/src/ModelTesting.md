# Test models

UniversalDiffEq.jl provides a number of functions to test the performance of NODE and UDE models on in sample data. These test validate the model fitting procedure. Functions to estiamte the model's performance on out of sample data are discussed in the section on cross validation.  

## Evaluating model fits

There are two primary functions to evaluate model fits: `plot_state_estimates` and `plot_predictions`. The training procedure used by UniversalDiffEq.jl simutaniously smooths the training data and trains trains the paramters of the UDE model on the smoothed data set. The funciton `plot_state_estimates` compares the smoothed time series produced by the training procedure to the observations in the data set. The smoothed time series (grey line) needs to capture the main trends in trainig data (blue dots) for the model to accurately recover the dynamics in the data set. 

```@docs canonical=false
UniversalDiffEq.plot_state_estimates(UDE::UDE)
```

We can make this analysis a bit more rigorous by looking for correlations in the obervation errors using the `observation_error_correlations` function. This creates a lag plot for each pair of variables in the model and calcualtes the correlation coeficent. Large correlations in the observation erros suggest the UDe model may be missing predictable variaiton in the data set. 

```@docs canonical=false
UniversalDiffEq.observation_error_correlations(UDE;fig_size = (600,500))
```

The `plot_predictions`  functions compares the predictions of UDE model one step into the future to the estimated sequence of state variables. This function quantifies the in sample predictive accuracy of the model. 

```@docs; canonical=false
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


