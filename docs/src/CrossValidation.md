# Cross validation

The out-of-sample performance of UDE models can be estimated using leave future out cross-validation. Leave future out cross-validation creates testing data sets by leaving observation off of the end of the data set. The model is trained on the remaining data to predict the test sets. The difference between the models' forecasts at the training data provides an estimate of the model's performance on new data.

Our specific implementation constructs a use-defined number of testing data sets `k` by sequentially leaving one data point off of the end of the data set. The model is trained on the remaining data and used to forecast from that last time point in the training data set to the last time point in the training data set. If the data are sampled at regular intervals, This resutls in `k` estimates of the model's ahead forecasting skill `k-1` estimates of is forecasting skill two steps and more generally `k-h+1` estimates of its forecasting skill `h` time step into the future.

The cross-validation function will return a data frame with the mean absolute error of the forecasts and the associated standard error at each forecast horizon `h<k`. The function will also return the raw values in the testing data set and the values predicted by the model. These rate data can be used to calculate other statistics, like the mean squared error.

The leave future out cross validation function will work if there are missing data or the data set is sampled at irregular intervals, but this will affect the number of times the algorithm can test each forecast horizon `h`. In these cases, we recommend calculating your own model performance statistics using the raw data returned by the algorithm rather than relying on the summary statistics.

The `leave_future_out` cross-validation function takes a model object, a training routine, and the number of tests `k` as arguments. The model object is either a `MultiUDE` or `UDE` object. The training routine `training!` is a function that takes the model as an argument and updates its parameters in place. In the simplest case, the `training!` function could just be a wrapper for `UniversalDiffEq.train`.

```julia
training! = (model) -> train!(model,loss_function = "derivative matching")
```

The taining routine function can also be used to test a combination of methods.

```julia
function training!(model)
   UniversalDiffEq.train!(model,loss_function = "derivative matching",
                                   optim_options = (maxiter = 500, ),
                                   loss_options = (observation_error = 0.25,))
   UniversalDiffEq.train!(model,loss_function = "marginal likeihood",
                                   optim_options = (maxiter = 100, ),
                                   loss_options = (observation_error = 0.25,))
end
```

The function will return a data set with the high level summary statistics from the cross validaiton tests this data frame has three columns the time horizon, the mean absolute error the forecasts (MAE) and the standartd error of the MAE estimates. 

| horizon | mean_absolute_error | standard_error_of_MAE |
|---------|---------------------|-----------------------|
|1        |MAE1                 | SE1                   |
|2        |MAE2                 | SE2                   |
|...      |...                  | ...                   |

In addtion to the sumamery data frame the function will return a named tuple with additonal data frames. These include the raw data from the leave future out cross validaiton algorithm and more detailed summary statistics. The raw data report the iteration of the cross validation algorithm `fold`, the time of the testing observation in the original data set `time` , the time horizon of the forecast `horizon`, the value of the testing data point `testing`  and the value of the forecast `forecast`. If the algorithm was run on a model with multiple time series there will also be a column indicating which time sereis the testing data came from `series`


| fold | time  | horizon |varaible| testing|forecast|
|------|-------|---------|--------|--------|--------|
|1     |50     | 1       | x      | 0.1    | 0.12   |
|2     |50     | 2       | x      | 0.1    | 0.05   |
|2     |49     | 1       | x      | 0.251  | 0.201  |
|3     |50     | 3       | x      | 0.1    | -0.012 |
|3     |49     | 2       | x      | 0.251  | 0.12   |
|3     |48     | 1       | x      | 0.184  | 0.191  |
|...   |...    | ...     | ...    | ...    | ...    |


The other data sets report the mean absolute error and assocaited statnard errors grouping the data in differnt ways. For example, the `horizon_by_var` data set reports the mean absolute error for each variable independently. 


## doc string
```@docs; canonical=false
UniversalDiffEq.leave_future_out(model::UDE, training!, k; kwargs... )
```

## Minimal example

```julia
using UniversalDiffEq, Random, DataFrames

data = DataFrame(time = 1:40,  x = rand(40), y = rand(40), z = rand(40))

NN,NNparams = UniversalDiffEq.SimpleNeuralNetwork(3,3)
function derivs!(u,i,p,t)
   zeros(3)
end

init_parameters = (NN = NNparams,)

model = UniversalDiffEq.CustomDerivatives(training,derivs!,init_parameters)

function training_routine_4!(model)
   UniversalDiffEq.train!(model,loss_function = "conditional likelihood",
                                   optim_options = (maxiter = 2, ),
                                   loss_options = (observation_error = 0.25,))
end

cv_summary, cv_details = UniversalDiffEq.leave_future_out(model, training_routine_4!, 10)
```
