# Fitting a model to multiple time series

UniversalDiffEq.jl provides a set of functions to fit models to multiple time series. The functions for this mirror the functions for fitting NODEs and UDEs to single time series with the prefix `Multi`. For example, to build a NODE model for multiple time series you would use the `MultiNODE` function. The functions for model fitting, testing, and visualization have the same names. The other important difference is the data format: a column with a unique index for each time series must be included.

**Table**: Example data frame with multiple time series

|`time`|`series`|``x_1``|``x_2``|
|---|--------|---------------|----------------|
|1  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|2  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|3  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|1  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |
|2  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |
|3  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |

```@docs; canonical=false
UniversalDiffEq.MultiNODE(data;kwargs...)
UniversalDiffEq.MultiNODE(data,X;kwargs...)
```

## Multiple time series custom models

Custom models can be trained on multiple time series using the `MultiCustomDerivatives` function. The user-defined function that builds these models requires an additional argument, `i` added as the third argument  (e.g., `derivs!(du,u,i,X,p,t)`, `derivs!(du,u,i,p,t)`). The UniversalDiffEq.jl library will use this argument to pass a unique index for each time series. These indices can then be used to estimate different parameter values for each time series as illustrated in the following examples.

```@docs; canonical=false
UniversalDiffEq.MultiCustomDerivatives(data,derivs!,initial_parameters;kwargs...)
```
### Example 1: Estimating unique growth rates for population time series

To illustrate how unique parameters can be estimated for separate time series, we build a generalized logistic model that uses a neural network to describe the density dependence of the populations and estimates unique growth rates for each time series
```math
\frac{du_i}{dt} = r_i u_i NN(u_i).
```

To build this model, we use the argument `i` of the `derivs!` function to index it into a vector of growth rate parameters. Notice that we need to transform `i` to be an integer for the indexing operation by calling `round(Int,i)` and we initialize the growth rate parameter `r`.

```julia
# set up neural network
using Lux
dims_in = 1
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize parameters
using Random
rng = Random.default_rng()
NNparameters, NNstates = Lux.setup(rng,NN)

function derivs!(du,u,i,p,t)
    index = round(Int,i)
    du .= p.r[index].* u .* NN(u,p.NN, NNstates)[1]

end

m = 2 # number of time series
init_parameters = (NN = NNparameters, r = zeros(m), )

model = MultiCustomDerivatives(training_data,derivs!,init_parameters;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
```

### Example 2: Allowing a neural network to vary between time series

In some cases, we may want to allow the neural networks to vary between time series so that they can estimate unknown functions specific to each time series. This can be achieved by passing an indicator variable to the neural network that encodes the time series being fit using [one-hot encoding](https://www.geeksforgeeks.org/ml-one-hot-encoding/). This method allows the model to learn unique functions for each time series if appropriate, while also sharing information about the unknown function between time series.

To illustrate this, define a NODE model that takes the indicator variable as an additional argument.


```julia
# set up neural network
m = 10 # number of time series
using Lux
dims_in = 2+m #two inputs for the state variable plus m inputs for the one-hot encoding
hidden_units = 10
nonlinearity = tanh
dims_out = 2
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize parameters
using Random
rng = Random.default_rng()
NNparameters, NNstates = Lux.setup(rng,NN)

function derivs!(du,u,i,X,p,t)
    index = round(Int,i)
    one_hot = zeros(m)
    one_hot[index] = 1
    du .=  NN(vcat(u,one_hot) ,p.NN, NNstates)[1]
end

init_parameters = (NN = NNparameters, )

model = MultiCustomDerivatives(training_data,derivs!,init_parameters;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
nothing
```
