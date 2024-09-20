# Model Constructors

UniversalDiffEq.jl provides a set of functions to construct universal differential equations (UDEs) and neural ordinary differential equations (NODEs) with varying levels of customization. The model constructors all require the data to be passed using a DataFrame object from the DataFrames.jl library. The data frame should be organized with one column for time and one column for each dimension of the observations. The name of the column for time is passed using a keyword argument `time_column_name` that has a default value "`time`".

**Table**: Example dataset with two state variables

|`time`|``y_1``| ``y_2``|
|---|----|----|
|0.1| 0.0| -1.1|
|0.2| 0.01| -0.9|
|0.5| 0.51|-1.05|


Currently, missing data are not directly supported, but irregular intervals between time points are allowed.

Each constructor function requires additional inputs to specify the model structure. For example, the `CustomDerivatives` function requires the user to supply the known functional forms through the `derivs!` argument. The subsection for each model type describes these arguments in detail.

Finally, the constructor functions share a set of keyword arguments (`kwargs`) used to tune the model fitting procedure. These control the weights given to the process model, observation model, and regularization in the loss function. Larger values of the regularization weight limit the complexity of the relationships learned by the neural network to reduce the likelihood of overfitting. The observation weight controls how closely the estimated states ``u_t`` match the observations ``y_t``; smaller values of the observation weight correspond to datasets with larger amounts of observation error and vice versa.

- `proc_weight=1.0` : The weight given to the model predictions in the loss function
- `obs_weight=1.0` : The weight given to the state estimates in the loss function
- `reg_weight=0.0` : The weight given to regularization in the loss function

In addition to these weighting parameters, the keyword argument `l` controls how far the model will extrapolate beyond the observed data before reverting to a default value `extrap_rho` when forecasting.


## NODEs (i.e., nonparametric universal dynamic equations)

UniversalDiffEq.jl has two functions to build time series models that use a neural network to learn the dynamics of a time series. The function `NODE` builds a continuous-time UDE with a neural network representing the right-hand side of the differential equation


```math
   \frac{du}{dt} = NN(u;w,b),
```

The function `NNDE` (neural network difference equation) constructs a discrete-time model with a neural network on the right-hand side

```math
   x_{t+1} = x_t + NN(x_t).
```

```@docs; canonical=false
UniversalDiffEq.NODE(data;kwargs ... )
UniversalDiffEq.NNDE(data;kwargs ...)
```

Covariates can be included in the model by supplying a second data frame, `X.` This data frame must have the same column name for time as the primary dataset, but the time points do not need to match because the values of the covariates between time points included in the data frame `X` are interpolated using a linear spline.  The `NODE` and `NNDE` functions will append the value of the covariates at each point in time to the neural network inputs

```math
   \frac{dx}{dt} = NN(x,X(t);w,b) \\
   x_{t+1} = x_t + NN(x_t, X(t)).
```


```@docs; canonical=false
UniversalDiffEq.NODE(data,X;kwargs ... )
```

Long-format datasets can be used to define models with multiple covariates that have different sampling frequencies. If a long-format dataset is provided, the user must specify which column contains the variable names and which column contains the values of the variables using the `variable_column_name` and `value_column_name` keywords. In the example below, the variable column name is "`variable`", and the value column name is "`value`".

**Table**: Example covariate data in long format

|`time`|`variable`| `value`|
|---|----|----|
|1.0| X_1| -1.1|
|2.0| X_1| -0.9|
|3.0| X_1|-1.05|
|...|...|...|
|1.0| X_2| 3.0|
|2.0| X_2| 0.95|
|4.0| X_2|-1.25|


## Customizing universal dynamic equations

The `CustomDerivatives` and `CustomDifference` functions can be used to build models that combine neural networks and known functional forms. These functions take user-defined models, construct a loss function, and access the model fitting and testing functions provided by UniversalDiffEq.jl.

### Continuous-time models

The `CustomDerivatives` function builds SS-UDE models based on a user-defined function `derivs!(du,u,p,t)`, which updates the vector `du` with the right-hand side of a differential equation evaluated at time `t` in state `u` given parameters `p`. The function also needs an initial guess at the model parameters, specified by a NamedTuple `initial_parameters`


```@docs; canonical=false
UniversalDiffEq.CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )
```

### Example
The following block of code shows how to build a UDE model based on the Lotka-Volterra predator-prey model where the growth rate of the prey ``r``, mortality rate of the predator ``m``, and conversion efficiency ``\theta`` are estimated, and the predation rate is described by a neural network ``NN``

```math
\frac{dN}{dt} = rN - NN(N,P)

\\

\frac{dP}{dt} = \theta NN(N,P) - mP.
```

To implement the model, we define the neural network and initialize its parameters using the `Lux.Chain` and  `Lux.setup` functions.

```julia
using Lux
# Build the neural network with Lux.Chain
dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize the neural network states and parameters
using Random
rng = Random.default_rng()
NNparameters, states = Lux.setup(rng,NN)
```


With the neural network in hand, we define the derivatives of the differential equations model using standard Julia syntax. The `derivs` function first evaluates the neural network given the abundances of the predators and prey `u`. The neural network function `NN` requires three arguments: the state variables `u`, the network parameters, and the network states. In this example, we store the model parameters in a named tuple `p`, and we access the neural network parameters with the `NN`. We access the other model parameters using keys corresponding to their respective names.

```julia
function lotka_volterra_derivs!(du,u,p,t)
    C, states = NN(u,p.NN,states)
    du[1] = p.r*u[1] - C[1]
    du[2] = p.theta*C[1] -p.m*u[2]
end
```

Finally, we can define the initial parameters as a NamedTuple and build the model using the `CustomDerivatives` function.

```julia
using UniversalDiffEq
initial_parameters = (NN = NNparameters, r = 1.0, m = 0.5, theta = 0.5)
model = CustomDerivatives(data,lotka_volterra_derivs!,initial_parameters)
```


### Discrete-time models

Discrete-time models are constructed similarly to continuous-time models. The user provides a dataset, initial parameter values, and the right-hand side of a discrete-time equation with the function `step`

```math
U_{t+1} = \text{step}(u_t,t,p).
```

The function `step(u,t,p)` takes three arguments: the value of the state variables `u`, time `t`, and model parameters `p`.

```@docs; canonical=false
UniversalDiffEq.CustomDifference(data,step,initial_parameters;kwargs ...)
```


## Adding covariates

Covariates are added to UDE models by passing a data frame `X` to the constructor function. The covariates must also be added as an argument to the `derivs!` function, which has the new form `derivs!(du,u,X,p,t)`, where the third argument `X` is a vector with the value of each covariate at time `t`.

```@docs; canonical=false
UniversalDiffEq.CustomDerivatives(data::DataFrame,X::DataFrame,derivs!::Function,initial_parameters;kwargs ... )
```

Covariates can  be added to discrete-time models in the same way. In this case the `step` function should have four arguments `step(u,X,t,p)`.

```@docs; canonical=false
UniversalDiffEq.CustomDifference(data::DataFrame,X,step,initial_parameters;kwargs ... )
```
### Example

We extend the Lotka-Volterra equations defined in the prior example to  incorporate a covariate `X` that influences the abundance of predators and prey. We model this effect with linear coefficients ``\beta_N`` and ``\beta_P``

```math
\frac{dN}{dt} = rN - NN(N,P) + \beta_N N

\\

\frac{dP}{dt} = \theta NN(N,P) - mP + \beta_P P.
```

```julia
# set up neural network
using Lux
dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize parameters
using Random
rng = Random.default_rng()
NNparameters, NNstates = Lux.setup(rng,NN)

function derivs!(du,u,X,p,t)
    C, states = NN(u,p.NN, NNstates)
    du[1] = p.r*u[1] - C[1] + p.beta[1] * X[1]
    du[2] = p.theta*C[1] -p.m*u[2] + p.beta[2] * X[1]
end

init_parameters = (NN = NNparameters, r = 1.0, m = 0.5, theta = 0.5, beta = [0,0])


model = CustomDerivatives(training_data,X,derivs!,init_parameters;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
nothing
```

## Adding prior information

Users can add priors and custom neural network regularization functions by passing a function to the model constructor that takes the parameters as an argument and returns the loss associated with those parameter values. Please note that the loss functions defined by UniversalDiffEq.jl use the mean squared errors of the model rather than a likelihood function, so priors over the model parameters will not have the usual Bayesian interpretation.

```@docs; canonical=false
UniversalDiffEq.CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.CustomDifference(data::DataFrame,step,initial_parameters,priors::Function;kwargs ... )
```
