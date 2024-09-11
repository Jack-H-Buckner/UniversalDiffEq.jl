# Model Constructors

UniversalDiffEq.jl provides a set of functions to construct UDEs with varying levels of customization. The model constructors all require the data to be passed using a DataFrame object from the DataFrames.jl library. The data frame should be organized with a column for time named `t`, and the remaining columns should have the values of the state variables (`y_i`) at each point in time.

Table: **Example dataset with two state variables**
|`t`|``y_1``| ``y_2``|
|---|----|----|
|0.1| 0.0| -1.1|
|0.2| 0.01| -0.9|
|0.5| 0.51|-1.05|


Currently, missing data are not supported, but irregular intervals between time points are allowed.

Each constructor function will require additional inputs to specify the model structure. For example, the `CustomDerivatives` function requires the user to supply the known functional forms through the `derivs!` argument. These arguments are described in detail in the subsection for each model type.

Finally, the constructor functions share a set of keyword arguments used to tune the model fitting procedure. These control the weights given to the process model, observation model, and regularization in the loss function. Larger values of the regularization weight limit the complexity of the relationships learned by the neural network reducing the likelihood of overfitting. The observation weight controls how closely the estimates states ``u_t`` match the data ``y_t``, smaller values of the observation weight correspond to data sets with larger amount of observation error

- proc_weight=1.0 : The weight given to the model predictions in loss function
- obs_weight=1.0 : The weight given to the state estimates in loss function
- reg_weight=0.0 : The weight given to regularization in the loss function.

In addition to these weighting parameters, the key word argument `l` control how far the model will extrapolates beyond the observed data before reverting to a default value `extrap_rho` when forecasting.   


## Nonparametric UDES (Neural Ordinary differntial equation)

UniversalDiffEq.jl has two functions to build time series model that use a neural network to learn all of the relationship in the data. These models are sometimes referred to and neural ordinary differential equations (NODEs). 

The function `NODE` builds a continuous time UDE with a neural network representing the right hand side of the differential equation 

```math
   \frac{du}{dt} = NN(u;w,b),
```

The function `NNDE` construct a difference equation with a neural network on the right hand side

```math
   x_{t+1} = x_t + NN(x_t).
```

```
@docs; canonical=false
UniversalDiffEq.NODE(data;kwargs ... )
UniversalDiffEq.NNDE(data;kwargs ...)
```

Covariates can be added to the model by supplying a second dataframe `X`. This dataframe must have the same column name for time as the primary dataset, but the time points do not need to match because the values of the covariates between time points included in the data frame `X` are interpolated using a linear spline. The `NODE` and `NNDE` functions will append the value of the covariates at each point in time to the neural network inputs.

```math
   \frac{dx}{dt} = NN(x,X(t);w,b) \\
   x_{t+1} = x_t + NN(x_t, X(t)).
```
 

```
@docs; canonical=false
UniversalDiffEq.NODE(data,X;kwargs ... )
```

Multiple covariates with different observation frequencies can be supplied to the model with a long format dataset. If a long format dataset is provided, the user must specify which column contains the variable names and which column contains the value of the variable using the `variable_column_name` and `value_column_name` keyword arguments. In the example below, the variable column name is "variable" and the value column name is "value".

Table: **Example covariate data in long format**
|`t`|``variable``| ``value``|
|---|----|----|
|1.0| X_1| -1.1|
|2.0| X_1| -0.9|
|3.0| X_1|-1.05|
|...|...|...|
|1.0| X_2| 3.0|
|2.0| X_2| 0.95|
|4.0| X_2|-1.25|


## Customizing universal dynamic equations

### Continuous time models
The `CustomDerivatives` and `CustomDifference` functions can be used to build models that combine neural networks and known functional forms. These functions take user-defined models, construct a loss function, and provide access to the model fitting and testing functions provided by UniversalDiffEq.jl.

The `CustomDerivatives` function builds SS-UDE models based on a user-defined function `derivs!(du,u,p,t)`, which updates the vector `du` with the right-hand side of a differential equation evaluated at time `t` in state `u` given parameters `p`. The function also needs an initial guess at the model parameters, specified by a NamedTuple `initial_parameters`

```@docs; canonical=false
UniversalDiffEq.CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )
```

### Example
The following block of code shows how to build a UDE model based on the Lotka-Volterra predator-prey model where the growth rate of the prey ``r``, mortality rate of the predator ``m``, and conversion efficiency ``\theta`` are estimated and the predation rate is described by a neural network ``NN``. The equations for the ODE are

```math
\frac{dN}{dt} = rN - NN(N,P) \\
\frac{dP}{dt} = \theta NN(N,P) - mP.
```

To implement the model, we start by defining the neural network and initializing its parameters using the `Lux.Chain` and  `Lux.setup` functions. 

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

With hte neurla network in hand we can define the derivatives of the differntial equaitons model using standard Julia syntax. The `derivs` function first evaluates the neural network given the abundance of the predators and prey in the vector `u`. The neural network function `NN` requires three arguments: the current state, the network parameters, and the network states. In this example, the weights and biases are accessed through the parameters NamedTuple `p` with the key `NN`. The other model parameters are accessed with keys corresponding to their respective names.

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


### Discrete time model 

Discrete time models are constructed in a similar way to continuous time models. The user provides the right-hand side of a difference equation with the function `step` and initial parameters. The function `step(u,t,p)` takes three arguments: the value of the state variables `u`, time `t`, and model parameters `p`. 

```@docs; canonical=false
UniversalDiffEq.CustomDifference(data,step,initial_parameters;kwargs ...)
```

## Adding covariates

Covariates can also be added to UDE models by passing a data frame `X` and adding covariates as an argument to the `derivs!` function which has the new form `derivs!(du,u,X,p,t)`, where the third argument `X` is a vector of covariates. 

```@docs; canonical=false
UniversalDiffEq.CustomDerivatives(data::DataFrame,X::DataFrame,derivs!::Function,initial_parameters;kwargs ... )
```

Covariates can also be added to a discrete time framework in the same way. The `step` function should have four arguments `step(u,X,t,p)`.

```@docs; canonical=false
UniversalDiffEq.CustomDifference(data::DataFrame,X,step,initial_parameters;kwargs ... )
```
### Example

To show how adding covariates can work, the following example extends the Lotka-Volterra equations defined above to incorporate a covariate `X` that influences the abundance of predators and prey. We can model this as a linear effect with coefficients ``\beta_N`` and ``\beta_P``
```math
\frac{dN}{dt} = rN - NN(N,P) + \beta_N N \\
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
    C, states = NN(u,p.NN, NNstates) # NNstates are
    du[1] = p.r*u[1] - C[1] + p.beta[1] * X[1]
    du[2] = p.theta*C[1] -p.m*u[2] + p.beta[2] * X[1]
end

init_parameters = (NN = NNparameters, r = 1.0, m = 0.5, theta = 0.5, beta = [0,0])


model = CustomDerivatives(training_data,X,derivs!;init_parameters;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
nothing
```
### Covariates with different sampling frequencies

If you wish to build a model with covariates that are measured at different points in time, then you can provide a list of data frames as the covariates argument. Each data frame in the list should have time in the first column and the value of one of the covariates in the second column. The model will interpolate each time series with a linear spline. The value of the covariates can be accessed in the `derivs` and `step` functions for custom models by indexing into the covariates argument `X`. The values will be listed in the same order as the data frames are provided in the covariates argument.

## Adding prior information to custom models 

```@docs; canonical=false
UniversalDiffEq.CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.CustomDifference(data::DataFrame,step,initial_parameters,priors::Function;kwargs ... )
```

