# Model Constructors

UniversalDiffEq provides a set of functions to construct NODE and UDE with varying levels of customization. The model constructors all require the data to be passed using a DataFrame object from the DataFrames.jl library. The data frame should be organized with a column for time named `t`, and the remaining columns shoud have the value of the state variables at each point in time.


|t  |``X_1`` | ``X_2``|
|---|----|----|
|0.1| 0.0| -1.1|
|0.2| 0.01| -0.9|
|0.5| 0.51|-1.05|
Table: Example data set with two state variables

Currently, missing data are not supported, but irregular intervals between time points are allowed.

Each constructor function will require additional inputs to specify the model structure. For example, the `CustomDerivatives` function requires the user to supply the known functional forms through the `derivs!` argument. These arguments are described in detail in the subsection for each model type.

Finally, the constructor functions share a set of key work arguments used to tune the model fitting procedure. These control the weight given to the process model, observaiton model and regularization in the loss function and can be tuned to control the complexity of the estimated model and to accommodate varying levels of observational errors:

- proc_weight=1.0 : The weight given to the model predictions in loss function
- obs_weight=1.0 : The weight given to the state estimates in loss function
- reg_weight=0.0 : The weight given to regularization in the loss function.

In addition to these weighting parameters, two key work arguments, `l = 0.25` and `extrap_rho = 0.0`, controls how the model extapolates beyond the observed data. The parameter `l` defines how far away the model will extrapolate before shifting to the default behavior and `extrap_rho` defines the default when extrapolating. When forecasting, the model will modify the trained process model ``f(u_t;\theta)`` when extrapolating to a new function the combines the fitted model and the default behavior


```math
\bar{f}(u_t|\theta,l,\rho )=   \left\{
\begin{array}{ll}
      f(u_t;\theta) & min(\hat{u}) < u_t < max(\hat{u}) \\
      e^{(\frac{u_t - min(\hat{u}_t)}{l})^2}f(u_t;\theta) + (1-e^{(\frac{u_t - min(\hat{u}_t)}{l})^2}) \rho &u_t < min(\hat{u}) \\
      e^{(\frac{u_t - max(\hat{u}_t)}{l})^2}f(u_t;\theta) - (1-e^{(\frac{u_t - min(\hat{u}_t)}{l})^2}) \rho &u_t > max(\hat{u}) \\
\end{array} 
\right.  
```


## NODES and NNDE
NODEs and NNDEs use neural networks to build fully nonparametric models in continuous and discrete time respectively. NODEs use a neural network as the right hand side of a differntial equation 

```math
   \frac{dx}{dt} = NN(x;w,b),
```

and NNDE use a neural network as the right hand side of a differnce equation

```math
   x_{t+1} = x_t + NN(x_t).
```

The `NODE` and `NNDE` function construct each model type.

```@docs
UniversalDiffEq.NODE(data;kwargs ... )
UniversalDiffEq.NNDE(data;kwargs ...)
```

Covariates can be added to the model by supplying a second data frame `X` This data frame must have the same format as the primary data set, but the time points need not match. The `NODE` and `NNDE` functions will append the value of the covarates at each point in time to the nerual network inputs

```math
   \frac{dx}{dt} = NN(x,X(t);w,b) \\
   x_{t+1} = x_t + NN(x_t, X(t)).
```
The value of the covarates between time point included in the data frame `X` are interpolated using a linear spline.  

```@docs
UniversalDiffEq.NODE(data,X;kwargs ... )
```

## UDEs

### Continuous time model 
The CustomDerivatives and CustomDifference function can be used to build models that combine nerual networks and known functional forms. These function take user defined models and consturct a loss function and provide access to the model fitting and testing functions provided by UniversalDiffEq.

The CustomDerivatives function build UDE models based on a user defined function `derivs!(du,u,p,t)`, which updates the vector `du` with the right hand side of a differntial equation evaluated at time `t` in state `u` given parameters `p`. The function also needs an initial guess at the model paramters, specified by a NamedTuple `initial_parameters`

```@docs
UniversalDiffEq.CustomDerivatives(data,derivs!,initial_parameters;kwargs ... )
```

### Example
The following block of code shows how to build UDE model based on the loka volterra predator prey model where the growth rate of the prey ``r``, mortaltiy rate of the predatory ``m`` and conversion efficency ``\theta`` are estiamted and the predation rate is described by a neural network ``NN``. The resulting ODE is defined by 

```math
\frac{dN}{dt} = rN - NN(N,P) \\
\frac{dP}{dt} = \theta NN(N,P) - mP.
```

To implemnt the model we start by defining he neural network object using the `Lux.Chain` funciton, 

```julia
using Lux
# Build the neurla network with lux Chain 
dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize the neurla network states and paramters 
using Random
rng = Random.default_rng() 
NNparameters, states = Lux.setup(rng,NN) 
```

New we can define the model derivatives using the usual julia functions syntax. The derivs function first evaluates the neural network given the abundance of the predators and prey in the vector `u`. The neurla network fucntion `NN` requires three arguments the current state, he newtork parameters and the network states. In this example, the wieghts and biases are accessed through the paramters NamedTupe `p` with the key `NN`. The other model parameters are accessed with key corresponding to their respective names.

```julia
function loka_volterra_derivs!(du,u,p,t)
    C, states = NN(u,p.NN, states) 
    du[1] = p.r*u[1] - C[1]
    du[2] = p.theta*C[1] -p.m*u[2]
end
```

Finally, we can define the initial paramters as a named tuple and build the model using the CustomDerivatives function.

```julia
using UniversalDiffEq
initial_parameters = (NN = NNparameters,r = 1.0,m=0.5,theta=0.5)
model = CustomDerivatives(data,loka_volterra_derivs!,initial_parameters)
```


### Discrete time model 

Discrete time models are onstructed ina similar way to continous time models. The user provides the right hand side of a differnce equation with the function `step` and initial paramters. The function `step(u,t,p)` takes three arguments the value of the state variables `u`, time `t` and model paramters `p`.

```@docs
UniversalDiffEq.CustomDiffernce(data,step,initial_parameters;kwrags...)
```

## Adding Covariates

Covariates can also be added to UDE models by passing a data frame X and adding covariates as an argument to the derivs! function which has the new form `derivs!(du,u,X,p,t)`, where the third argument `X` are a vector of covariates. 
```@docs
UniversalDiffEq.CustomDerivatives(data::DataFrame,X::DataFrame,derivs!::Function,initial_parameters;kwargs ... )
```

Covarates can also be added to discrete time framework in the same way. the step function should have four arguments `step(u,X,t,)`.
```@docs
UniversalDiffEq.CustomDiffernce(data::DataFrame,X::DataFrame,step,initial_parameters;kwargs ... )
```
### Example

To show how adding covartes can work the following example extends the loka volterra equations defined above to incorperate a covariate X that influences the abunance of predators and prey. We can model this as a linear effect with coeficents ``\beta_N`` and ``\beta_P``
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

init_parameters = (NN = NNparameters,r = 1.0,m=0.5,theta=0.5, beta = [0,0])


model = CustomDerivatives(training_data,X,derivs!;init_parameters;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
nothing
```

## adding prior information to custom models 

```@docs
UniversalDiffEq.CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.CustomDiffernce(data::DataFrame,step,initial_parameters,priors::Function;kwargs ... )
```


## Other functions
```@docs
UniversalDiffEq.DiscreteUDE(data,step,init_parameters;kwargs ...)

UniversalDiffEq.UDE(data,derivs,init_parameters;kwargs...)
```