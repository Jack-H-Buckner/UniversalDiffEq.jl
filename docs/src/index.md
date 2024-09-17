# UniversalDiffEq.jl

UniversalDiffEq.jl is a library for building state-space universal dynamic equations (SS-UDEs) and neural ordinary differential equations (NODEs). These models use neural networks to learn unknown nonlinear relationships from time series data. The package provides model constructor functions to build discrete and continuous time models that combine known parametric functions with unknown functions represented by neural networks. UniversalDiffEq.jl defines a set of training procedures that can simultaneously estimate the parameters of known functions and train a neural network to represent unknown functions. The models are trained using a state-space modeling procedure, which makes them robust to unexplained variation in the system's dynamics and measurement errors. The package leverages the Julia scientific machine learning (SciML) ecosystem to build and train the models. The term "universal" comes from the [universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem).

## Universal dynamic equations

Universal dynamic equations include two classes of models: universal differential equations and universal difference equations. Universal differential equations are continuous-time models that learn the right-hand side of a system of ordinary differential equations from time series data

```math
\frac{du}{dt} = f(u).
```

Universal difference equations are discrete-time models that learn the right-hand side of a system of difference equations

 ```math
u_{t+1} = u_{t} + f(u).
```

In the simplest case, a neural network ``NN`` represents the entire right-hand side of the model ``f(\mathbf{x}) = NN(\mathbf{x})``. These models are sometimes called neural ordinary differential equations (NODEs). However, in general, the right-hand side of the model can include both neural networks and parametric functions. For example, the classic Lotka-Volterra predator-prey model includes a growth term for the prey ``rN``, a mortality rate for the predator ``mP``, and a linear interaction term between the two species ``\alpha NP``. We can use universal differential equations to build a more flexible version of this model by replacing the linear interaction term with a neural network

```math
\frac{dN}{dt} = rN - NN(N,P)

\\

\frac{dP}{dt} = \theta NN(N,P) - mP.
```

We can train the neural network ``NN`` and estimate the biological parameters ``r``, ``m``, and ``\theta`` from time series data using the training routines defined in UniversalDiffEq.jl.

## How UniversalDiffEq.jl works

UniversalDiffEq.jl builds and trains universal dynamic equations within a state-space modeling framework. State-space models describe noisy time series data by combining two models: (1) an observation model that describes the relationship between the data and the actual underlying state of the system, and (2) a process model that describes the changes in the system's state over time. This model structure allows state-space models to account for uncertainty caused by noisy observations (observation error) and inherent randomness within the system's dynamics (process error).

State-space models simultaneously estimate the parameters of the process and observation models ``\theta`` and the value of the state variables at each point in time ``\hat{u}_t``. These parameters are estimated by optimizing a loss function that combines two components: the observation loss and the process loss. The observation loss compares the state estimates to the data ``y_t`` using the observation model ``g(u_t)``

```math
L_{obs} = \frac{1}{T} \sum_{i = 1}^T (y_t - g(\hat{u}_t;\theta ))^2.
```

The process loss compares predictions of the process model ``f(u_t;\theta)`` to the estimated state variables at the next time point

```math
L_{proc} = \frac{1}{T-1} \sum_{i = 2}^T\frac{1}{\Delta t}(\hat{u}_t - f(\hat{u}_{t-1};\theta ))^2.
```

The process loss is weighted by the inverse of the time between data points ``\Delta t``.

In addition to the process and observation loss, more terms can be added to the loss function to regularize the neural network parameters and to incorporate prior information about the parameters of parametric functions

```math
L_{reg} = R(\theta).
```

The UniversalDiffEq.jl package combines weighted sums of these three components to create the full loss function

```math
L(\hat{u},\theta) = \omega_{obs} L_{obs} + \omega_{proc} L_{proc} + \omega_{reg} L_{reg}.
```

UniversalDiffEq.jl uses Optimizers.jl to find the state and parameter estimates that minimize the loss function. Currently, two optimization algorithms are available, the Adam gradient descent algorithm and the quasi-Newton algorithm BFGS.

## Data types

UniversalDiffEq.jl can train models on individual time series ``y_t`` or multiple time series ``y_{i,t}`` from systems with similar dynamics. Separate model constructors are provided for models trained on single and multiple time series. UniversalDiffEq.jl allows the models trained on multiple time series to include parameters that have different values for each time series in the training set.

UniversalDiffEq.jl can accommodate irregularly sampled data. Data for continuous-time models can be sampled at any point in time, while discrete-time models require integer-valued time steps between observations. UniversalDiffEq.jl does not accommodate observations that are missing a subset of the state variables. In these cases, data imputation is necessary.

Models built with UniversalDiffEq.jl can incorporate covariates ``X_t`` that influence the dynamics of the primary state variables. Discrete-time models require the observations of the covariates to match the time when the state variables were observed.

## Package contents
```@contents
Pages = ["Models.md","MultipleTimeSeries.md", "ModelTesting.md", "CrossValidation.md", "NutsAndBolts.md","modelanalysis.md","examples.md","API.md"]
```
