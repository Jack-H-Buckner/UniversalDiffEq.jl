# UniversalDiffEq.jl

A library for building nonlinear time series models for ecological data with neural networks. UniversalDiffEq.jl uses the Lux.jl and DiffEqFlux.jl libraries to construct time series models and defines a set of training procedures designed to handle common problems with ecological data, like sparse and noisy observations.

## Universal differential equations

UniversalDiffEq.jl builds universal differential equations (UDEs), a class of models that uses neural networks to learn a differential equation. For example, we can build a model that uses a neural network to learn the right-hand side of a system of differential equations

```math
\frac{dx}{dt} = NN(x).
```

It is also possible to create models that combine neural networks with known functional forms. For example, we might use a neural network to learn the interaction term in a predator-prey model based on the Lotka-Volterra equations

```math
\frac{dN}{dt} = rN - NN(N,P) \\
\frac{dP}{dt} = \theta NN(N,P) - mP.
```

UniversalDiffEq.jl also provides functions to construct models based on difference equations

```math
x_{t+1} = NN(x).
```

## How UniversalDiffEq.jl works

UniversalDiffEq.jl builds and trains UDEs within a state-space modeling framework. State-space models describe noisy time series data by combining two models: (1) an observation model that describes the relationship between the data and the true underlying state of the system, and (2) a process model that describes the changes in the system's state over time. This model structure allows state-space models to account for uncertainty caused by noisy observations (observation error) and inherent randomness within the system's dynamics (process error).

State-space models simultaneously estimate the parameters of the process and observation models ``\theta`` and the value of the state variables at each point in time ``\hat{u}_t``. These parameters are estimated by optimizing a loss function that combines two components, the observation loss and the process loss, which evaluate the performance of the state estimates and the process model parameters, respectively. The observation loss compares the state estimates ``\hat{u}_t`` to the data ``x_t`` using the observation model ``g(u_t)``. For example, we can use the mean squared error loss

```math
L_{obs} = \frac{1}{T} \sum_{i = 1}^T (x_t - g(\hat{u}_t;\theta ))^2.
```

The process loss is calculated by comparing the predictions of the process model ``f(u_t;\theta)`` to the estimated states one step ahead again. The mean squared error performs well

```math
L_{proc} = \frac{1}{T-1} \sum_{i = 2}^T(\hat{u}_t - f(\hat{u}_{t-1};\theta ))^2.
```

Additional terms can be added to regularize neural network parameters or incorporate prior information

```math
L_{reg} = R(\theta).
```

The UniversalDiffEq.jl package combines weighted sums of these three components to create the full loss function

```math
L(\hat{u},\theta) = \omega_{obs} L_{obs} + \omega_{proc} L_{proc} + \omega_{reg} L_{reg}.
```

UniversalDiffEq.jl uses Optimizers.jl to find the state and parameter estimates that minimize the loss function. Currently, two optimization algorithms are available, the Adam gradient descent algorithm and the quasi-Newton algorithm BFGS.


## Package Contents
```@contents
Pages = ["Models.md", "ModelTesting.md","MultipleTimeSeries.md","NutsAndBolts.md","modelanalysis.md","examples.md","API.md"]
```



