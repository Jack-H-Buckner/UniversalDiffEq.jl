<img alt="Package logo" width = "500px" src="README images/Package_logo.png" />


| Minimum V-1.7 | V-1.10 | Nightly |
|-----------------|---------------------|-------------------------|
| [![Build Status](https://github.com/jarroyoe/UniversalDiffEq.jl/actions/workflows/CI-V1-7.yml/badge.svg)](https://github.com/jarroyoe/UniversalDiffEq.jl/actions/workflows/CI-V1-7.yml) | [![Build Status](https://github.com/jarroyoe/UniversalDiffEq.jl/actions/workflows/CI-V1-10.yml/badge.svg)](https://github.com/jarroyoe/UniversalDiffEq.jl/actions/workflows/CI-V1-10.yml)| [![Build Status](https://github.com/jarroyoe/UniversalDiffEq.jl/actions/workflows/CI-Nightly.yml/badge.svg)](https://github.com/jarroyoe/UniversalDiffEq.jl/actions/workflows/CI-Nightly.yml)|


[![Docs](https://img.shields.io/badge/docs-dev-blue)](https://jack-h-buckner.github.io/UniversalDiffEq.jl/dev/)
[![Preprint](https://img.shields.io/badge/preprint-arXiv-red)](https://arxiv.org/abs/2410.09233)

UniversalDiffEq.jl builds [Universal Differential Equations](https://arxiv.org/abs/2001.04385) (UDEs), dynamic models that combine neural networks with parametric equations to learn nonlinear dynamics from time series data. This package provides functions to build and train UDEs. It includes several training routines designed to work well when the data contain observation error and the underlying process is stochastic. The package uses [Lux.jl](https://lux.csail.mit.edu/stable/) to construct neural networks built into the model and [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) for automatic differentiation.

The package provides one specific implementation of universal differential equations. If you need to develop highly customized models, please use [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) instead.

To install and load UniversalDiffEq.jl, open Julia and type the following code:

```
]add LLVM
add FFMPEG
add UniversalDiffEq
using UniversalDiffEq
```

To access the latest version under development with the newest features use:

```
add https://github.com/Jack-H-Buckner/UniversalDiffEq.jl.git
```

# Tutorial
As a simple example to get started on UniversalDiffEq.jl, we fit a UDE model to a synthetic data set generated with the classical [Lotka-Volterra predator-prey model](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), which uses a neural network to learn the species interaction term. Model fit is given by the normalized root mean square error (NRMSE) for prey ($x_1$) and predator ($x_2$).

```math
\frac{dx_1}{dt} = rx_1 (1-x_1/k)- NN(x_1,x_2)
```

```math
\frac{dx_2}{dt} = \theta NN(x_1,x_2) - mx_2
```

```julia
using UniversalDiffEq, DataFrames

data,plt = UniversalDiffEq.LotkaVolterra() # Generate synthetic predator prey data

# Build neural network
NN,params = UniversalDiffEq.SimpleNeuralNetwork(2,1)

# Set model parameters
init_parameters = (NN = params, r = 0.1, k = 10.1, m = 0.1, theta = 0.1)

# Define rates of change UDE model
function dudt(u,p,t)

    C = abs(NN(u,p.NN)[1]) # Calculate prey consumption rate with neural network
    r, k, theta, m = abs.([p.r, p.k, p.theta, p.m]) # transform model parameters to get positive values

    # Calculate rates of change for prey u[1] and predator u[2]
    dx1 = r*u[1]*(1-u[1]/k) - C
    dx2 = theta*C - m*u[2]

    return [dx1,dx2]
end

# Construct UDE model using the CustomDerivatives function
model = CustomDerivatives(data,dudt,init_parameters)

# Use the train function to fit the model to the data
train!(model;  loss_function = "derivative matching",
               optimizer = "ADAM",
               regularization_weight = 0.0,
               verbose = false,
               loss_options = (d = 10, ),
               optim_options = (maxiter = 1000, step_size = 0.01))

# Compare the estimated values of the state variables to the data set
plot_state_estimates(model)
```

<img alt="Lotka-Volterra Predictions" width = "500px" src="README images/state_plot.png" />

```julia
# Compare predicted to observed changes
plot_predictions(model)
```
<img alt="Lotka-Volterra States" width = "500px" src="README images/predictions_plot.png" />

```julia
# plot forecast
p1, (p2,p3) = UniversalDiffEq.plot_forecast(model, 50)
p1
```
<img alt="Lotka-Volterra States" width = "500px" src="README images/forecast_plot.png" />

Please see the documentation for a detailed tutorial.

# Acknowledgements
<img alt="NSF Logo" width="200px" src="README images/NSF_logo.png" />

The development of this package is supported by the National Science Foundation, award \#2233982 on Model Enabled Machine Learning (MnML) for Predicting Ecosystem Regime Shifts.
