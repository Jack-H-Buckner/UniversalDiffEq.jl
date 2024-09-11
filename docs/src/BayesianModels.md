# Bayesian Models

UniversalDiffEq.jl provides training algorithms for uncertainty quantification in NODEs and UDEs using [Bayesian NODEs and UDEs](https://arxiv.org/abs/2012.07244). UniversalDiffEq.jl currently supports the classical No-U-Turn Sampling (NUTS) and the [Stochastic Gradient Langevin Dynamics](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) (SGLD) algorithms. These algorithms are available for the `BayesianUDE` constructor. This constructor can be created using the `BayesianNODE` and `BayesianCustomDerivatives` functions. These functions follow the same structure as their non-Bayesian versions `NODE` and `CustomDerivatives`.


```@docs; canonical=false
UniversalDiffEq.BayesianNODE(data;kwargs ... )
UniversalDiffEq.BayesianNODE(data,X;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )
```

## Training of Bayesian UDEs

Instead of training Bayesian UDEs using the `gradient_descent!` and `BFGS!` functions, we use the algorithms for Bayesian optimization. We repeat the [tutorial](Models.md) for regular UDEs but create a `BayesianUDE` using `BayesianCustomDerivatives` instead of `CustomDerivatives`.

```julia
using UniversalDiffEq, DataFrames, Lux, Random

data, plt = LotkaVolterra()
dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),
        Lux.Dense(hidden_units,dims_out))

# initialize the neural network states and parameters 
rng = Random.default_rng() 
NNparameters, states = Lux.setup(rng,NN) 

function lotka_volterra_derivs!(du,u,p,t)
    C, states = NN(u,p.NN,states) 
    du[1] = p.r*u[1] - C[1]
    du[2] = p.theta*C[1] -p.m*u[2]
end

initial_parameters = (NN = NNparameters, r = 1.0, m = 0.5, theta = 0.5)
model = BayesianCustomDerivatives(data,lotka_volterra_derivs!,initial_parameters)
```

Training is then done using `NUTS!` or `SGLD!`:
```julia
NUTS!(model,samples = 500)
```

```@docs; canonical=false
NUTS!(UDE::BayesianUDE;kwargs ...)
SGLD!(UDE::BayesianUDE;kwargs ...)
```

The other functions for [model analysis](modelanalysis.md) have methods for the `BayesianUDE` constructor:

```@docs; canonical=false
predict(UDE::BayesianUDE,test_data::DataFrame;summarize = true,ci = 95,df = true)
```

```@docs; canonical=false
plot_predictions(UDE::BayesianUDE;ci=95)
```

```@docs; canonical=false
plot_forecast(UDE::BayesianUDE, T::Int;ci = 95)
```

```@docs; canonical=false
plot_forecast(UDE::BayesianUDE, test_data::DataFrame;ci = 95)
```