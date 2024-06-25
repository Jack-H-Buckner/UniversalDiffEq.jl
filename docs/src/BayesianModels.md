# Bayesian Models

```@docs
UniversalDiffEq.BayesianUDE
UniversalDiffEq.BayesianNODE
```



# NODEs

```@docs
UniversalDiffEq.BayesianNODE(data;kwargs ... )
UniversalDiffEq.BayesianNODE(data,X;kwargs ... )
```

# UDEs

```@docs
UniversalDiffEq.BayesianCustomDerivatives(data,derivs!,initial_parameters;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )

```