# Bayesian Models

```@docs
UniversalDiffEq.BayesianUDE
UniversalDiffEq.BaesianNODE
```



# NODEs

```@docs
UniversalDiffEq.BayesianNODE(data;kwargs ... )
UniversalDiffEq.BayesianNODE(data,X;kwargs ... )
```

# UDEs

```@docs
UniversalDiffEqBayesianCustomDerivatives(data,derivs!,initial_parameters;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,priors::Function;kwargs ... )
UniversalDiffEq.BayesianCustomDerivatives(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )

```