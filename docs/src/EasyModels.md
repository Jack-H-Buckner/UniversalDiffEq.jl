# Easy NODE and UDE

`EasyNODE` and `EasyUDE` provide a quick, simple alternative to the other model constructors featured by UniversalDiffEq.jl. They each return pre-trained models, in which neural networks are kept to 1 hidden layer.

# EasyNODE constructors:

```@docs
EasyNODE(data,X;kwargs ... )
EasyNODE(data;kwargs ... )
```

# EasyUDE constructors:

```@docs
EasyUDE(data,derivs!,initial_parameters;kwargs ... )
EasyUDE(data::DataFrame,X,derivs!::Function,initial_parameters;kwargs ... )
```
