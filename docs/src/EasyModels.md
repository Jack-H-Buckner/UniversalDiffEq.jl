# Easy NODE and UDE

`EasyNODE` and `EasyUDE` provide a quick, simple alternative to the other model constructors featured by UniversalDiffEq.jl. They each return pre-trained models, in which neural networks are kept to one hidden layer. The models are trained using the `gradient_descent!` function.

## EasyNODE constructors

```@docs; canonical=false
EasyNODE(data,X;kwargs ... )
EasyNODE(data;kwargs ... )
```

Creating an `UDE` constructor using the `EasyNODE` function is equivalent to creating it using the `NODE` function and running `gradient_descent!`.

```julia
easy_model = EasyNODE(data)

#Is equivalent to running
model = NODE(data)
gradient_descent!(model)
```

## EasyUDE constructors

```@docs; canonical=false
EasyUDE(data,known_dynamics!,initial_parameters;kwargs ... )
EasyUDE(data::DataFrame,X,known_dynamics!::Function,initial_parameters;kwargs ... )
```

Unlike `EasyNODE`, running `EasyUDE` is not equivalent to running `CustomDerivatives` and `gradient_descent!`. `EasyUDE` creates `UDE` constructors with a continuous process model of the form

```math
\frac{dx}{dt} = NN(x;w,b) + f(x;a).
```

where $f$ corresponds to the `known_dynamics!` argument, and $a$ is the `initial_parameters` argument in `EasyUDE`.

```julia
function known_dynamics!(du,u,parameters,t)
    du .= parameters.a.*u .+ parameters.b #some function here
    return du
end
initial_parameters = (a = 1, b = 0.1)
easy_model = EasyUDE(data,known_dynamics!,initial_parameters)

#Is equivalent to running
using Lux, Random
dims_in = 1
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),
                Lux.Dense(hidden_units,dims_out))

rng = Random.default_rng()
NNparameters, states = Lux.setup(rng,NN)

function derivs!(du,u,p,t)
    C, states = NN(u,p.NN,states)
    du .= C .+ a*u .+ b
    return du
end

initial_parameters = (a = 1, b = 0.1)
model = CustomDerivatives(data,derivs!,initial_parameters)
```
