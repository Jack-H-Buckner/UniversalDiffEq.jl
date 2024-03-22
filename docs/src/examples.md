# Examples

## Using time dependent NODEs to predict regiem changes

One interesting use of NODE and UDE models in ecology is detecting and predicting regime changes, sudden shifts in the structure and function of an ecosystem caused by a small change in conditions. Regiem changes are caused by the interaction of non-linear feedback mechanisms, environmental variability, and long-term environmental change. NODE and UDE models built with UniversalDiffEq can capture all three of these processes, opening up he possibility of detecting and predicting regime changes from data.


In the following example, we build a NODE model for a two-species system that undergoes a regime change. The data are simulated from the Mumby-hastings model of coral-algae competition with an added term for stochastic coral mortality events and a long-term increase in the coral mortality rate from increasing temperature. The increasing temperature eventually causes the system to shift from a coral to and algae-dominated state (figure 1). The data from the time of the regime change are used to fit the model.


![figure 1: simulated regime chagne data ](figures/regiem_changes_state_estiamtes.png)


The model is a function of the area covered by coral ``p_C`` and algae ``p_A``, an environmental covariate ``X`` that is related to coral mortality and time ``t`` to capture the effect of the slowly increasing coral mortality rate. The coral and macroalgae abundances are transformed to ``x_i = softmax^{-1}(p_i)`` using the inverse soft_max transformation  before fitting the model


```math
   \frac{dx_C}{dt} = NN_1(x_C,x_A,X,t) \\
   \frac{dx_A}{dt} = NN_2(x_C,x_A,X,t) \\
```


UniversalDiffEq does not have built-in methods to construct tim
e-dependent NODES, but they can be built easily using the CustomDerivatives function. In this case, we initialize a neural network that takes four inputs (one for each species, the environmental covariate, and time) and two outputs using Lux.jl. The derivatives function `derivs!` divides time by 50 to match the sale of the other inputs, concatenates the abundances of each species, the covariate ``X`` and time into a vector, and evaluates the neural network. The UDE model is constructed using the CustomDerivatives function, passing both the species data in a data frame called `data` and the covariate in a data frame called `X`.

```julia
using Lux, UniversalDiffEq
# set neural network dimensions
dims_in = 4 
dims_out = 2
hidden = 10


# Define neurla network with Lux.jl
NN = Lux.Chain(Lux.Dense(dims_in, hidden, tanh), Lux.Dense(hidden,dims_out))
rng = Random.default_rng() 
NNparameters, NNstates = Lux.setup(rng,NN) 
parameters = (NN = NNparameters,)


# Define derivatives (time dependent NODE)
function derivs!(du,u,X,p,t)
    inputs = [u[1],u[2],X[1],t/50-1.0]
    vals = NN(inputs,p.NN,NNstates)[1]
    du[1] = vals[1]
    du[2] = vals[2]
    return du 
end 

model = UniversalDiffEq.CustomDerivatives(data[1:80,:],X,derivs!,parameters;proc_weight=2.5,obs_weight=10.0,reg_weight=10^-3.5)

gradient_decent!(model, verbos = true, maxiter = 250)
BFGS!(model, verbos = true )
```

We can use the `plot_state_estiamtes` and `plot_predictions` funcitons to test the fit of the model to the training data. 

```julia
UniversalDiffEq.plot_state_estiamtes(model)
```
![](figures/regiem_changes_state_estiamtes.png)

```julia
UniversalDiffEq.plot_predictions(model)
```
![](figures/regiem_changes_state_estiamtes.png)

Unsuprisingly, given that this is simulated data our model was able to fit the training data very closely. 

Given that the model fit that data well we can move on to our analysis. The goal of this model is to capture the effects of a slowly chanign variable on the dynamics of the coral - algae system. 