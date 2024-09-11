# UDE model construction 

Some users may wish to access elements of a fitted model directly to create custom model visualizations, performance tests, or other applications not foreseen by the developers. To this end, we provide documentation of classes (Julia mutable structs) used by UniversalDiffEq.jl to build the NODE and UDE objects. The package is built around the `UDE` class which stores the data used to fit a model and instances of six submodel classes used to define the full model. 

UniversalDiffEq.jl uses a state-space modeling framework to define and fit NODE and UDE models. State-space models are a class of time series models that describe time series data with a process model that describes the dynamics of a sequence of unobserved state variables ``u_t``, as well as an observation model that defines the relationship between the state variables ``u_t`` and the observations ``x_t``.  The process model ``f`` predicts value of the state variables one step ahead
```math
\hat{u}_{t+\Delta t } = f(u_t; t, \Delta t, \theta_{proc})
```
where ``\Delta t`` is the time span between observations, and ``\theta_{proc}`` is the process model parameters. The observation model maps from the state variables ``u_t`` to the observations
```math
x_{t} = h(u_t; t, \Delta t, \theta_{obs})
```
where ``\theta_{obs}`` is the observation model parameters. In addition to these primary functions, both the observation model and process model have a loss function to measure the accuracy of their predictions. This can be thought of as the likelihood models used in generalized linear models. For example, we can measure the performance of the process model with a normal likelihood
```math
L(\hat{u}_t,u_t) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{\hat{u}_t-u_t}{\sigma})^2}
``` 
where ``\sigma`` is the variance of the prediction errors. Although in principle any likelihood can be used, we use the mean squared error in our base model specification. 

The UDE models also include submodels to regularize the process and observation models. The regularization models are functions of the model parameters that add to the loss function. The regularization models are in effect priors on the model parameters. Regularization is especially important for neural network models to reduce overfitting to training data and make the models more generalizable. For example, our default model constructors apply `L2` regularization to neural network parameters in the process model
```math
R(\theta_{proc}) = \omega ||\theta_{proc}||_{L2}
``` 
where ``\omega`` is the weight given to regularization in the overall loss function. 

These six model components are all combined into one loss functions used to fit the UDE models
```math
L(u,\theta_{proc},\theta_{obs};x) = \sum_{t =1}^{T} L_{obs}(x_t,h(u_t,\theta_{obs});\sigma_{obs}) + \sum_{t=2}^{T} L_{proc}(u_t,f(u_{t-1},\theta_{proc});\sigma_{proc}) + R_{obs}(\theta_{obs}) + R_{proc}(\theta_{proc}).
```
where the ``\sigma_i`` are parameters for the loss functions and the ``\theta_i`` are parameters for the prediction functions. 

The UDE object combines the observation and process models and their respective loss and regularization models into one larger model object along with the data used to fit the model.
```@docs; canonical=false
UDE
```

```@docs; canonical=false
UniversalDiffEq.ProcessModel
```

```@docs; canonical=false
UniversalDiffEq.LossFunction
```

```@docs; canonical=false
UniversalDiffEq.Regularization  
```

