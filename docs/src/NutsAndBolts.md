# UDE model construction 

Some users may wish to access elements of a fitted model directly to create custom model visualizations, performance tests, or other applications not forseen by thr developers. To this end, we provide documentation of classes (Julia mutable structs) used by UniversalDiffEq.jl to build the NODE and UDE objects. The package is built around the `UDE` class which stores the data used to fit a model and instances of six submodel classes used to define the full model. 

UniversalDiffEq uses a state space modeling framework to define and fit NODE and UDE models. State space models are a class of time series models that describe a time series data with a process model that describes the dynaics of a sequence of unobserved state variables ``u_t`` a second observaiton model defines the relationship between the state variables ``u_t`` and the observations ``x_t``.  The process model ``f`` predicts value of the state variables one step ahead
```math
\hat{u}_{t+\Delta t } = f(u_t; t, \Delta t, \theta_{proc})
```
where ``\Delta t`` is the time span between observations, and ``\theta_{proc}`` is the model paramters. The observaiton model maps from the state variables ``u_t`` to the observations
```math
x_{t} = h(u_t; t, \Delta t, \theta_{obs})
```
where ``\theta_{obs}`` are the observaiton model parameters. In addition to these primary functions both the observaiton models and process models have loss funtion to measure the accuracy of thier predictions. These can be thought of like the likelihood models used in generalized linear models. for example, we can measure the perforance of the process model with a normal likelihood
```math
L(\hat{u}_t,u_t) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}(\frac{\hat{u}_t-u_t}{\sigma})^2}
``` 
where ``\sigma`` is the variance of the prediciton errors. Although in principal any likelihood can be used, we use the mean squared error in our base model specification. 

The UDE models also include submodels to regualrize the process and observaiton models. The regualrization models are functions of the model parameters that add to the loss funtion. The regularizaiton models are ineffect priors on the model parameters. Regularization in expecially important for nerual network models. For example, out default model constructors apply `L2` regualrizaiton to neuarl network paramters in the process model
```math
R(\theta_{proc}) = \omega ||\theta_{proc}||_{L2}
``` 
where ``\omega`` is the weight given to regualrization in the over all loss function. 

These six model components are all combined into one loss functions used to fit the UDE models
```math
L(u,\theta_{proc},\theta_{obs};x) = \sum_{t =1}^{T} L_{obs}(x_t,h(u_t,\theta_{obs});\sigma_{obs}) + \sum_{t=2}^{T} L_{proc}(u_t,f(u_{t-1},\theta_{proc});\sigma_{proc}) + R_{obs}(\theta_{obs}) + R_{proc}(\theta_{proc}).
```
where the ``\sigma_i`` are paramters for the loss functions and the ``\theta_i`` are paramters for the prediction functions. 

The UDE object combines the observation and process models and their rpective loss and regualrization models into one larger model object along with the data used to fit the model.
```@docs
UDE
```

```@docs
UniversalDiffEq.ProcessModel
```

```@docs
UniversalDiffEq.LossFunction
```

```@docs
UniversalDiffEq.Regularization  
```

