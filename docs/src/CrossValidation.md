
# Cross validation

Cross validation approximates the performance of a trained model on new data. These methods are useful for validating the performance of UDE models and for choosing between alternative models. UniversalDiffEq.jl supports two Cross validation methods, leave future out cross validation and k-fold cross validation. These two methods are based on the same assumption, we can approximate the models perforance on new data by testing it on data left out of the existing data set. Leave future out and k-fold cross validation differ in how these dat sets are constructed. Leave future out cross validation sequentially leaves data off the end of the time sereis data set with the goal of approximating the models preformance forecasting new observaitons. k-fold cross validation approximates the models ability to explain the historical dynamics of the time series by leaving block of consequtive observations out of the middel data set. Each method constructes several training and testing data sets to reduce the effect of random variation on the estiamtes of model perforamnce.  

## K-fold cross validation

k-fold cross validation breaks the data set up into k equally sized blocks of sequentially observations. The algorithm trains the model on all but one of the blocks which is used as the testing data set and repeates this procedure laving each block out one at a time. 

The models performance is evalauted predicting one time step ahead in the testing data set. The initial points for the preditions are estimated using a particle filter algorithm, described in detail below. The forecasts are calcualted starting from the estiamte of the state variable estiamted by the particle filter and compared to the observed data point one step into the future.  

```@docs
cross_validation_kfold(model::UDE; kwagrs...)
```


## Particle filter algorithm

Particel filter algorithms are a method for estimating the vaue of unobserved state variables given a time series and state space model. The Particle filter algorithms used in the cross validation procedures use the trained UDE model for the deterministic components of the process and observation models. The process and observaiton errors are estiamted from the 