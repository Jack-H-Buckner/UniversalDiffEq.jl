
# Cross validation

Cross validation approximates the performance of a trained model on new data. These methods are useful for validating the performance of UDE models and for choosing between alternative models. UniversalDiffEq.jl supports two Cross validation methods, leave future out cross validation and k-fold cross validation. These two methods are based on the same assumption, we can approximate the models perforance on new data by testing it on data left out of the existing data set. Leave future out and k-fold cross validation differ in how these dat sets are constructed. Leave future out cross validation sequentially leaves data off the end of the time sereis data set with the goal of approximating the models preformance forecasting new observaitons. k-fold cross validation approximates the models ability to explain the historical dynamics of the time series by leaving block of consequtive observations out of the middel data set. Each method constructes several training and testing data sets to reduce the effect of random variation on the estiamtes of model perforamnce.  

## K-fold cross validation

k-fold cross validation breaks the data set up into k equally sized blocks of sequentially observations. The algorithm trains the model on all but one of the blocks which is used as the testing data set and repeates this procedure leaving each block out one at a time. 

The models performance is evalauted predicting one time step ahead in the testing data set. The initial points for the preditions are estimated using a particle filter algorithm, described in detail below. The forecasts are calcualted starting from the estiamte of the state variable estiamted by the particle filter and compared to the observed data point one step into the future.  

```@docs; canonical=false
cross_validation_kfold(model::UDE; kwagrs...)
```

## Leave future out cross validation

Leave future out cross validation creates training data sets by leaving observation off of the end of the data set. The model performacne is calculated by forecasting over the length of the data set and and the forecasting skill is quatified by the squared error between the model predictions and the testing data. The process is repreated on new testing data sets constructed by reoving observaitons from the end of the data set. 

```@docs; canonical=false
leave_future_out_cv(model::UDE; kwargs ...)
```

### Particle filter algorithm

Particel filter algorithms are a method for estimating the vaue of unobserved state variables given a time series and state space model. The Particle filter algorithms used in the cross validation procedures use the trained UDE model for the deterministic components of the process and observation models. The process and observaiton errors are estiamted by calcualting the total variance of the process and observation residuals ``\sigma_{total}``. The total variacne is then partitioned between process and observaiton error to match the ratio between the process and observaiton weights used to train the model. The full details of this algorithm are discussed in the supporting information to Buckner et al. 2024. 

