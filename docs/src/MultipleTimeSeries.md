# Fitting a model to multiple time series

UniversalDiffEq.jl provides a set of functions to fit models to multiple time series. The functions for this mirror the functions for fitting NODEs and UDEs to single time series with the prefix `Multi`. For example, to build a NODE model for multiple time series you would use the `MultiNODE` function. The functions for model fitting, testing, and visualization have the same names. The other important difference is the data format: a column with a unique index for each time series must be included. 

## Dataframe
|t  | series | x1            | x2             |
|---|--------|---------------|----------------|
|1  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|2  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|3  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|1  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |
|2  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |
|3  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |

Covariates can be added to the models as well. The covariates data frame must have the same structure. 
```@docs; canonical=false
UniversalDiffEq.MultiNODE(data;kwargs...)
UniversalDiffEq.MultiNODE(data,X;kwargs...)
```