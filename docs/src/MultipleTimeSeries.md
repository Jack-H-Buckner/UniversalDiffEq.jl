# Fitting a model to multiple time series

UniversalDiffEq provides a set of functions to fit models to multiple time series. The functions for this mirror the fucntions for fitting NODEs and UDE to single time series with prefix Multi. For example, to build a NODE model for multiple time series you woudl use the `MultiNODE` function. The functions for model fitting, testing and visualization have the same names. The other imporant differnce is the data formate, a colum with a unique index for each time series must be included. 

## Dataframe
|t  | series | x1            | x2             |
|---|--------|---------------|----------------|
|1  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|2  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|3  | 1      | ``x_{1,1,t}`` | ``x_{1,2,t}``  |
|1  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |
|2  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |
|3  | 2      | ``x_{2,1,t}`` | ``x_{2,2,t}``  |

Covarate can be added to the models as well. The covarates dataframe must have the same sturcture 
```@docs
UniversalDiffEq.MultiNODE(data;kwargs...)
```