# tests
using RCall, UniversalDiffEq, DataFrames

# generate data
X = DataFrame(year = [1,2,3], x1 = rand(3), x2 = rand(3))
X_wide = X

# get long format
@rput X
R"""
X <- reshape2::melt(X, id.var = "year")
"""
@rget X
X[1:4,:]

# derivatives
function dudt!(du,u,p,t)
    du .= zeros(3)
end 


function dudtX!(du,u,X,p,t)
    du .= zeros(3)
end 

# priors
priors = x -> 0


# without covars or priors
model = UniversalDiffEq.CustomDerivatives(data,dudt!,(p = 0, );time_column_name = "year",
                             proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,
                            extrap_rho=0.1,l=0.25,reg_type = "L2")
model.constructor(data)
gradient_descent!(model,maxiter = 2)

# with priors
model = UniversalDiffEq.CustomDerivatives(data,dudt!,(p = 0, ), priors;time_column_name = "year",
                             proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,
                            extrap_rho=0.1,l=0.25,reg_type = "L2")
model.constructor(data)
gradient_descent!(model,maxiter = 2)

# with long covars
model = UniversalDiffEq.CustomDerivatives(data,X, dudt!,(p = 0, );time_column_name = "year",
                            variable_column_name = "variable", value_column_name = "value",
                            proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,
                            extrap_rho=0.1,l=0.25,reg_type = "L2")
model.constructor(data, X)
gradient_descent!(model,maxiter = 2)

# with wide covars
model = UniversalDiffEq.CustomDerivatives(data, X_wide, dudt!,(p = 0, );time_column_name = "year",
                            proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,
                            extrap_rho=0.1,l=0.25,reg_type = "L2")
model.constructor(data, X_wide)
gradient_descent!(model,maxiter = 2)

# with long covars and priors
model = UniversalDiffEq.CustomDerivatives(data,X, dudt!,(p = 0, ),priors;time_column_name = "year",
                            variable_column_name = "variable", value_column_name = "value",
                            proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,
                            extrap_rho=0.1,l=0.25,reg_type = "L2")
model.constructor(data, X)
gradient_descent!(model,maxiter = 2)

# with wide covars and priors
model = UniversalDiffEq.CustomDerivatives(data, X_wide, dudt!,(p = 0, ), priors; time_column_name = "year",
                            proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,
                            extrap_rho=0.1,l=0.25,reg_type = "L2")
model.constructor(data, X_wide)
gradient_descent!(model,maxiter = 2)
