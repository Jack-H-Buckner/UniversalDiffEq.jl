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


# without covars or priors
model = UniversalDiffEq.NODE(data;time_column_name = "year")
model.constructor(data)
gradient_descent!(model,maxiter = 2, verbose= true)
derivative_matching!(model,maxiter = 2, verbose= true)


# with wide covars
model = UniversalDiffEq.NODE(data, X_wide;time_column_name = "year")
model.constructor(data, X_wide)
gradient_descent!(model,maxiter = 2, verbose= true)
derivative_matching!(model,maxiter = 2, verbose= true)

# with long covars
model = UniversalDiffEq.NODE(data, X;time_column_name = "year", variable_column_name = "variable", value_column_name = "value")
model.constructor(data, X)
gradient_descent!(model,maxiter = 2, verbose= true)



# without covars or priors
model = UniversalDiffEq.NNDE(data;time_column_name = "year")
model.constructor(data)
gradient_descent!(model,maxiter = 2, verbose= true)

# with wide covars
model = UniversalDiffEq.NNDE(data, X_wide;time_column_name = "year")
model.constructor(data, X_wide)
gradient_descent!(model,maxiter = 2, verbose= true)

# with long covars
model = UniversalDiffEq.NNDE(data, X;time_column_name = "year", variable_column_name = "variable", value_column_name = "value")
model.constructor(data, X)
gradient_descent!(model,maxiter = 2, verbose= true)
