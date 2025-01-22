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
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))

# with wide covars
model = UniversalDiffEq.NODE(data, X_wide;time_column_name = "year")
model.constructor(data, X_wide)
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))
# with long covars
model = UniversalDiffEq.NODE(data, X;time_column_name = "year", variable_column_name = "variable", value_column_name = "value")
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))


# without covars or priors
model = UniversalDiffEq.NNDE(data;time_column_name = "year")
model.constructor(data)
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))
# with wide covars
model = UniversalDiffEq.NNDE(data, X_wide;time_column_name = "year")
model.constructor(data, X_wide)
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))
# with long covars
model = UniversalDiffEq.NNDE(data, X;time_column_name = "year", variable_column_name = "variable", value_column_name = "value")
model.constructor(data, X)
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))