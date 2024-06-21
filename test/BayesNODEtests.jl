using UniversalDiffEq, DataFrames

data,plt = LotkaVolterra()
training_data = data[1:(end-6),:]

# test Bayesian NODEs
model = BayesianNODE(training_data)
NUTS!(model,samples = 10)
SGLD!(model,samples = 10)