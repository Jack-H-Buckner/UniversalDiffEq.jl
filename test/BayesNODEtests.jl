using UniversalDiffEq, DataFrames

data,plt = LotkaVolterra()
training_data = data[1:(end-6),:]

# test Bayesian NODEs
model = BayesianNODE(training_data)
NUTS!(model,samples = 10)
SGLD!(model,samples = 10)


plot_predictions(model)
test_data = data[(end-5):end,:]
plt,(p1,p2) = plot_forecast(model, test_data)
plot_predictions(model, test_data)