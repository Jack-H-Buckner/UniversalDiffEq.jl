using UniversalDiffEq, DataFrames, CSV
# load data
multi_series = CSV.read("Two logistic growth series.csv", DataFrame)
training_data = multi_series[multi_series.t .<= 75, :]
testing_data = multi_series[multi_series.t .> 75, :]

# set up neural network
dims_in = 1
dims_out = 1

NN, init_params = UniversalDiffEq.SimpleNeuralNetwork(dims_in,dims_out)

# create derivative function
function multi_logistic(u, i, p, t)
    nn_out = NN(u, p.NN)[1]
    idx = round(Int, i)
    dudt = u[1] * nn_out * (1.0 - u[1] / p.K[idx])
    return [dudt]
end

# set up and train model
init_parameters = (NN = init_params, K=[100.0, 200.0])

model = UniversalDiffEq.MultiCustomDerivatives(training_data,multi_logistic,init_parameters, 
                                time_column_name = "t", series_column_name = "series")

UniversalDiffEq.train!(model; loss_function = "derivative matching", optimizer = "ADAM",
                                optim_options = (maxiter = 1,), loss_options = (d = 8,))

# error testing
predictions = UniversalDiffEq.predict(model, testing_data)
forecasts   = UniversalDiffEq.forecast(model, testing_data)