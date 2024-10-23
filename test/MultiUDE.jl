using UniversalDiffEq, DataFrames, Random

# No covariates

# random data set
training_data = DataFrame(time = vcat(1:5,1:5), 
                series = vcat(repeat([1],5),repeat([2],5)),
                x = rand(10))


# neural network 

dims_in = 1
series = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1

NN, init_params = UniversalDiffEq.MultiTimeSeriesNetwork(dims_in,series,dims_out)

# model derivitives
function derivs!(du,u,i,p,t)
    du[1] = NN(u,i,p.NN)[1] - p.m*u[1]
end

# parameters
init_parameters = (NN = init_params, m=0.5)

model = UniversalDiffEq.MultiCustomDerivatives(training_data,derivs!,init_parameters,
                                time_column_name = "time", series_column_name = "series")

gradient_descent!(model,step_size = 0.05,maxiter=2)