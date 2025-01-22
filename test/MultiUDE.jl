using UniversalDiffEq, DataFrames, Random


# random data set

training_data = DataFrame(time = vcat(1:10,1:10), 
                series = vcat(repeat([1],10),repeat([2],10)),
                x = rand(20))

# neural network 

dims_in = 1
hidden_units = 10
nonlinearity = tanh
dims_out = 1

NN, init_params = UniversalDiffEq.SimpleNeuralNetwork(dims_in,dims_out)

# model derivitives
function derivs!(u,i,p,t)
    NN(u,p.NN)[1] .- p.m*u[1]
end

# parameters
init_parameters = (NN = init_params, m=0.5)

model = UniversalDiffEq.MultiCustomDerivatives(training_data,derivs!,init_parameters,
                                time_column_name = "time", series_column_name = "series")
#UniversalDiffEq.train!(model; loss_function = "shooting", optimizer = "ADAM",optim_options = (maxiter = 1,))
UniversalDiffEq.train!(model; loss_function = "derivative matching", optimizer = "ADAM",
                                optim_options = (maxiter = 1,), loss_options = (d = 8,))

UniversalDiffEq.train!(model; loss_function = "shooting", optimizer = "ADAM",optim_options = (maxiter = 1,))
UniversalDiffEq.train!(model; loss_function = "multiple shooting", optimizer = "ADAM",optim_options = (maxiter = 1,))
UniversalDiffEq.train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))
UniversalDiffEq.train!(model; loss_function = "marginal likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,),loss_options = (observation_error = 0.1,))



# test discrete time models 

NN, init_params = UniversalDiffEq.SimpleNeuralNetwork(dims_in,dims_out)

function diff(u,i,p,t)
    ut = u .+ NN(u,p.NN)[1]
    return ut
end

init_parameters2 = (NN = init_params,)

model = UniversalDiffEq.MultiCustomDifference(training_data,diff,init_parameters2;
                            time_column_name = "time", series_column_name = "series")

UniversalDiffEq.train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))
UniversalDiffEq.train!(model; loss_function = "marginal likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,),loss_options = (observation_error = 0.1,))

                            
function diff2(u,i,X,p,t)
    ut = u .+ NN(u,p.NN)[1] .- p.b* X[1]
    return ut
end

init_parameters3 = (NN = init_params, b = 1.0)
training_X = DataFrame(time = vcat(1:5,1:5), series = vcat(repeat([1],5),repeat([2],5)),X = rand(10))                        
                            
model = UniversalDiffEq.MultiCustomDifference(training_data,training_X,diff2,init_parameters3;
                            time_column_name = "time", series_column_name = "series")

UniversalDiffEq.train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))
UniversalDiffEq.train!(model; loss_function = "marginal likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,),loss_options = (observation_error = 0.1,))
