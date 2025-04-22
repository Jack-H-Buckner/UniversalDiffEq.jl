using UniversalDiffEq, DataFrames, Random, Lux

# No covariates

# test NODE and gradient decent 
data,plt = LotkaVolterra()
training_data = data[1:(end-6),:]

# neural network 

dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1

NN, init_params = SimpleNeuralNetwork(dims_in,dims_out; hidden = hidden_units, nonlinearity = nonlinearity, seed = 123)


# model derivitives
function derivs!(u,p,t)
    C  = NN(u,p.NN) # NNstates are
    du1 = p.r*u[1] - C[1]
    du2 = p.theta*C[1] -p.m*u[2]
    return [du1,du2]
end

# parameters
init_parameters = (NN = init_params ,r = 1.0,m=0.5,theta=0.5)


model = UniversalDiffEq.CustomDerivatives(training_data,derivs!,init_parameters)
function training!(model)
    train!(model; loss_function = "derivative matching", optimizer = "ADAM",optim_options = (maxiter = 1,))
end

leave_future_out(model, training!, 3; path = false)

# with covariates

# with covariates
dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN_X = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize parameters 
rng = Random.default_rng() 
NNparameters, NNstates = Lux.setup(rng,NN_X) 

function derivs_X!(du,u,covariates,p,t)
    C, states = NN_X(u,p.NN, NNstates) # NNstates are
    du[1] = p.r*u[1] - C[1] + p.beta[1] * covariates[1]
    du[2] = p.theta*C[1] -p.m*u[2] + p.beta[2] * covariates[1]
end

init_parameters = (NN = NNparameters,r = 1.0,m=0.5,theta=0.5, beta = [0,0])
data,X,plt = LorenzLotkaVolterra(T = 7.0, datasize = 120)

model = CustomDerivatives(data,X,derivs_X!,init_parameters;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)

leave_future_out(model, training!, 3; path = false)





# multiple time series

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

leave_future_out(model, training!, 3; path = false)
