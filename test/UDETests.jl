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


# priors 
function priors(p)
    L = 0.025*(p.r .- 4.0)^2
    L += 0.025*(p.m .- 5.0)^2
    L += 0.01*(p.m .- 0.75)^2
    return L
end 

model = UniversalDiffEq.CustomDerivatives(training_data,derivs!,init_parameters,priors)

train!(model; loss_function = "derivative matching", optimizer = "ADAM",optim_options = (maxiter = 1,))
train!(model; loss_function = "shooting", optimizer = "ADAM",optim_options = (maxiter = 1,))
train!(model; loss_function = "multiple shooting", optimizer = "ADAM",optim_options = (maxiter = 1,))
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))
train!(model; loss_function = "marginal likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,), loss_options = (observation_error = 0.01,))
train!(model; loss_function = "spline gradient matching", optimizer = "ADAM",optim_options = (maxiter = 1,), loss_options = (observation_error = 0.01,))
train!(model; loss_function = "neural gradient matching", optimizer = "ADAM",optim_options = (maxiter = 1,), loss_options = (observation_error = 0.01,))



# alternative training routines
# model derivitives
function derivs(du,u,p,t)
    C  = NN(u,p.NN) # NNstates are
    du .= [p.r*u[1] - C[1], p.theta*C[1] - p.m*u[2]]
    return du
end

# parameters
init_parameters = (NN = init_params ,r = 1.0,m=0.5,theta=0.5)


model = UniversalDiffEq.CustomDerivatives(training_data,derivs,init_parameters)

train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))

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

function priors(p)
    L = 0.01 * (p.r - 4.0).^2
    L = 0.01 * (p.m - 5.0).^2
    L = 0.01 * (p.theta - 0.75).^2
end 

init_parameters = (NN = NNparameters,r = 1.0,m=0.5,theta=0.5, beta = [0,0])

data,X,plt = LorenzLotkaVolterra(T = 7.0, datasize = 120)

model = CustomDerivatives(data,X,derivs_X!,init_parameters,priors;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
train!(model; loss_function = "conditional likelihood", optimizer = "ADAM",optim_options = (maxiter = 1,))