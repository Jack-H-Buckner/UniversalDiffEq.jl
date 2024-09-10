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
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# model derivitives
function derivs!(du,u,p,t)
    C, states = NN(u,p.NN, NNstates) # NNstates are
    du[1] = p.r*u[1] - C[1]
    du[2] = p.theta*C[1] -p.m*u[2]
end

# parameters

rng = Random.default_rng() 
NNparameters, NNstates = Lux.setup(rng,NN) 

init_parameters = (NN = NNparameters,r = 1.0,m=0.5,theta=0.5)

# priors 
function priors(p)
    L = 0.025*(p.r .- 4.0)^2
    L += 0.025*(p.m .- 5.0)^2
    L += 0.01*(p.m .- 0.75)^2
    return L
end 

model = UniversalDiffEq.CustomDerivatives(training_data,derivs!,init_parameters,priors)

gradient_descent!(model,step_size = 0.05,maxiter=2)

# with covariates
dims_in = 2
hidden_units = 10
nonlinearity = tanh
dims_out = 1
NN = Lux.Chain(Lux.Dense(dims_in,hidden_units,nonlinearity),Lux.Dense(hidden_units,dims_out))

# initialize parameters 
rng = Random.default_rng() 
NNparameters, NNstates = Lux.setup(rng,NN) 

function derivs!(du,u,covariates,p,t)
    C, states = NN(u,p.NN, NNstates) # NNstates are
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

model = CustomDerivatives(data,X,derivs!,init_parameters,priors;proc_weight=2.0,obs_weight=0.5,reg_weight=10^-4)
gradient_descent!(model,step_size = 0.05,maxiter=2)
