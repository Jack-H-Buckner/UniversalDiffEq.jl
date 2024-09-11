using UniversalDiffEq, DataFrames

data,plt = LotkaVolterra()
training_data = data[1:(end-6),:]

# test Bayesian NODEs
model = EasyNODE(training_data)

function known_dynamics!(du,u,parameters,t)
    du .= parameters.a.*u .+ parameters.b #some function here
    return du
end
initial_parameters = (a = 1, b = 0.1)
easy_model = EasyUDE(training_data,known_dynamics!,initial_parameters)