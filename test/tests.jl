using UniversalDiffEq

# test NODE and gradient decent 
data,plt = LotkaVolterra()
model = NODE(training_data;hidden_units=20,seed=123,proc_weight=2.0,obs_weight=1.0,reg_weight=10^-3.5,l = 0.5,extrap_rho = 0.0,reg_type = "L2")
gradient_descent!(model,step_size = 0.05,maxiter=2)
plot_state_estimates(model)
plot_predictions(model)
test_data = data[(end-Ntest):end,:]
plt,(p1,p2) = plot_forecast(model, test_data)
plot_predictions(model, test_data)