using UniversalDiffEq, DataFrames, Distributions, Plots

y = rand(Normal(),200)
x  = rand(Normal(),200)
data = DataFrame(t = 1:200, y = y, x = x)


init_params = (theta = 0, )
function derivs!(du,u,p,t)
    du .= [0.0,0.0]
end 


proc_weight = 0.005; obs_weight = 1.0
model_1 = UniversalDiffEq.CustomDerivatives(data,derivs!,init_params,time_column_name = "t",proc_weight=proc_weight,obs_weight=obs_weight,reg_weight = 0.0)

proc_weight = 1.0; obs_weight = 0.05
model_2 = UniversalDiffEq.CustomDerivatives(data,derivs!,init_params,time_column_name = "t",proc_weight=proc_weight,obs_weight=obs_weight,reg_weight = 0.0)


total_rmse, variable_rmse, diagnostics = UniversalDiffEq.cross_validation_kfold(model_1; k = 20,maxiter = 200,step_size2 = 0.01,maxiter2 = 100,N  = 1000)
println(total_rmse)
println(variable_rmse)
plt1, plt2 = UniversalDiffEq.kfold_diagnositcs_plot(diagnostics)
savefig(plt1,"~/Documents/test_CV_mod1_plt1.pdf")
savefig(plt2,"~/Documents/test_CV_mod1_plt2.pdf")


total_rmse, variable_rmse, diagnostics  = UniversalDiffEq.cross_validation_kfold(model_2; k = 20,maxiter = 200,step_size2 = 0.01,maxiter2 = 100,N  = 1000)
println(total_rmse)
println(variable_rmse)
plt1, plt2 = UniversalDiffEq.kfold_diagnositcs_plot(diagnostics)
savefig(plt1,"~/Documents/test_CV_mod2_plt1.pdf")
savefig(plt2,"~/Documents/test_CV_mod2_plt2.pdf")