# old function names 

function CustomDiffernce(data,step,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type="L2")
    @warn ("Deprecated due to spelling error, please use CustomDifference()")
    return CustomDifference(data,step,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight = reg_weight,extrap_rho = extrap_rho,l = l,reg_type=reg_type)
end

function plot_state_estiamtes(UDE::UDE)
    @warn ("Deprecated due to spelling error, please use plot_state_estimates()")
    return plot_state_estimates(UDE)
end

function LokaVolterra(;plot = true, seed = 123,datasize = 60,T = 3.0,sigma = 0.075) 
    @warn ("Deprecated due to spelling error, please use LotkaVolterra()")
    return LotkaVolterra(plot=plot, seed=seed, datasize=datasize, T=T, sigma=sigma)
end

function LorenzLokaVolterra(;plot = true, seed = 123,datasize = 60,T = 3.0,sigma = 0.025)
    @warn ("Deprecated due to spelling error, please use LorenzLotkaVolterra()")
    return LorenzLotkaVolterra(plot=plot, seed=seed, datasize=datasize, T=T, sigma=sigma)
end

function gradient_decent!(UDE,t_skip; step_size = 0.05, maxiter = 500, verbose = false, verbos = false)
    @warn ("Deprecated due to spelling error, please use gradient_descent!()")
    return gradient_descent!(UDE,t_skip; step_size = step_size, maxiter = maxiter, verbose = verbose, verbos = verbos)
end

function gradient_decent!(UDE; step_size = 0.05, maxiter = 500, verbose = false, verbos = false)
    @warn ("Deprecated due to spelling error, please use gradient_descent!()")
    return gradient_descent!(UDE; step_size = step_size, maxiter = maxiter, verbose = verbose, verbos = verbos)
end

function interpolate_covariates(X::DataFrame,time_column_name, variable_column_name::Nothing, value_column_name::Nothing)
    @warn ("Wide format for covariate matricies is no longer supported. Please use long format by specifying variable_column_name and value_column_name")
    N, dims, T, times, data, dataframe = process_data(X,time_column_name)
    interpolations = [linear_interpolation(times, data[i,:],extrapolation_bc = Interpolations.Flat()) for i in 1:dims]
    function covariates(t)
        return [interpolation(t) for interpolation in interpolations]
    end 
    return covariates, nothing
end 

function interpolate_covariates(X::DataFrame,time_column_name,series_column_name, variable_column_name::Nothing, value_column_name::Nothing)
    @warn ("Wide format for covariate matricies is no longer supported. Please use long format by specifying variable_column_name and value_column_name")
    data_sets, times = process_multi_data2(X,time_column_name,series_column_name)
    dims = size(data_sets[1])[1]
    interpolations = [[linear_interpolation(times[j], data_sets[j][i,:],extrapolation_bc = Interpolations.Flat()) for i in 1:dims] for j in eachindex(times)]
    function covariates(t,series)
        return [interpolation(t) for interpolation in interpolations[round(Int,series)]]
    end 
    return covariates, nothing
end 