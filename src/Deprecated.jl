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

function gradient_decent!(model)
    @warn ("Deprecated due to spelling error, please use gradient_descent!()")
    return gradient_descent!(model)
end