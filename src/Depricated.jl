# old function names 
"""
    CustomDiffernce(data,step,initial_parameters;kwrags...)

Depreciated spelling of CustomDifference()
...
# Arguments
- data: a DataFrame object with the time of observations in a column labeled `t` and the remaining columns the value of the state variables at each time point. 
- step: a Function of the form `step(u,t,p)` where `u` is the value of the state variables, `p` are the model parameters.
- init_parameters: A `NamedTuple` with the model parameters. Neural network parameters must be listed under the key `NN`.
...
"""
function CustomDiffernce(data,step,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25,reg_type="L2")
    @warn ("Depreciated due to spelling error, please use CustomDifference()")
    return CustomDifference(data,step,initial_parameters;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight = reg_weight,extrap_rho = extrap_rho,l = l,reg_type=reg_type)
end

"""
    plot_state_estiamtes(UDE::UDE)

Depreciated spelling of plot_state_estimates() 
"""
function plot_state_estiamtes(UDE::UDE)
    @warn ("Depreciated due to spelling error, please use plot_state_estimates()")
    return plot_state_estimates(UDE)
end

function LokaVolterra(;plot = true, seed = 123,datasize = 60,T = 3.0,sigma = 0.075) 
    @warn ("Depreciated due to spelling error, please use LotkaVolterra()")
    return LotkaVolterra(plot=plot, seed=seed, datasize=datasize, T=T, sigma=sigma)
end

function LorenzLokaVolterra(;plot = true, seed = 123,datasize = 60,T = 3.0,sigma = 0.025)
    @warn ("Depreciated due to spelling error, please use LorenzLotkaVolterra()")
    return LorenzLotkaVolterra(plot=plot, seed=seed, datasize=datasize, T=T, sigma=sigma)
end