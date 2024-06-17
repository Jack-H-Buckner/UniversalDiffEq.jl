
"""
data:

level 1 | level2 | ... | level n | series | t | x1 | x2 | ... | xn |

level 1  > level 2 > ... > level n > series

"""
function init_nested_parameters(data, construct_parameters, n_levels; seed = 10)
    
    dims = size(data)[2] - n_levels - 1
    
    # set seed to initialize NN parameters
    Random.seed!(seed)
    parameters = (level_0 = (group_1 = construct_parameters(),), )
    
    # iniailize lower levels
    for i in 1:(n_levels-1)
        Ngroups = length(unique(data[:,i]))
        parameters_level_i = (group_1 = construct_parameters(), )
        
        for j in 2:Ngroups
 
            pars_dict = Dict(string("group_", j) => construct_parameters(),)
            pars_tuple = (; (Symbol(k) => v for (k,v) in pars_dict)...)
            
            parameters_level_i = merge(parameters_level_i, pars_tuple)
        end
        
        pars_dict = Dict(string("level_", i) => parameters_level_i,)
        pars_tuple = (; (Symbol(k) => v for (k,v) in pars_dict)...)
        
        parameters = merge(parameters, pars_tuple)
    end
    
    # initialize base models

    parameters_series = (series_1 = construct_parameters(), )
    for i in 2:length(unique(data.series))
            
        pars_dict = Dict(string("series_", i) => construct_parameters(),)
        pars_tuple = (; (Symbol(k) => v for (k,v) in pars_dict)...)
        
        parameters_series = merge(parameters_series, pars_tuple)
    end
    
    pars = (series = parameters_series,)
    parameters = merge(parameters, pars)
    
    return parameters
end 


function hierarcical_loss(u0,u1,dt,predict,loss,weights,parameters,series,groups)
    
    series_pars = parameters.series[keys(parameters.series)[series]]
    pred, eps = predict(u0,dt,series_pars)
    L = loss(pred,u1,series_pars)
    
    nlevels = length(groups)
    level_i = nlevels
    level_keys = keys(parameters)[1:(end-1)]
    
    for i in 1:nlevels
   
        group_key = keys(parameters[level_keys[level_i]])[groups[level_i]]
        group_pars = parameters[level_keys[level_i]][group_key]
        
        pred_i, eps = predict(u0,dt,group_pars)
        L += weights[level_i]*loss(pred_i,pred,group_pars)
        
        pred = pred_i
        level_i += -1
    end 
    
    return L
end


function hierarcical_regularization(parameters, loss, reg_pars)
    
    level_keys = keys(parameters)
    L = 0
    for level in level_keys
        groups = keys(parameters[level])
        for group in groups
            L+=loss(parameters[level][group], reg_pars)
        end
    end 
    
    return L
end


function get_series_info(data,nlevels)
    
    inds = collect(1:nrow(data))  
    series = unique(data.series)
    starts = [ inds[data.series .== i][1] for i in series ]
    
    len(i) = sum(data.series .== i)
    
    lengths = [ len(i) for i in series ]
    
    function get_dts(i)
        times = data.t[data.series .== i] 
        return times[2:end] .- times[1:(end-1)]
    end
    
    dts = [ get_dts(i) for i in series ]
    
    
    function groups(i)
        if nlevels > 1
            ind = inds[data.series .== i][1] 
            groups_ = data[ind,1:(nlevels-1) ]
            groups_ = Vector(groups_)
            return vcat([1],groups_)
        else
            return [1]
        end
    end
        
    groups_list = [ groups(i) for i in series ] 
    
    return starts,lengths,dts,groups_list
end 


function initialize_state_estimates(data,dims)
    dims = size(data)[2]-2
    uhat = zeros(size(data)[1],dims)
    return uhat
end 



function init_loss_function(nlevels,dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,weights)
    
    
    starts,lengths,dts,groups_list = get_series_info(dataframe,nlevels)

    function loss_function(parameters)
        
        level_keys = keys(parameters.process_model)[1:(end-1)]
        nlevels = length(level_keys)
        
        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end

        L_proc = 0
        i = 0
        for t0 in starts .-1
            i+=1
            for t in 2:lengths[i]
                u0 = parameters.uhat[:,t0+t-1]
                u1 = parameters.uhat[:,t0+t]
                series = i; groups = groups_list[i];dt = dts[i][t-1]
                L_proc += hierarcical_loss(u0,u1,dt,process_model.predict,process_loss.loss,weights,parameters.process_model,series,groups)
            end
        end 
        
        L_reg = hierarcical_regularization(parameters.process_model, process_regularization.loss, parameters.process_regularization)
        
        return L_reg + L_obs + L_proc
    end  

    
    return loss_function
        
end


mutable struct HierarchicalUDE
    times
    data
    data_frame
    parameters
    loss_function
    process_model
    process_loss 
    observation_model
    observation_loss 
    process_regularization
    observation_regularization
    constructor
end

function HierarchicalNODE(data,level_weights;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
    nlevels = length(level_weights)
    # convert data
    dataframe = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,(2+nlevels):size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.NODE_process(size(data)[1];hidden = hidden_units, seed = NN_seed)
    params_constructor() = deepcopy(process_model.parameters)
    process_parameters = init_nested_parameters(dataframe, params_constructor, nlevels; seed = 10)
    
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_loss_function(nlevels,dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,level_weights)
    
    constructor = (data) -> HierarchicalNODE(data,nlevels;
                            hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return HierarchicalUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end 



function HierarchicalNODESimplex(data,level_weights;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    nlevels = length(level_weights)
    # convert data
    dataframe = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,(2+nlevels):size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.NODE_process(size(data)[1];hidden = hidden_units, seed = NN_seed)
    params_constructor() = deepcopy(process_model.parameters)
    process_parameters = init_nested_parameters(dataframe, params_constructor, nlevels; seed = 10)
    
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.softmax()
    observation_loss = LossFunctions.softmaxMSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_loss_function(nlevels,dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,level_weights)
    
    
    constructor = (data) -> HierarchicalNODESimplex(data,nlevels;
                            hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return HierarchicalUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end 


function HierarchicalUDE(data,level_weights,derivs,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
    nlevels = length(level_weights)
    # convert data
    dataframe = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,(2+nlevels):size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    dims = size(data)[2]
    
    process_model = ProcessModels.ProcessModel(derivs,ComponentArray(initial_parameters),dims)
    params_constructor() = deepcopy(process_model.parameters)
    process_parameters = init_nested_parameters(dataframe, params_constructor, nlevels; seed = 10)
    
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2UDE(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_loss_function(nlevels,dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,level_weights)
    
    
    constructor = (data) -> HierarchicalUDE(data,level_weights,derivs,initial_parameters;
                            hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return HierarchicalUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
end



function forecast(UDE::HierarchicalUDE, u0::AbstractVector{}, times::AbstractVector{},parameters)
    estimated_map = (x,dt) -> UDE.process_model.predict(x,dt,parameters)[1]
    x = u0
    df = zeros(length(times),length(x)+1)
    df[1,:] = vcat([times[1]],UDE.observation_model.link(x,UDE.parameters.observation_model))
    for t in 2:length(times)
        dt = times[t]-times[t-1]
        x = estimated_map(x,dt)
        df[t,:] = vcat([times[t]],UDE.observation_model.link(x,UDE.parameters.observation_model))
    end 
    
    return df
end 



function plot_time_series(UDE::HierarchicalUDE)
    
    nlevels = length(keys(UDE.parameters.process_model))
    dims = size(UDE.data)[1]

    plts = [plot() for i in 1:dims]

    for d in 1:dims
        for group in unique(UDE.data_frame[:,nlevels-1])
            df = UDE.data_frame[UDE.data_frame[:,nlevels-1] .== group,:]
            plot!(plts[d],df.t,df[:,nlevels+d], c = d, label = "", width = 2)
        end
    end
    return plot(plts...)
end


function phase_plane(UDE::HierarchicalUDE,level_,group;u1s=-3:0.25:3, u2s=-3:0.25:3,T = 100, color = :roma)
    
    # caclaute time to evaluate 
    level_ = keys(UDE.parameters.process_model)[level_]
    parameters = UDE.parameters.process_model[level_]
    parameters = parameters[keys(parameters)[group]]

    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))
    
    # calcaulte u0s
    u0s = vcat([[u1,u2] for u1 in u1s for u2 in u2s])
    plt = plot()
    for u0 in u0s
        data = forecast(UDE,u0, times,parameters) 
        Plots.plot!(plt,data[:,2],data[:,3], label = "",
                            line_z = data[:,1], c = color)
    end 
    
    return plt
    
end 


function plot_time_series(UDE)
    
    nlevels = length(keys(UDE.parameters.process_model))
    dims = size(UDE.data)[1]

    plts = [plot() for i in 1:dims]

    for d in 1:dims
        for group in unique(UDE.data_frame[:,nlevels-1])
            df = UDE.data_frame[UDE.data_frame[:,nlevels-1] .== group,:]
            plot!(plts[d],df.t,df[:,nlevels+d], c = d, label = "", width = 2)
        end
    end
    return plts
end

function get_uhat_final(UDE::HierarchicalUDE)
    series = unique(UDE.data_frame.series)
    uhats = []
    for s in series
        inds = UDE.data_frame.series .== s
        uhat_final = UDE.parameters.uhat[:,inds]
        push!(uhats,uhat_final[:,end])
    end 
    return uhats
end 

# function forecast(UDE, u0::AbstractVector{}, times::AbstractVector{},parameters)
#     estimated_map = (x,dt) -> UDE.process_model.predict(x,dt,parameters)[1]
#     x = u0
#     df = zeros(length(times),length(x)+1)
#     df[1,:] = vcat([times[1]],UDE.observation_model.link(x,UDE.parameters.observation_model))
#     for t in 2:length(times)
#         dt = times[t]-times[t-1]
#         x = estimated_map(x,dt)
#         df[t,:] = vcat([times[t]],UDE.observation_model.link(x,UDE.parameters.observation_model))
#     end 
    
#     return df
# end 




function forcasts_(UDE::HierarchicalUDE,level_;T=10)
    level_key = keys(UDE.parameters.process_model)[level_]
    group_keys = keys(UDE.parameters.process_model[level_key])
    nlevels = length(keys(UDE.parameters.process_model))
    uhats = get_uhat_final(UDE)
    group_number = 0
    forcasts = []
    for group_key in group_keys
        group_number += 1
        group_parameters = UDE.parameters.process_model[level_key][group_key]
        group_index = 1:nrow(UDE.data_frame)
        if level_ > 1
            group_index = UDE.data_frame[:,level_-1] .== group_number
        end 
        group_data = UDE.data_frame[group_index,:]
        group_states = UDE.parameters.uhat[:,group_index]
        group_series = unique(group_data.series)

        for s in group_series
            inds = UDE.data_frame.series .== s
            times = UDE.times[inds]; dts = times[2:end] .- times[1:(end-1)]
            dt = sum(dts)/length(dts)
            times =times[end]:dt:(times[end] + T*dt )
            forcast_s = forecast(UDE, uhats[s], times,group_parameters)
            push!(forcasts,forcast_s)
        end 

    end 
    return forcasts

end




function plot_forecasts(UDE::HierarchicalUDE;T=10, level_ = 1)
    plts = plot_time_series(UDE)
    fcasts = forcasts_(UDE,level_,T = T)
    dims = length(plts)
    println(dims)
    for d in 1:dims
        for s in 1:length(fcasts)
            plot!(plts[d], fcasts[s][:,1], fcasts[s][:,d+1], linestyle = :dash, color = "grey", width = 1.5, label = "")
        end 
    end 
    return plts
end 



function predictions(UDE,level_)
    
    level_key = keys(UDE.parameters.process_model)[level_]
    group_keys = keys(UDE.parameters.process_model[level_key])
    nlevels = length(keys(UDE.parameters.process_model))

    group_number = 0
    inits = []; obs = []; preds = []
    for group_key in group_keys
        
        group_number += 1
        group_parameters = UDE.parameters.process_model[level_key][group_key]
        group_index = 1:nrow(UDE.data_frame)
        
        if level_ > 1
            group_index = UDE.data_frame[:,level_-1] .== group_number
        end 
        
        group_data = UDE.data_frame[group_index,:]
        group_states = UDE.parameters.uhat[:,group_index]
        group_series = unique(group_data.series)

        for s in group_series
            
            inds = UDE.data_frame.series .== s
            times = UDE.times[inds]; dts = times[2:end] .- times[1:(end-1)]
            dt = sum(dts)/length(dts)
            
            uhats = UDE.parameters.uhat[:,inds]
            inits_s = uhats[:,1:(end-1)]
            obs_s = uhats[:,2:end]
            preds_s = zeros(size(uhats[:,2:end]))
            
            for t in 1:(size(inits_s)[2])
                u0 = inits_s[:,t]
                u1 = obs_s[:,t]
                dt = UDE.times[t+1] - UDE.times[t]
                preds_s[:,t] = UDE.process_model.predict(u0,dt,group_parameters)[1]
            end
            
            push!(inits,inits_s);push!(obs,obs_s);push!(preds,preds_s)
            
        end 
        
    end 
    return inits, obs, preds

end


function plot_predictions(UDE,level_)
    
    inits, obs, preds = predictions(UDE,level_)
    
    plots = []
    for dim in 1:size(obs[1])[1]
        
        difs = obs[1][dim,:].-inits[1][dim,:]

        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot()
        scatter!(difs,preds[1][dim,:].-inits[1][dim,:], label = "", xlabel = "Observed change Delta hatu_t", 
                                ylabel = "Predicted change hatut - hatu_t")
        
        for i in 2:length(inits)
            difs = obs[i][dim,:].-inits[i][dim,:]
            mins = [xmin,difs[argmin(difs)]];xmin = mins[argmin(mins)]
            maxs = [xmax,difs[argmax(difs)]];xmax = maxs[argmin(maxs)]
            scatter!(difs,preds[i][dim,:].-inits[i][dim,:], label = "", xlabel = "Observed change Delta hatu_t", 
                                ylabel = "Predicted change hatut - hatu_t")
        end  
        plot!([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "45 degree")
        push!(plots, plt)
        
    end
        
    return plot(plots...)
    
end 
