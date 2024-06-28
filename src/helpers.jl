function init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    
    parameters = (uhat = zeros(size(data)) .+ 10^-3, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = observation_regularization.reg_parameters)
    
    return ComponentArray(parameters)
end



function init_loss(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
        # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += observation_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    
    # skips the prediction steps for intervals starting at a time in t_skip
    function loss_function(parameters,t_skip)
        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            if !(times[t-1] in t_skip)
                u1hat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model) 
                L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
            end
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += observation_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    return loss_function
end 



function process_data(data,time_column_name)
    
    time_alias_ = time_column_name
    try
        sort!(data,[time_alias_])
    catch e
        if isa(e, ArgumentError)
            error("time_column_name was not found in provided data. Please set the kwarg: time_column_name to the correct value")
        end
    end
    dataframe = sort!(data,[time_alias_])
    times = dataframe[:,time_alias_]

    times = dataframe[:,time_alias_] # time in colum 1
    N = length(times); dims = size(dataframe)[2] - 1; T = times[end] - times[1]
    varnames = names(dataframe)[names(dataframe).!=time_alias_]
    data = transpose(Matrix(dataframe[:,varnames]))
        
    return  N, dims, T, times, data, dataframe
end 


function find_NNparams_alias(nms)
    NN_alias = ["NNparams", "NN", "NN1", "NN2", "NN3", "NN4", "NN5", "Network", "NeuralNetwork", "NNparameters"] 
    ind = broadcast(nm -> nm in nms, NN_alias)
    if any(ind)
        return NN_alias[ind]  
    end
    error("Cannot find alias for neural network parameters. \n Please try: NNparams, NN, NN1, ... NN5, Network, NeuralNetwork, or NNparameters") 
end 

function series_indexes(dataframe,series_column_name,time_column_name)
    
    series = length(unique(dataframe[:,series_column_name]))
        
    inds = collect(1:nrow(dataframe))  
    starts = [inds[dataframe[:,series_column_name] .== i][1] for i in unique(dataframe[:,series_column_name])]
    lengths = [sum(dataframe[:,series_column_name] .== i) for i in unique(dataframe[:,series_column_name])]
    times = [dataframe[dataframe[:,series_column_name] .== i,time_column_name] for i in unique(dataframe[:,series_column_name])]  

    return series, inds, starts, lengths, times
end 


function process_multi_data(data, time_column_name, series_column_name)

    time_alias_ = time_column_name
    series_alias_ = series_column_name

    # collect series index names 
    try
        inds = levelcode.(CategoricalArray(data[:,series_alias_]))
    catch e
        if isa(e, ArgumentError)
            error("series_column_name was not found in provided data. Please set the kwarg: series_column_name to the correct value")
        end
    end

    labels_df = DataFrame(label = unique(data[:,series_alias_]),
                         index = levelcode.(CategoricalArray(unique(data[:,series_alias_])))
                        )
    

    data.series .= inds

    try
        sort!(data,["series",time_alias_])
    catch e
        if isa(e, ArgumentError)
            error("time_column_name was not found in provided data. Please set the kwarg: time_column_name to the correct value")
        end
    end
    dataframe = sort!(data,["series",time_alias_])
    times = dataframe[:,time_alias_]
    series =dataframe.series
    T = times[argmax(times)]
    
    N = length(times); dims = size(dataframe)[2] - 2
    inds_time = names(dataframe).!=time_alias_ 
    inds_series = (names(dataframe).!=series_alias_) .& (names(dataframe).!= "series")
    varnames = names(dataframe)[inds_time .& inds_series]
    data = transpose(Matrix(dataframe[:,varnames]))

    series, inds, starts, lengths, times_ls= series_indexes(dataframe,series_column_name,time_column_name)
    dims = size(data)[1]
    return N, T, dims, data, times,  dataframe, series, inds, starts, lengths, varnames, labels_df
end 


function process_multi_data2(data,time_column_name,series_column_name)
    
    time_alias_ = time_column_name
    series_alias_ = series_column_name

    inds = levelcode.(CategoricalArray(data[:,series_alias_]))
    data.series = inds

    dataframe = sort!(data,["series",time_alias_])
    inds_time = names(dataframe).!=time_alias_ 
    inds_series = (names(dataframe).!=series_alias_) .& (names(dataframe).!= "series")
    varnames = names(dataframe)[inds_time .& inds_series]
    
    times = dataframe[:,time_alias_]
    series =dataframe[:,series_alias_]
    T = times[argmax(times)]
    data_sets = [transpose(Matrix(dataframe[series.==s,varnames])) for s in unique(series)] 
    times = [Vector(dataframe[series.==s,time_alias_]) for s in unique(series)] 
    
    return data_sets, times
end 



