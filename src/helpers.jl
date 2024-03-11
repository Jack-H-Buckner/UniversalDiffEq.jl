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
        L_reg = process_regularization.loss(parameters.process_model.NN,parameters.process_regularization)
        L_reg += observation_regularization.loss(parameters.process_model.NN,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    
    return loss_function
end 


time_alias = ["T", "t","time", "Time", "times", "Times"] 
time_alias_ind = [:T, :t,:time,:Time,:times,:Times] 


function find_time_alias(nms)
    ind = broadcast(nm -> nm in nms, time_alias)
    if any(ind)
        return time_alias[ind][1]   
    end
    print("Cannot find column for time ")
    throw(error()) 
end 



function process_data(data)
    
    time_alias_ = find_time_alias(names(data))
    dataframe = sort!(data,[time_alias_])
    times = dataframe[:,time_alias_]

    times = dataframe.t # time in colum 1
    N = length(times); dims = size(dataframe)[2] - 1; T = times[end] - times[1]
    varnames = names(dataframe)[names(dataframe).!=time_alias_]
    data = transpose(Matrix(dataframe[:,varnames]))
        
    return  N, dims, T, times, data, dataframe
end 


series_alias = ["location", "site", "series"] 


function find_series_alias(nms)
    ind = broadcast(nm -> nm in nms, series_alias)
    if any(ind)
        return series_alias[ind][1]   
    end
    print("Cannot find column for time ")
    throw(error()) 
end 


series_alias = ["NNparams", "NN", "NN1", "NN2", "NN3", "NN4", "NN5", "Network", "NeuralNetwork", "NNparameters"] 


function find_NNparams_alias(nms)
    ind = broadcast(nm -> nm in nms, series_alias)
    if any(ind)
        return series_alias[ind]  
    end
    print("Cannot find column for time ")
    throw(error()) 
end 


function process_multi_data(data)
    
    time_alias_ = find_time_alias(names(data))
    series_alias_ = find_series_alias(names(data))
    
    dataframe = sort!(data,[series_alias_,time_alias_])
    
    times = dataframe[:,time_alias_]
    series =dataframe[:,time_alias_]
    T = times[argmax(times)]
    
    times = dataframe.t # time in colum 1
    N = length(times); dims = size(dataframe)[2] - 2
    inds_time = names(dataframe).!=time_alias_ 
    inds_series = names(dataframe).!=series_alias_
    varnames = names(dataframe)[inds_time .| inds_series]
    data = transpose(Matrix(dataframe[:,varnames]))
                
    return N, T, dims, data, dataframe, starts, length, times
end 


function series_indexes(dataframe)
    
    series = length(unique(dataframe.series))
        
    inds = collect(1:nrow(dataframe))  
    starts = [inds[dataframe.series .== i][1] for i in unique(dataframe.series)]
    lengths = [sum(dataframe.series .== i) for i in unique(dataframe.series)]
    times = [dataframe.t[dataframe.series .== i] for i in unique(dataframe.series)]  
end 

