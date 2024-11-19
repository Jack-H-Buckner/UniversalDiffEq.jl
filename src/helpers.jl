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


function init_loss(data,times, state_variable_transform,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            uhat = state_variable_transform(parameters.uhat[:,t])
            yhat = observation_model.link(uhat,parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 =  state_variable_transform(parameters.uhat[:,t-1])
            u1 =  state_variable_transform(parameters.uhat[:,t])
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
            uhat = state_variable_transform(parameters.uhat[:,t])
            yhat = observation_model.link(uhat,parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 =  state_variable_transform(parameters.uhat[:,t-1])
            u1 =  state_variable_transform(parameters.uhat[:,t])
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
    labels_df = DataFrame(label = unique(data[:,series_alias_]),
                         index = levelcode.(CategoricalArray(unique(data[:,series_alias_])))
                        )
    
    inds = levelcode.(CategoricalArray(data[:,series_alias_]))
    data.series .= inds

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

function find_time_alias(nms)
    time_alias = ["T", "t","time", "Time", "times", "Times"] 
    ind = broadcast(nm -> nm in nms, time_alias)
    if any(ind)
        return time_alias[ind][1]   
    end
    error("time_column_name was not found in provided data. Please set the kwarg: time_column_name to the correct value")
end

function find_series_alias(nms)
    series_alias = ["location", "site", "Series", "series"] 
    ind = broadcast(nm -> nm in nms, series_alias)
    if any(ind)
        return series_alias[ind][1]   
    end
    error("series_column_name was not found in provided data. Please set the kwarg: series_column_name to the correct value")
end 

function find_values_alias(nms)
    value_alias = ["values", "x", "value", "Value", "Values", "reading", "Reading"] 
    ind = broadcast(nm -> nm in nms, value_alias)
    if any(ind)
        return value_alias[ind][1]   
    end
    error("value_column_name was not found in provided data. Please set the kwarg: variable_column_name to the correct value\n for wide-formatted covariate data, set value_column_name and value_column_name to nothing")
end 

function find_variable_alias(nms)
    variable_alias = ["variables", "variable", "Variable", "Variables", "measurement"] 
    ind = broadcast(nm -> nm in nms, variable_alias)
    if any(ind)
        return variable_alias[ind][1]   
    end
    error("variable_column_name was not found in provided data. Please set the kwarg: variable_column_name to the correct value\n for wide-formatted covariate data, set value_column_name and variable_column_name to nothing")
end 

function check_column_names(data::DataFrame; time_column_name = nothing, series_column_name = nothing)

    global col_names = [time_column_name, series_column_name]

    if(time_column_name !== nothing)
        try data[:,time_column_name]
        catch e
            if(isa(e, ArgumentError))
                global col_names[1] = find_time_alias(names(data))
                @warn("Found unexpected value for time_column_name:" * col_names[1] * ", It is reccomended to set kwarg: time_column_name to match your data")
            else
                throw(e)
            end
        end
    end
    if(series_column_name !== nothing)
        try data[:,series_column_name]
        catch e
            if(isa(e, ArgumentError))
                global col_names[2] = find_series_alias(names(data))
                @warn("Found unexpected value for series_column_name:" * col_names[2] * ", It is reccomended to set kwarg: series_column_name to match your data")
            else
                throw(e)
            end
        end

    end
    return col_names
end

function check_column_names(data::DataFrame, covariates::DataFrame; time_column_name = nothing, series_column_name = nothing, value_column_name = nothing, variable_column_name = nothing)
    global col_names = [time_column_name, series_column_name, value_column_name, variable_column_name]

    if(time_column_name !== nothing)
        try data[:,time_column_name]
        catch e
            if(isa(e, ArgumentError))
                global col_names[1] = find_time_alias(names(data))
                @warn("Found unexpected value for time_column_name:" * col_names[1] * ", It is reccomended to set kwarg: time_column_name to match your data")
            else
                throw(e)
            end
        end

        try covariates[:,col_names[1]]
        catch e
            if(isa(e, ArgumentError))
                error("time_column_name not found in covariate data. Make sure time column names are consistent across all dataframes")
            else
                throw(e)
            end
        end

    end
    if(series_column_name !== nothing)
        try data[:,series_column_name]
        catch e
            if(isa(e, ArgumentError))
                global col_names[2] = find_series_alias(names(data))
                @warn("Found unexpected value for series_column_name:" * col_names[2] * ", It is reccomended to set kwarg: series_column_name to match your data")
            else
                throw(e)
            end
        end
        
        try covariates[:,col_names[2]]
        catch e
            if(isa(e, ArgumentError))
                error("series_column_name not found in covariate data. Make sure time column names are consistent across all dataframes")
            else
                throw(e)
            end
        end
    end
    if(value_column_name !== nothing)
        try covariates[:,value_column_name]
        catch e
            if(isa(e, ArgumentError))   
                global col_names[3] = find_value_alias(names(covariates))
                @warn("Found unexpected value for value_column_name:" * col_names[3] * ", It is reccomended to set kwarg: value_column_name to match your data")
            else
                throw(e)
            end
        end
    end
    if(variable_column_name !== nothing)
        try covariates[:,variable_column_name]
        catch e
            if(isa(e, ArgumentError))
                
                global col_names[4] = find_variable_alias(names(covariates))
                @warn("Found unexpected value for variable_column_name:" * col_names[4] * ", It is reccomended to set kwarg: variable_column_name to match your data")
            else
                throw(e)
            end
        end
    end
    return col_names
end

function check_test_data_names(modelData::DataFrame, test_data::DataFrame)

    unfound_names = []
    for column_name in names(modelData)
        try test_data[:, column_name]
        catch e
            if(isa(e, ArgumentError))
                push!(unfound_names, column_name)
            else
                throw(e)
            end
        end
    end
    if(length(unfound_names) > 0)
        err_msg = "Error: Please ensure that all column names match provided training data. The following columns were not found in provided testing data: \n"
        col_names = ""
        for name in unfound_names
            col_names = col_names * name * " "
        end
        error(err_msg * col_names)
    end
end

function melt(data; id_vars = nothing)

    vars = names(data)
    inds = broadcast(nm -> !(nm in id_vars), vars)
    vars = vars[inds]
    
    long_data = DataFrame(zeros(length(vars)*nrow(data), length(id_vars)),id_vars)
    long_data.variable .= repeat(["variable"], length(vars)*nrow(data))
    long_data.value .= repeat([0.0], length(vars)*nrow(data))
    n = 0
    for i in 1:nrow(data)
        for j in eachindex(vars)
            n += 1
            id_values = Vector(data[i,id_vars])
            variable = vars[j]
            value = data[i,variable]
            long_data[n,id_vars] .= id_values
            long_data.variable[n] = variable
            long_data.value[n] = value
        end 
    end 

    return long_data
end 

function max_(x)
    x[argmax(x)]
end

function min_(x)
    x[argmin(x)]
end

function mean_(x)
    sum(x)/length(x)
end