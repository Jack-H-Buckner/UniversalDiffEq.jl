
function forecast_data(UDE::UDE, test_data::DataFrame)
    #check_test_data_names(UDE.data_frame, test_data)
    u0 = UDE.parameters.uhat[:,end]
    N, dims, T, times, data, dataframe =  process_data(test_data,UDE.time_column_name)
    df =  forecast(UDE, u0, UDE.times[end], times)
    
    # get variable names 
    nms = names(UDE.data_frame)
    df = DataFrame(df,nms)

    return df
end



function leave_future_out_cv(model::UDE, training!, k, skip )

    training_data = []
    testing_data = []
    data = model.data_frame
    skip = round(Int,skip)
    for i in skip:skip:(skip*k)
        push!(training_data, data[1:(end-i),:])
        push!(testing_data, data[(end-i+1):end,:])
    end

    forecasts = Array{Any}(nothing, k)

    Threads.@threads for i in 1:k
        
        training_i = training_data[i]
        testing_i = testing_data[i]

        model_i = model.constructor(training_i)
        training!(model_i)
        
        forecast_i = forecast_data(model_i, testing_i)
     
        forecast_i.horizon .= forecast_i[:,model.time_column_name] .- maximum(training_i[:,model.time_column_name])
        forecasts[i] =  forecast_i
    end
    
    return training_data, testing_data, forecasts
end 



function knit_leave_future_out_data(training_data, testing_data, forecasts; time_column_name = "time")

    # number of folds
    k = length(training_data)

    # get data at index 1
    training_i = training_data[1] 
    testing_i = testing_data[1] 
    forecast_i = forecasts[1]

    # rename variables
    nms = names(testing_i)
    nms = nms[nms .!= time_column_name]
    testing_i = stack(testing_i, nms, value_name = "testing")
    forecast_i = stack(forecast_i, nms, value_name = "forecast")
    # join data sets 
    df = innerjoin(testing_i,forecast_i,on = [time_column_name,"variable"])

    df.fold .= 1
    
    for i in 2:k

        # Get data at index i
        training_i = training_data[i] 
        testing_i = testing_data[i] 
        forecast_i = forecasts[i]
        
        # rename variables
        testing_i = stack(testing_i, nms, value_name = "testing")
        forecast_i = stack(forecast_i, nms, value_name = "forecast")

        # join data sets 
        df_i = innerjoin(testing_i,forecast_i,on = [time_column_name,"variable"])
        df_i.fold .= i
        df = vcat(df,df_i)

    end
    
    return df
end 


function summarize_leave_future_out(training, testing, forecasts; time_column_name = "time")

    df = knit_leave_future_out_data(training, testing, forecasts; time_column_name = time_column_name)

    SE(x) = std(x)/sqrt(length(x))

    # calcautle mean absolute error for each variable 
    df_summary = combine(groupby(df, [:horizon, :variable]),
        [:testing, :forecast] => ((x,y) -> mean(abs.(x .- y))) => :mean_absolute_error,
        [:testing, :forecast] => ((x,y) -> SE(x.-y)) => :standard_error_of_MAE,
    )

    # calculate mean abosulte error summed over each variable 
    df_summary2 = combine(groupby(df, [:horizon]),
        [:testing, :forecast] => ((x,y) -> mean(abs.(x .- y))) => :mean_absolute_error,
        [:testing, :forecast] => ((x,y) -> SE(x.-y)) => :standard_error_of_MAE,
    )

    return df_summary2, df_summary, df
end 


"""
    leave_future_out(model::UDE, training!, k; kwargs... )

Runs a leave future out cross validation on the UDe model `model` using the training routine `train!` with `k`  folds. 
if a path to a csv file is provided usign the path key word then the raw testing data and forecasts will be saved for each fold.  
    
The funtion returns three data frames. The first contains an estimate of the mean aboslue error of the forecasts and assocaited standard error as a fuction of the forecast horizon (1 to k time steps into the future).
The second and third are returned in a named tuple with two elements `horizon_by_var` and `raw`. The data frame `horizon_by_var` is containds the forecasting errors seperated by variable and the data frame `raw` contains the raw testing and forecasting data. 
If the model is trained on multiple time series the named tupe will include a third data frame `horizon_by_var_by_series`.

    # kwargs

- `path`: the path to a directory to save the output dataframes, defaults to Null
- `skip`: the number of observations to skip in each fold, defaults to 1
"""
function leave_future_out(model::UDE, training!, k; skip = 1,path = false)

    training, testing, forecasts = leave_future_out_cv(model, training!, k, skip)

    df_summary2, df_summary, df = summarize_leave_future_out(training, testing, forecasts; time_column_name = model.time_column_name)
    println(typeof(path))
    if typeof(path) != String
        return df_summary2, (horizon_by_var = df_summary, raw = df)
    end
    
    CSV.write(path, df)

    return df_summary2, (horizon_by_var = df_summary, raw = df)
end 



function marginal_likelihood(UDE, test_data, Pν, Pη; α = 10^-3, β = 2,κ = 0)


    # observaiton model is identity function
    N, dims, T, times, data, dataframe = UniversalDiffEq.process_data(test_data,UDE.time_column_name)

    L = size(data)[1]

    Pν=UniversalDiffEq.errors_to_matrix(Pν, L)
    Pη=UniversalDiffEq.errors_to_matrix(Pη, L)
    H = Matrix(I,L,L)

    y = data
    times = UDE.times
    f = (u,t,dt,p) -> UDE.process_model.predict(u,t,dt,p)[1]

    ll = UniversalDiffEq.ukf_likelihood(y,times,f,UDE.parameters.process_model,H,Pν,Pη,L,α,β,κ)

    return ll
end


function kfold_cv(model::UDE, training!, k)

    training_data = []
    testing_data = []
    t_skip = []
    data = model.data_frame
    T = floor(Int,length(model.times)/k)
    t = 0
    for i in 1:k
        push!(t_skip,t)
        push!(testing_data, data[(t+1):(t+T),:])
        push!(training_data, vcat(data[1:t,:],data[(t+T+1):end,:]))
        t += T
    end

    forecasts = Array{Any}(nothing, k)

    for i in 1:k
        training_i = training_data[i]
        testing_i = testing_data[i]

        model_i = model.constructor(training_i)
        Pν, Pη = training!(model_i,t_skip[i])
        
        forecasts[i] = marginal_likelihood(model_i, testing_i, Pν, Pη)
    end
    
    return sum(forecasts)
end 






function forecast_data(UDE::MultiUDE, test_data::DataFrame)
    check_test_data_names(UDE.data_frame, test_data)
    N, T, dims, test_data, test_times,  test_dataframe, test_series, inds, test_starts, test_lengths, labs = process_multi_data(test_data,UDE.time_column_name,UDE.series_column_name)
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, labs = process_multi_data(UDE.data_frame,UDE.time_column_name,UDE.series_column_name)

    series = 1
    time = times[starts[series]:(starts[series]+lengths[series]-1)]
    dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
    uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
    series_ls = unique(test_dataframe[:,UDE.series_column_name])
    
    println(UDE.series_column_name," ")

    df = forecast(UDE, uhat[:,end], time[end], test_dataframe[test_dataframe[:,UDE.series_column_name] .== series_ls[1], UDE.time_column_name], series_ls[1])

    for series in 2:length(series_ls)

        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]

        test_time = test_times[test_starts[series]:(test_starts[series]+test_lengths[series]-1)]
        test_dat = test_data[:,test_starts[series]:(test_starts[series]+test_lengths[series]-1)]

        df = vcat(df,forecast(UDE, uhat[:,end], time[end], test_dataframe[test_dataframe[:,UDE.series_column_name] .== series_ls[series], UDE.time_column_name], series_ls[series]))
    end

    return df

end


function leave_future_out_cv(model::MultiUDE, training!, k, skip)

    training_data = []
    testing_data = []
    data = model.data_frame
    series = unique(data.series)
    skip = round(Int,skip)

    for i in skip:skip:(skip*k)
        data_series = data[data.series .== series[1],:]
        train = data_series[1:(end-i),:]
        test = data_series[(end-i+1):(end),:]
        for s in 2:length(series)
            data_series = data[data.series .== series[s],:]
            train_i = data_series[1:(end-i),:]
            test_i = data_series[(end-i+1):(end),:]
            train = vcat(train,train_i)
            test = vcat(test,test_i)
        end
        push!(training_data,train)
        push!(testing_data,test)
    end

    forecasts = Array{Any}(nothing, k)

    Threads.@threads for i in 1:k
        training_i = training_data[i]
        testing_i = testing_data[i]

        model_i = model.constructor(training_i)
        training!(model_i)
        
        forecast_i = forecast_data(model_i, testing_i)

        forecast_i.horizon .= 0.0
        for series in unique(forecast_i[:,model.series_column_name])
            inds = forecast_i[:,model.series_column_name] .== series
            inds_training = training_i[:,model.series_column_name] .== series
            forecast_i.horizon[inds] .= forecast_i[inds,model.time_column_name] .- maximum(training_i[inds_training,model.time_column_name]) 
        end 
        forecasts[i] = forecast_i
    end
    
    return training_data, testing_data, forecasts
end 



function knit_leave_future_out_data_multi(training_data, testing_data, forecasts; time_column_name = "time",series_column_name = "series")

    # number of folds
    k = length(training_data)

    # get data at index 1
    training_i = training_data[1] 
    testing_i = testing_data[1] 
    forecast_i = forecasts[1]

    # rename variables
    nms = names(testing_i)
    nms = nms[nms .!= time_column_name]
    nms = nms[nms .!= series_column_name]

    testing_i = stack(testing_i, nms, value_name = "testing")
    forecast_i = stack(forecast_i, nms, value_name = "forecast")
    # join data sets 
    df = innerjoin(testing_i,forecast_i,on = [time_column_name,series_column_name,"variable"])

    df.fold .= 1
    
    for i in 2:k

        # Get data at index i
        training_i = training_data[i] 
        testing_i = testing_data[i] 
        forecast_i = forecasts[i]
        
        # rename variables
        testing_i = stack(testing_i, nms, value_name = "testing")
        forecast_i = stack(forecast_i, nms, value_name = "forecast")

        # join data sets 
        df_i = innerjoin(testing_i,forecast_i,on = [time_column_name,series_column_name,"variable"])
        df_i.fold .= i
        df = vcat(df,df_i)

    end
    
    return df
end 


function summarize_leave_future_out_multi(training, testing, forecasts; time_column_name = "time", series_column_name = "series")

    df = knit_leave_future_out_data_multi(training, testing, forecasts; time_column_name = time_column_name, series_column_name=series_column_name)

    SE(x) = std(x)/sqrt(length(x))

    # calcautle mean absolute error for each variable 
    df_summary = combine(groupby(df, [:horizon, :variable, Symbol(series_column_name)]),
        [:testing, :forecast] => ((x,y) -> mean(abs.(x .- y))) => :mean_absolute_error,
        [:testing, :forecast] => ((x,y) -> SE(x.-y)) => :standard_error_of_MAE,
    )

    df_summary2 = combine(groupby(df, [:horizon, :variable]),
        [:testing, :forecast] => ((x,y) -> mean(abs.(x .- y))) => :mean_absolute_error,
        [:testing, :forecast] => ((x,y) -> SE(x.-y)) => :standard_error_of_MAE,
    )

    # calculate mean abosulte error summed over each variable 
    df_summary3 = combine(groupby(df, [:horizon]),
        [:testing, :forecast] => ((x,y) -> mean(abs.(x .- y))) => :mean_absolute_error,
        [:testing, :forecast] => ((x,y) -> SE(x.-y)) => :standard_error_of_MAE,
    )

    detailed_summaries = (horizon_by_var = df_summary2, horizon_by_var_by_sereis = df_summary, raw = df)
    return df_summary3, detailed_summaries 
end 


function leave_future_out(model::MultiUDE, training!, k ; skip = 1, path = false)

    training, testing, forecasts = leave_future_out_cv(model, training!, k, skip)

    df_summary3, detailed_summaries = summarize_leave_future_out_multi(training, testing, forecasts; time_column_name = model.time_column_name, series_column_name=model.series_column_name)
    if !path
        return df_summary3, detailed_summaries
    end
    
    CSV.write(path, detailed_summaries.raw)

    return df_summary3, detailed_summaries
end 


### leave sites out 
# CRPS
# 1/M sum(abs(xi-y)) - 1/2M^2 sum(sum(abs(xi-xj)))
# energy score?

# CRQS
# 1/M sum((xi-y)^2) - 1/2M^2 sum(sum((xi-xj)^2))

# Train model
# forecast with the model for each sites
# calcualte CRPS / CRQS using each forecast as a sample
# smooth using obs and proc errors from training?
function leave_site_out_i(model::MultiUDE, training!,i)

    training_inds = model.data_frame[:,model.series_column_name] .== i 
    testing_inds = model.data_frame[:,model.series_column_name]  .!= i

    training_i = model.data_frame[training_inds,:]
    testing_i = model.data_frame[testing_inds,:]

    model_i = model.constructor(training_i)
    training!(model_i)

    
end 


function leave_site_out_sample(model, training!,i)

    training_inds = model.data_frame[:,model.series_column_name] .!= i 
    testing_inds = model.data_frame[:,model.series_column_name]  .== i

    training_i = model.data_frame[training_inds,:]
    testing_i = model.data_frame[testing_inds,:]

    model_i = model.constructor(training_i)
    training!(model_i)
    inds =  unique(model_i.data_frame[:,model.series_column_name])

    preds = []
    for j in inds
        testing_i[:,model.series_column_name] .= j
        push!(preds,UniversalDiffEq.predict(model_i,testing_i))
    end
    tmin = minimum(testing_i[:,model.time_column_name])
    testing_i = testing_i[testing_i[:,model.time_column_name] .> tmin,:]

    return testing_i, training_i, preds 
end 


function energy_score(testing,preds,model)

    M = length(preds)
    T = length(testing[:,1])

    # remove time and sereis column names 
    testing = testing[:,names(testing).!=model.time_column_name]
    testing = testing[:,names(testing).!=model.series_column_name]
    for i in 1:M
        preds[i] = preds[i][:,names(preds[i]).!=model.time_column_name]
        preds[i] = preds[i][:,names(preds[i]).!=model.series_column_name]
    end 
    score = 0
    for t in 1:T
        for i in 1:M
            pred_i = Vector(preds[i][t,1:end])
            test_i = Vector(testing[t,1:end])
            score += 1/M * sum(sqrt.((pred_i .- test_i).^2))
            for j in 1:M
                pred_j = Vector(preds[j][t,1:end])
                score += -1/(2*M^2) * sum(sqrt.((pred_i - pred_j).^2))
            end
        end
    end 

    return score/T
end 

using StatsBase
"""
    leave_site_out(model, training!; kwargs...)

Calcualtes the average forecasting accuracy of the model's one-step-ahead predictions leaving the full time series for
one site out at a time. Each time step in the testing data set is forecast using the model paramters for each of the 
sites in the training data set. The pserfoance of these forecasts is quantified using the energy score which is a 
multivariate extension of the continuous rank probability score (CRPS). This metric reduces to the mean absolue 
forecasting error when an identical model is fit to each site in the training data set. 

...

# Keyword Arguments
- `sites`: the number of sites to leave out, defualts to the number of sites in the training data set.  
"""
function leave_site_out(model, training!; sites = length(unique(model.data_frame[:,model.series_column_name])))
    
    site_ls = sample(unique(model.data_frame[:,model.series_column_name]),sites,replace=false)
    scores = zeros(length(site_ls))

    Threads.@threads for i in site_ls
        testing, training, preds  =  leave_site_out_sample(model, training!,i)
        scores[i] = energy_score(testing,preds,model)
    end

    return sum(scores)/length(scores)
end


### Leave blocks of future data out. 

function filter_data(UDE::UDE, test_data::DataFrame,H, Pν, Pη,L,α,β,κ)

    #check_test_data_names(UDE.data_frame, test_data)
    u0 = UDE.parameters.uhat[:,end]
    N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
    ll = ukf_likelihood(y,u0,times,f,parameters.UDE.process_model,H,Pν,Pη,L,α,β,κ)

    return ll
end

function update_filter_options!(options, UDE)
    # set up observaiton and error matrices 
    L = size(UDE.data)[1]
    H = Matrix(1.0I, L, L)
    Pν = errors_to_matrix(0.1, L)
    if :process_error in keys(options)
      Pν = errors_to_matrix(options.process_error, L)
    end

    Pη = errors_to_matrix(0.1, L)
    if :observation_error in keys(options)
      Pη = errors_to_matrix(options.observation_error, L)
    else
      println("Using the default observation error (0.1) for training, which may be inappropriate for your data. \n Please see the documentation for guidance of these model parameters. \n You can use `loss_options = (observation_error = ...,)` to specify their values.")
    end

    # add scalar values to options
    new_options = ComponentArray(options)
    options = ComponentArray((α = 10^-3, β = 2,κ = 0))
    inds  = broadcast(i -> !(keys(new_options)[i] in [:process_error,:observation_error]), 1:length(keys(new_options)))
    keys_ = keys(new_options)[inds]
    options[keys_] .= new_options[keys_]

    # add matrices to options 
    options.H = H
    options.Pν = Pν
    options.Pη = Pη
    return options
end 

# used to get SEs for model skill metrics 
function standard_error(x)
    std(x)/sqrt(length(x))
end

# Summarizes cross validation results
function summarize_cv_results(model,preds,obs,times)

    # bind together resutls form each fold 
    preds_cat = reduce(hcat, preds)
    obs_cat = reduce(hcat, obs)
    times_cat = reduce(vcat, times)

    # get variable names from the model object
    nms = names(model.data_frame)
    nms = nms[nms .!= model.time_column_name]

    # build data frames from predictions
    df_preds = DataFrame(preds_cat', nms)
    df_preds[:,model.time_column_name] = times_cat
    df_preds = stack(df_preds, nms, value_name = "predicted")

    # build data frame from observations
    df_obs = DataFrame(obs_cat', nms)
    df_obs[:,model.time_column_name] = times_cat
    df_obs = stack(df_obs, nms, value_name = "observed")

    # join predictions and observations calcualte errors 
    df = innerjoin(df_obs,df_preds,on = [model.time_column_name,"variable"])
    df.AE=abs.(df.observed.-df.predicted)
    df.SE=(df.observed.-df.predicted).^2

    # group by variable and summarize
    gdf = groupby(df, :variable)
    summary = combine(gdf, 
        :SE => mean => :MSE,
        :SE => standard_error => :MSE_SE,
        :AE => mean => :MAE,
        :AE => standard_error => :MAE_SE,
        [:observed,:predicted] => cor => :correlation,
        nrow => :count
    )

    # return summary and raw data for user defined summaries 
    return summary, df
end 

"""
    leave_future_out_predict(model, training!, n_per_fold, k_folds)

This method performs leave future out cross validation, by leaving blocks of data off of the 
end of the training data set for each fold. The folds contain `n_per_fold` data points which
are used to evaluate the models predictive skill by predicting one-step-ahead between each 
observation in the testing fold.  The model predictive skill is evaluated by comparine the 
observed chagnes between data points to the change forecast by the UDE. The UDE's predicting skill 
is evaluated by calcaulteing the mean suared error (MSE), the mean absolute error (MAE), and the 
correlation between observed and predicted values. The standard errors are also estiamted for MSE 
(MSE_SE) and MAE (MAE_SE). The model skill in qunatified seperately for each state variable incuded
in the model. The summarized model performacne is reture din a data frame along with the raw observed
and predicted values used to calcualte the performance metrics. 

### Arguments 
- model: A UDE model object 
- training!: A function that implemnet the UDE training process, updating the model in place.
- n_per_fold: The number of data points in each fold.
- k_folds: The number of folds.

### Value
1. A data frame with summarized model preformance 

| variable | MSE | MSE_SE |	MAE	| MAE_SE | correlation | count |
|----------|-----|--------|-----|--------|-------------|-------|
| X1       | *   | *      | *   | *      | *           | *     |
| X2       | *   | *      | *   | *      | *           | *     |

2. A data frame with the raw observed and predicted values 

| time     | variable | observed | predicted | AE | SE |
|----------|----------|----------|-----------|----|----|
| T        | X1       | *        | *         | *  | *  |
| T-1      | X1       | *        | *         | *  | *  |
| T-2      | X1       | *        | *         | *  | *  |
|...       | ...      | ...      | ...       |... |... |
| T        | X2       | *        | *         | *  | *  |
| T-1      | X2       | *        | *         | *  | *  |
| T-2      | X2       | *        | *         | *  | *  |
| ...      | ...      | ...      | ...       |... |... |

"""
function leave_future_out_predict(model::UDE, training!, n_per_fold::Int, k_folds::Int )

    training_data = []
    testing_data = []
    data = model.data_frame
    for i in n_per_fold:n_per_fold:(n_per_fold*k_folds)
        push!(training_data, data[1:(end-i),:])
        push!(testing_data, data[(end-i):(end-i+n_per_fold),:]) # including the last data point for one step ahead predictions
    end

    predictions = Array{Any}(nothing, k_folds)
    observations = Array{Any}(nothing, k_folds)
    times = Array{Any}(nothing, k_folds)

    Threads.@threads for i in 1:k_folds
        
        training_i = training_data[i]
        testing_i = testing_data[i]

        model_i = model.constructor(training_i)
        training!(model_i)

        # include first over lapping data point to use for first prediction 
        inits,obs_i,preds_i = UniversalDiffEq.predictions(model_i, testing_i)
        delta_obs = obs_i .- inits
        delta_pred = preds_i .- inits
        predictions[i] =  delta_pred 
        observations[i] =  delta_obs
        times[i] = testing_i[2:end,model.time_column_name]
    end
    
    return summarize_cv_results(model,predictions,observations,times)
end 



function leave_future_out_filter(model::UDE, training!, method, n_per_fold::Int, k_folds::Int,  options  = NamedTuple() )

    training_data = []
    testing_data = []
    data = model.data_frame
    for i in n_per_fold:n_per_fold:(n_per_fold*k)
        push!(training_data, data[1:(end-i),:])
        push!(testing_data, data[(end-i):(end-i+n_per_fold),:]) # including the last data point for one step ahead predictions
    end

    predictions = Array{Any}(nothing, k)

    Threads.@threads for i in 1:k
        
        training_i = training_data[i]
        testing_i = testing_data[i]
        model_i = model.constructor(training_i)
        Pν, Pη = training!(model_i)

        # omit first overlapping data point IC come for the model 
        options = update_filter_options!(options, model_i)
        ll_i = filter_data(model_i, testing_i[2:end,:], options.H, Pν, Pη, options.L, options.α, options.β, options.κ)
        predictions[i] =  ll_i
    end
    
    return sum(predictions)/length(predictions)
end 
