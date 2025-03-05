
function forecast(UDE::UDE, test_data)
    #check_test_data_names(UDE.data_frame, test_data)
    u0 = UDE.parameters.uhat[:,end]
    N, dims, T, times, data, dataframe =  process_data(test_data,UDE.time_column_name)
    df =  forecast(UDE, u0, UDE.times[end], times)
    
    # get variable names 
    nms = names(UDE.data_frame)
    df = DataFrame(df,nms)

    return df
end


function leave_future_out_cv(model::UDE, training!, k)

    training_data = []
    testing_data = []
    data = model.data_frame
    for i in 1:k
        push!(training_data, data[1:(end-i),:])
        push!(testing_data, data[(end-i+1):end,:])
    end

    forecasts = Array{Any}(nothing, k)

    Threads.@threads for i in 1:k
        training_i = training_data[i]
        testing_i = testing_data[i]

        model_i = model.constructor(training_i)
        training!(model_i)
        
        forecast_i = forecast(model_i, testing_i)
        forecast_i.horizon .= forecast_i[:,model.time_column_name] .- minimum(forecast_i[:,model.time_column_name]) .+ 1 
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
    leave_future_out(model, training!, k; path = false)

Runs a leave future out cross validation on the UDe model `model` using the training routine `train!` with `k`  folds. 
if a path to a csv file is provided usign the path key word then the raw testing data and forecasts will be saved for each fold.  
    
The funtion returns three data frames. The first contains an estimate of the mean aboslue error of the forecasts and assocaited standard error as a fuction of the forecast horizon (1 to k time steps into the future).
The second is containds the forecasting errors seperated by variable and the third contains the raw testing and forecasting data. 
"""
function leave_future_out(model::UDE, training!, k; path = false)

    training, testing, forecasts = leave_future_out_cv(model, training!, k)

    df_summary2, df_summary, df = summarize_leave_future_out(training, testing, forecasts; time_column_name = "time")
    if !path
        return df_summary2, df_summary, df 
    end
    
    CSV.write(path, df)

    return df_summary2, df_summary, df
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



function leave_future_out_cv(model::MultiUDE, training!, k)

    training_data = []
    testing_data = []
    data = model.data_frame
    series = unique(data.series)
    for i in 1:k
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

    for i in 1:k
        training_i = training_data[i]
        testing_i = testing_data[i]

        model_i = model.constructor(training_i)
        training!(model_i)
        
        forecast_i = forecast_data(model_i, testing_i)

        forecast_i.horizon .= 0
        for series in unique(forecast_i[:,model.series_column_name])
            inds = forecast_i[:,model.series_column_name] .== series
            forecast_i.horizon[inds] .= forecast_i[inds,model.time_column_name] .- minimum(forecast_i[inds,model.time_column_name]) .+ 1 
        end 
        forecasts[i] = forecast_i
    end
    
    return training_data, testing_data, forecasts
end 



function leave_future_out(model::MultiUDE, training!, k; path = false)

    training, testing, forecasts = leave_future_out_cv_2(model, training!, k)

    df_summary2, df_summary, df = summarize_leave_future_out(training, testing, forecasts; time_column_name = "time")
    if !path
        return df_summary2, df_summary, df 
    end
    
    CSV.write(path, df)

    return df_summary2, df_summary, df
end 