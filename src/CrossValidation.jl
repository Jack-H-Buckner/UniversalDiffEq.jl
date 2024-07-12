

function rmse(testing_data, predicted_data)
    k = length(testing_data)
    MSE = 0
    for i in 1:k
        test = Matrix(testing_data[i][:,2:end])
        pred = Matrix(predicted_data[i][:,2:end])
        MSE += 1/k * sum((test .- pred).^2)/length(pred)
    end
    
    return sqrt(MSE)
end 


function kfold(model::UDE;k=10,leave_out=5,BFGS = false,step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500)
    
    # get final time
    data = model.data_frame
    T = size(data)[1]
    spacing = floor(Int, (T-leave_out)/k)
    starts = [floor(Int, 1+ spacing *i) for i in 0:(k-1)]
    t_skips = data[starts,model.time_column_name] 
    training_data = [vcat(data[1:t0,:],data[(t0+leave_out+1):end,:]) for t0 in starts]
    testing_data = [data[(t0+1):(t0+leave_out),:] for t0 in starts]
    
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]
    
    Threads.@threads for i in 1:k
        println("Training data set ", i)
        t_skip = [t_skips[i]] 

        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, t_skip, maxiter = maxiter, step_size = step_size)   

        if BFGS
            try
                BFGS!(model_i, t_skip, verbose = false)
            catch
                println("BFGS failed running gradient_descent")
                gradient_descent!(model_i, t_skip, maxiter = maxiter2, step_size = step_size2)                 
            end   
        else
            gradient_descent!(model_i, t_skip, maxiter = maxiter2, step_size = step_size2)  
        end
        # forecast
        u0 = model_i.parameters.uhat[:,starts[i]]
        times = testing_data[i][:,model.time_column_name]
        predicted_data = forecast(model_i, u0, t_skip[1], times)
        predicted_data= DataFrame(predicted_data,names(testing_data[i]))
            
        SE = copy(predicted_data)
        SE[:,2:end] .= (predicted_data[:,2:end] .- testing_data[i][:,2:end]).^2
        
        push!(standard_errors[Threads.threadid()], SE)
        push!(predicted[Threads.threadid()], predicted_data)             
    end 
    
    standard_error = standard_errors[1]
    predicted_data = predicted[1]         
    for i in 2:Threads.nthreads()
                            
        standard_error = vcat(standard_error,standard_errors[i])
        predicted_data = vcat(predicted_data,predicted[i])
                            
    end
    
    return training_data, testing_data, predicted_data, standard_error 
    
end 


"""
    kfold_cv(model::UDE; kwargs ...)
    
Runs K fold leave future out cross validation and returns the mean squared forecasting error and a plot to visualize the model fits.

...
# Arguments 
model - a UDE model object
k=10 - the number of testing and taining data sets to build
leave_out=5 - the number of data points to leave out in each testingdata set
BFGS = false - use the BFGS algorithm to train the model
step_size = 0.05 - step size for first run of the ADAM algorithm 
maxiter = 500 - maximum iterations for first trial of ADAM
step_size2 = 0.01 - step size for second iteration of ADAM, only used of BFGS is false
maxiter2 = 500 - step size for second iteration of ADAM, only used of BFGS is false
...
"""
function kfold_cv(model::UDE;k=10,leave_out=5,BFGS = false,step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500)
    
    training_data, testing_data, standard_errors, predicted_data = kfold(model;k=k,leave_out=leave_out,BFGS=BFGS,step_size=step_size,maxiter=maxiter,step_size2=step_size2,maxiter2=maxiter2)
    RMSE = rmse(testing_data, predicted_data)
    return RMSE                            
end



## leave future out cross validation
function leave_future_out(model::UDE; forecast_length = 10,  forecast_number = 10, spacing = 1, using_BFGS=false, 
                                step_size = 0.05, maxiter = 500, step_size2 = 0.01, maxiter2 = 500)
    
    # get final time
    data = model.data_frame
    T = length(data[:,model.time_column_name])
    start1 = T - forecast_length - spacing*(forecast_number-1)
    starts = [start1 + spacing *i for i in 0:(forecast_number-1)]
    training_data = [data[1:t0,:] for t0 in starts]
    testing_data = [data[t0:(t0+forecast_length),:] for t0 in starts]
    
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]
    
    Threads.@threads for i in 1:forecast_number

        println("Training data set ", i)
        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, step_size = step_size, maxiter = maxiter) 
           
        if using_BFGS
            try
                BFGS!(model_i,verbose = false)
            catch
                println("BFGS failed running gradient_descent")
                gradient_descent!(model_i, step_size = step_size2, maxiter = maxiter2)                
            end   
        else
            gradient_descent!(model_i, step_size = step_size2, maxiter = maxiter2) 
        end
                    
        # forecast
        u0 = model_i.parameters.uhat[:,end]
        times = testing_data[i][:,model.time_column_name]
        predicted_data = forecast(model_i, u0, times)
        predicted_data= DataFrame(predicted_data,names(testing_data[i]))
            
        SE = copy(predicted_data)
        SE[:,2:end] .= (predicted_data[:,2:end] .- testing_data[i][:,2:end]).^2
        
        push!(standard_errors[Threads.threadid()], SE)
        push!(predicted[Threads.threadid()], predicted_data)             
    end 
    
    standard_error = standard_errors[1]
    predicted_data = predicted[1]         
    for i in 2:Threads.nthreads()
                            
        standard_error = vcat(standard_error,standard_errors[i])
        predicted_data = vcat(predicted_data,predicted[i])
                            
    end
    
    return training_data, testing_data, standard_error, predicted_data
    
end 




                                        
"""
    leave_future_out_cv(model::UDE; kwargs ...)
    
Runs K fold leave future out cross validation and returns the mean squared forecasting error and a plot to visualize the model fits.

...
# Arguments 
model - the UDE model to test
forecast_length = 5 - The number of data points to forecast
forecast_number = 10 - The number of trainin gnad testing data sets to create
spacing = 2 - The number of data points to skip between the end of each new triaining data set
BFGS=false - use BFGS algorithm to train the model
step_size = 0.05 - step size for first run of the ADAM algorithm 
maxiter = 500 - maximum iterations for first trial of ADAM
step_size2 = 0.01 - step size for second iteration of ADAM, only used of BFGS is false
maxiter2 = 500 - step size for second iteration of ADAM, only used of BFGS is false
...
"""
function leave_future_out_cv(model::UDE;forecast_length = 5,  forecast_number = 10, spacing = 2, BFGS=false, 
                                step_size = 0.05, maxiter = 500, step_size2 = 0.01, maxiter2 = 500)
    
    training_data, testing_data, standard_errors, predicted_data = leave_future_out(model;forecast_length=forecast_length,forecast_number=forecast_number,using_BFGS=BFGS,spacing=spacing,step_size=step_size,maxiter=maxiter,step_size2 = step_size2, maxiter2 = maxiter2)
    RMSE = rmse(testing_data, predicted_data)
    return RMSE                            
end




# multiple time series

# find intervals of time spanned by at least one data set 
function t_in_intervals(t,tmins,tmaxs)
    @assert length(tmins) == length(tmaxs)
    for i in eachindex(tmins)
        if (t >= tmins[i]) & (t <= tmaxs[i])
            return true
        end  
    end
    return false
end 

function edge(tmins,tmaxs,dt)
    tmin = tmins[argmin(tmins)]
    tmax = tmaxs[argmax(tmaxs)]
    ts = tmin:dt:tmax
    test = broadcast(t -> t_in_intervals(t,tmins,tmaxs), ts)
    in_ = test[1]; edges = [tmin]
    for t in 2:length(ts)
        if in_ 
            if !test[t]
                push!(edges,ts[t])
            end
        else
            if test[t]
                push!(edges,ts[t-1])
            end
        end
        in_ = test[t]
    end
    return push!(edges,tmax)
end

function interval_lengths(edges)
    L = 0
    for i in 2:2:length(edges)
        L += edges[i] - edges[i-1]
    end
    return L
end

# K - number of intervals to leave out
# T - length of intervals left out 
function intervals(K,tmins,tmaxs,dt)
    edges = edge(tmins,tmaxs,dt)
    L = interval_lengths(edges)
    l = L/K
    lower = []
    for i in 2:2:length(edges)
        lower= vcat(lower,collect((edges[i-1]+dt):l:(edges[i]+dt)))
    end
    return lower[1:(end-1)]
end


max_value = x -> x[argmax(x)]



function multi_rmse(testing_data, predicted_data)
    k = length(testing_data)
    MSE = 0
    for i in 1:k
        test = Matrix(testing_data[i][:,3:end])
        pred = Matrix(predicted_data[i][:,3:end])
        MSE += 1/k * sum((test .- pred).^2)/length(pred)
    end
    
    return sqrt(RMSE)
end 

function kfold_cv(model::MultiUDE;k=10,leave_out=5,BFGS = false,step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500)
    
    # get final time
    df = model.data_frame
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(df,model.time_column_name,model.series_column_name)

    # split into training and testing data by leaving out
    tmins = [ times[t] for t in starts]
    tmaxs = [ times[starts[i] + lengths[i]-1] for i in eachindex(starts)]

    lower = intervals(k,tmins,tmaxs,10^-5)
    
    training_data = [df[(df[:,model.time_column_name] .<= t0) .| (df[:,model.time_column_name] .> (t0+leave_out)),:] for t0 in lower]
    testing_data = [df[(df[:,model.time_column_name] .> t0) .& (df[:,model.time_column_name] .<= (t0+leave_out)),:] for t0 in lower]
    
    # find time and series combos to skip  prediction step
    skips = [df[(df[:,model.time_column_name] .<= t0),:] for t0 in lower]
    skips = [combine(groupby(df,Symbol(model.series_column_name)), Symbol(model.time_column_name)  => max_value => Symbol(model.time_column_name)) for df in skips]
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]

    Threads.@threads for i in 1:k  
        print("k: ", i, "   ")
        skip = skips[i]
        
        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, skip, step_size = step_size, maxiter=maxiter, verbose = false)   

        if BFGS
            try
                BFGS!(model_i, skip, verbose = false)
            catch
                println("BFGS failed running gradient_decent")
                gradient_descent!(model_i, skip, step_size = 0.01)                 
            end  
        else
            gradient_descent!(model_i, skip, step_size = step_size2, maxiter=maxiter2)   
        end
        
        predictions =[]
        for s in skip[:,model.series_column_name]

            ind = model.series_labels[model.series_labels[:,"label"].==s,"index"]
            # get initial time point and state variables
            N_i, T_i, dims_i, data_i, times_i,  dataframe_i, series_i, inds_i, starts_i, lengths_i = process_multi_data(training_data[i],model.time_column_name,model.series_column_name)
            t0 = skip[skip[:,model.series_column_name] .== s,model.time_column_name][1]
            uhat_series = model_i.parameters.uhat[:,starts_i[ind][1]:(starts_i[ind][1]+lengths_i[ind][1]-1)]
            times_series = times_i[starts_i[ind][1]:(starts_i[ind][1]+lengths_i[ind][1]-1)]
            u0 = uhat_series[:,times_series .== t0]

            # get time from testing data
            times_test = testing_data[i][testing_data[i][:,model.series_column_name] .== s,model.time_column_name]
            predicted_data = forecast(model_i, u0[:,1], t0, times_test, s)

            predicted_data= DataFrame(predicted_data,names(testing_data[i])[1:(end-1)])

            # add time to predictions
            push!(predictions, predicted_data)
        end 
        
        # knit together predictions data frame
        predicted_data = predictions[1]
        for i in 2:length(predictions)
            predicted_data = vcat(predicted_data,predictions[i])
        end

        #SE = copy(predicted_data)
        #SE[:,3:end] .= (predicted_data[:,3:end] .- testing_data[i][:,3:end]).^2
        
        #push!(standard_errors[Threads.threadid()], SE)
        push!(predicted[Threads.threadid()], predicted_data)             
    end 
    
    #standard_error = standard_errors[1]
    predicted_data = predicted[1]         
    for i in 2:Threads.nthreads()
                            
        #standard_error = vcat(standard_error,standard_errors[i])
        predicted_data = vcat(predicted_data,predicted[i])
                            
    end
    # standard_error

    return  multi_rmse(testing_data, predicted_data)
    
end 



