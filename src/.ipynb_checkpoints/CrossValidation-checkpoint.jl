


function kfold_rmse(standard_errors)
    N = length(standard_errors)
    acc = zeros(size(Matrix(standard_errors[1])[:,2:end]))
        
    for i in 1:N
                                    
        acc .+= standard_errors[i][:,2:end] ./ N
                                    
    end 

    acc .= sqrt.(acc)
    rMSE = DataFrame(hcat(collect(1:size(acc)[1]), acc), names(standard_errors[1]))
    
    return rMSE
end 


function kfold_cv(model::UDE;k=10,leave_out=3,BFGS = false,maxiter = 500)
    
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
        
        t_skip = [t_skips[i]] 

        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, t_skip, maxiter = maxiter)   

        if BFGS
            try
                BFGS!(model_i, t_skip, verbose = false)
            catch
                println("BFGS failed running gradient_descent")
                gradient_descent!(model_i, t_skip, step_size = 0.01)                 
            end   
        end
        gradient_descent!(model_i, t_skip, maxiter = maxiter, step_size = 0.01)  
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
    
    SEs = kfold_rmse(standard_error)[:,2:end]
    rMSE = sum(Matrix(SEs).^2)
    return rMSE, kfold_rmse(standard_error)
    
end 

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





function kfold_cv(model::MultiUDE;k=10,leave_out=3,BFGS=false,maxiter=500)
    
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

    for i in 1:k #Threads.@threads 
        print("k: ", i, "   ")
        skip = skips[i]
        
        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, skip, maxiter=maxiter, verbose = false)   

        if BFGS
            try
                BFGS!(model_i, skip, verbose = false)
            catch
                println("BFGS failed running gradient_decent")
                gradient_descent!(model_i, skip, step_size = 0.01)                 
            end  
        else
            gradient_descent!(model_i, skip, step_size = 0.01,maxiter=maxiter)   
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
    return predicted_data,testing_data
    
end 

function root_mean_squared_errors(standard_errors)
    acc = zeros(size(standard_errors[1][:,3:end]))
    N = length(standard_errors)
    for i in eachindex(standard_errors)
        acc .+= standard_errors[i][:,3:end]./N
    end
    return sqrt(sum(acc)./length(acc)), sqrt.(acc) 
end


## leave future out cross validation




function leave_future_out(model; forecast_length = 10,  forecast_number = 10, spacing = 1, step_size = 0.05, maxiter = 500, step_size2 = 0.01, maxiter2 = 500,using_BFGS=false)
    
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
        print(i, " ")
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
                gradient_descent!(model_i, step_size = 0.25*step_size, maxiter = 2*maxiter)                 
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


function leave_future_out_mse(standard_errors)
    N = length(standard_errors)
    acc = zeros(size(Matrix(standard_errors[1])[:,2:end]))
        
    for i in 1:N
                                    
        acc .+= standard_errors[i][:,2:end] ./ N
                                    
    end 
    
    MSE = DataFrame(hcat(collect(1:size(acc)[1]), acc), names(standard_errors[1]))
    
    return MSE
end 

                                        
function plot_leave_future_out_cv(data,testing_data, standard_errors , predicted_data)
    plts1 = []
    plts2 = []
    for i in 2:(size(data)[2])
        p1=Plots.scatter(data.t,data[:,i],label = "")
        p2 = Plots.plot([0,data.t[end]],[0.0,0.0], color = "black", linestyle = :dash,label = "")
        for j in 1:length(testing_data)
            Plots.plot!(p1,predicted_data[j].t,predicted_data[j][:,i], linestyle = :dash, width= 2,label = "") 
            Plots.plot!(p2,standard_errors[j].t,standard_errors[j][:,i], width= 2,label = "")
        end 
        push!(plts1,p1)
        push!(plts2,p2)
    end
    p1 = plot(plts1...)
    p2 = plot(plts2...)
    return p1,p2
end 

"""
    leave_future_out_cv(model; forecast_length = 10,  K = 10, spacing = 1, step_size = 0.05, maxiter = 500)
    
Runs K fold leave future out cross validation and returns the mean squared forecasting error and a plot to visualize the model fits.

...
# Arguments 
model - the UDE model to test
forecast_length - the number of steps to calcualte the forecast performance (default 10).
K - the number of forecast tests to run (default 10).
spacing - the number of data points to skip between testing sets (default 1).
step_size - step size parameter for the gradient descent algorithm (default 0.05).
maxiter - number of iterations for gradient descent (default 500).. 
...
"""
function leave_future_out_cv(model; forecast_length = 10,  K = 10, spacing = 1, step_size = 0.05, maxiter = 500)
    training_data, testing_data, standard_errors, predicted_data = leave_future_out(model;forecast_length=forecast_length,forecast_number=K,spacing=spacing,step_size=step_size,maxiter=maxiter)
    MSE = leave_future_out_mse(standard_errors)
    plt = plot_leave_future_out_cv(model.data_frame,testing_data, standard_errors , predicted_data)
    return MSE, plt                             
end


function forecast_simulation_tests(N,simulator,model;train_fraction=0.9,step_size = 0.05, maxiter = 500)
    
    # get test data size and set accumulator
    sizeSE = forecast_simulation_SE(simulator,model,1;
                    train_fraction=train_fraction,step_size = step_size, maxiter = 1)
    
    MSE_acc = [zeros(size(sizeSE)) for i in 1:Threads.nthreads()]
    
    # run simulation tests with multithreading
    Threads.@threads for seed in 1:N
            
        MSE_acc[Threads.threadid()] .+= forecast_simulation_SE(simulator,model,seed;train_fraction=train_fraction,step_size=step_size,maxiter=maxiter)./N
            
    end 
        
    MSE = MSE_acc[1]
    for i in 2:Threads.nthreads()
        MSE .+= MSE_acc[i]
    end 
                
    T = size(MSE)[1]
    MSE = DataFrame(MSE ,:auto)
    MSE.t = 1:T
                
    return MSE
  
end 


