


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


function kfold_cv(model::UDE;k=10,leave_out=3)
    
    # get final time
    data = model.data_frame
    T = length(data.t)
    spacing = floor(Int, (T-leave_out)/k)
    starts = [floor(Int, 1+ spacing *i) for i in 0:(k-1)]
    t_skips = data.t[starts] 
    training_data = [vcat(data[1:t0,:],data[(t0+leave_out+1):end,:]) for t0 in starts]
    testing_data = [data[(t0+1):(t0+leave_out),:] for t0 in starts]
    
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]
    
    Threads.@threads for i in 1:k
        
        t_skip = [t_skips[i]] 

        model_i = model.constructor(training_data[i])
                        
        gradient_descent!(model_i, t_skip)   

        try
            BFGS!(model_i, t_skip, verbose = false)
        catch
            println("BFGS failed running gradient_descent")
            gradient_descent!(model_i, t_skip, step_size = 0.01)                 
        end   
         
        # forecast
        u0 = model_i.parameters.uhat[:,starts[i]]
        times = testing_data[i].t
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
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(df)

    # split into training and testing data by leaving out
    tmins = [ times[t] for t in starts]
    tmaxs = [ times[starts[i] + lengths[i]-1] for i in eachindex(starts)]

    lower = intervals(k,tmins,tmaxs,10^-5)
    
    training_data = [df[(df.t .<= t0) .| (df.t .> (t0+leave_out)),:] for t0 in lower]
    testing_data = [df[(df.t .> t0) .& (df.t .<= (t0+leave_out)),:] for t0 in lower]
    
    # find time and series combos to skip  prediction step
    skips = [df[(df.t .<= t0),:] for t0 in lower]
    skips = [combine(groupby(df, :series), :t => max_value => :t) for df in skips]

    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]

    for i in 1:k #Threads.@threads 
        
        skip = skips[i]
        model_i = model.constructor(training_data[i])
                        
        gradient_decent!(model_i, skip,maxiter=maxiter)   

        if BFGS
            try
                BFGS!(model_i, skip, verbose = false)
            catch
                println("BFGS failed running gradient_decent")
                gradient_decent!(model_i, skip, step_size = 0.01)                 
            end  
        else
            gradient_decent!(model_i, skip, step_size = 0.01,maxiter=maxiter)   
        end
        
        predictions =[]
        for s in skip.series

            # get initial time point and state variables
            N_i, T_i, dims_i, data_i, times_i,  dataframe_i, series_i, inds_i, starts_i, lengths_i = process_multi_data(training_data[i])

            t0 = skip.t[skip.series .== s][1]
            uhat_series = model_i.parameters.uhat[:,starts_i[s]:(starts_i[s]+lengths_i[s]-1)]
            times_series = times_i[starts_i[s]:(starts_i[s]+lengths_i[s]-1)]
            u0 = uhat_series[:,times_series .== t0]

            # get time from testing data
            times_test = testing_data[i].t[testing_data[i].series .== s]
            predicted_data = forecast(model_i, u0[:,1], t0, times_test, s)
            predicted_data= DataFrame(predicted_data,names(testing_data[i]))

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
