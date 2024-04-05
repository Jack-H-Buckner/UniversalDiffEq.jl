


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


function kfold_cv(model;k=10,leave_out=3)
    
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
                        
        gradient_decent!(model_i, t_skip)   

        try
            BFGS!(model_i, t_skip, verbose = false)
        catch
            println("BFGS failed running gradient_decent")
            gradient_decent!(model_i, t_skip, step_size = 0.01)                 
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



