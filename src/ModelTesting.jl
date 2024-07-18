

function get_final_state(UDE::UDE)
    return UDE.parameters.uhat[:,end]
end 

function get_final_state(UDE::BayesianUDE;summarize = true,ci = 95)
    uhats = reduce(hcat,[UDE.parameters[i].uhat[:,end] for i in 1:length(UDE.parameters)])
    if summarize
        return [percentile(uhats[i,:],[(100-ci)/2,50,ci+(100-ci)/2]) for i in 1:size(uhats,1)]   
    else
        return uhats
    end
end 




"""
    plot_state_estimates(UDE::UDE)

Plots the value of the state variables estimated by the UDE model. 
"""
function plot_state_estimates(UDE::UDE)
    
    plots = []
    
    for dim in 1:size(UDE.data)[1]
    
        # Calculate NRMSE for the current dimension
        N = length(UDE.parameters.uhat[dim,:])
        NRMSE = sqrt(sum((UDE.data[dim,:] .-UDE.parameters.uhat[dim,:]).^2)/N)/std(UDE.data[dim,:])

        plt=Plots.scatter(UDE.times,UDE.data[dim,:], label = "observations")
        
        Plots.plot!(UDE.times,UDE.parameters.uhat[dim,:], color = "grey", label= "estimated states",
                    xlabel = "time", ylabel = string("x", dim))
        
        xmax = UDE.times[argmax(UDE.times)]
        xmin = UDE.times[argmin(UDE.times)]
        text_x = 0.1*(xmax-xmin)+xmin

        ymax = UDE.data[dim,argmax(UDE.data[dim,:])]
        ymin = UDE.data[dim,argmin(UDE.data[dim,:])]
        text_y = 0.7*(ymax-ymin)+ymin 

        Plots.annotate!(text_x, text_y, text("NRMSE = $(round(NRMSE, digits=3))", :left, 10))

       
        push!(plots, plt)
    end 
            
    return plot(plots...)       
end 



"""
    print_parameter_estimates(UDE::UDE)

prints the value of the known dynamcis parameters. 
"""
function print_parameter_estimates(UDE::UDE)
    println("Estimated parameter values: ")
    i = 0
    for name in keys(UDE.parameters.process_model)
        i += 1
        if name == "NN"
        elseif name == :NN
        else
            println(name, ": ", round(UDE.parameters.process_model[name], digits = 3))
        end
    end 
end



"""
    get_parameters(UDE::UDE)

Returns model parameters.
"""
function get_parameters(UDE::UDE)
    return UDE.parameters.process_model
end


"""
    get_NN_parameters(UDE::UDE)

Returns value weights and biases of the neural network 
"""
function get_NN_parameters(UDE::UDE)
    return UDE.parameters.process_model.NN
end



"""
    get_right_hand_side(UDE::UDE)

Returns the right hand side of the differential equation (or difference equation) used to build the process model.

The fuction will take the state vector `u` and time `t` if the model does not include covariates. If covaraites are included the arguments are the state vector `u` , covariates vector `x`, and time `t`
"""
function get_right_hand_side(UDE::UDE)
    pars = get_parameters(UDE)
    if UDE.X == 0
        return (u,t) -> UDE.process_model.right_hand_side(u,pars,t)
    else
        return (u,x,t) -> UDE.process_model.right_hand_side(u,x,pars,t)
    end  
end 



function get_predict(UDE::UDE)
    pars = get_parameters(UDE)
    (u,t,dt) -> UDE.process_model.predict(u,t,dt,pars)
end 


function predictions(UDE::UDE)
 
    inits = UDE.parameters.uhat[:,1:(end-1)]
    obs = UDE.parameters.uhat[:,2:end]
    preds = UDE.parameters.uhat[:,2:end]
    
    for t in 1:(size(inits)[2])
        u0 = inits[:,t]
        u1 = obs[:,t]
        dt = UDE.times[t+1] - UDE.times[t]
        preds[:,t] = UDE.process_model.predict(u0,UDE.times[t],dt,UDE.parameters.process_model)[1]
    end

    return inits, obs, preds
end 

function predictions(UDE::BayesianUDE;summarize = true,ci = 95)
 
    inits = [UDE.parameters[i].uhat[:,1:(end-1)] for i in 1:length(UDE.parameters)]
    obs = [UDE.parameters[i].uhat[:,2:end] for i in 1:length(UDE.parameters)]
    preds = [UDE.parameters[i].uhat[:,2:end] for i in 1:length(UDE.parameters)]
    
    for i in 1:length(UDE.parameters)
        for t in 1:(size(inits[1])[2])
            u0 = inits[i][:,t]
            u1 = obs[i][:,t]
            dt = UDE.times[t+1] - UDE.times[t]
            preds[i][:,t] = UDE.process_model.predict(u0,UDE.times[t],dt,UDE.parameters[i].process_model)[1]
            preds[i][:,t] = UDE.process_model.predict(u0,UDE.times[t],dt,UDE.parameters[i].process_model)[1]
        end
    end

    if summarize
        inits = reduce((x,y) -> cat(x,y,dims = 3),inits)
        inits = [percentile(inits[i,j,:],50) for i in 1:size(inits,1), j in 1:size(inits,2)]

        obs = reduce((x,y) -> cat(x,y,dims = 3),obs)
        obs = [percentile(obs[i,j,:],50) for i in 1:size(obs,1), j in 1:size(obs,2)]

        preds = reduce((x,y) -> cat(x,y,dims = 3),preds)
        preds = [percentile(preds[i,j,:],[(100-ci)/2,50,ci+(100-ci)/2]) for i in 1:size(preds,1), j in 1:size(preds,2)]
    end

    return inits, obs, preds
end 



function predictions(UDE::UDE,test_data::DataFrame)
    check_test_data_names(UDE.data_frame, test_data)
    N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
    inits = data[:,1:(end-1)]
    obs = data[:,2:end]
    preds = data[:,2:end]
    
    for t in 1:(size(inits)[2])
        u0 = inits[:,t]
        u1 = obs[:,t]
        dt = times[t+1] - times[t]
        preds[:,t] = UDE.process_model.predict(u0,UDE.times[t],dt,UDE.parameters.process_model)[1]
    end

    return inits, obs, preds
end 

function predictions(UDE::BayesianUDE,test_data::DataFrame;summarize = true,ci = 95)
    check_test_data_names(UDE.data_frame, test_data)
    N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
    inits = data[:,1:(end-1)]
    obs = data[:,2:end]
    preds = [data[:,2:end] for i in 1:length(UDE.parameters)]
    
    for i in 1:length(UDE.parameters)
        for t in 1:(size(inits)[2])
            u0 = inits[:,t]
            u1 = obs[:,t]
            dt = UDE.times[t+1] - UDE.times[t]
            preds[i][:,t] = UDE.process_model.predict(u0,UDE.times[t],dt,UDE.parameters[i].process_model)[1]
        end
    end

    if summarize
        preds = reduce((x,y) -> cat(x,y,dims = 3),preds)
        preds = [percentile(preds[i,j,:],[(100-ci)/2,50,ci+(100-ci)/2]) for i in 1:size(preds,1), j in 1:size(preds,2)]
    end

    return inits, obs, preds
end 


function predict(UDE::UDE,test_data::DataFrame;df = true)
    check_test_data_names(UDE.data_frame, test_data)
    inits, obs, preds = predictions(UDE,test_data)
    if df
        N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
        names = vcat(["t"],[string("x",i) for i in 1:dims])
        return DataFrame(Array(vcat(times',preds)'),names)
    else
        return preds
    end

end 

function predict(UDE::BayesianUDE,test_data::DataFrame;summarize = true,ci = 95,df = true)
    check_test_data_names(UDE.data_frame, test_data)
    inits, obs, preds = predictions(UDE,test_data,summarize=summarize,ci=ci)
    if df
        meanForecast = [preds[i,j][2] for i in 1:size(preds,1), j in 1:size(preds,2)]
        lowerForecast = [preds[i,j][1] for i in 1:size(preds,1), j in 1:size(preds,2)]
        upperForecast = [preds[i,j][3] for i in 1:size(preds,1), j in 1:size(preds,2)]

        names = vcat(["t"],[string("x",i,"_lower",ci) for i in 1:dims],[string("x",i,"_median") for i in 1:dims],[string("x",i,"_higher",ci) for i in 1:dims])
        return DataFrame(Array(vcat(times[1:5]',lowerForecast,meanForecast,upperForecast)'),names)
    else
        return preds
    end
end 

"""
    plot_predictions(UDE::UDE)

Plots the correspondence between the observed state transitons and the predicitons for the model `UDE`. 

"""
function plot_predictions(UDE::UDE)
 
    inits, obs, preds = predictions(UDE)
    
    plots = []
    for dim in 1:size(obs,1)

        difs = obs[dim,:].-inits[dim,:]

        xmax = difs[argmax(difs)]
        xmin = difs[argmin(difs)]

        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1", legend_position = :topleft)
        duhat = preds[dim,:].-inits[dim,:]
        scatter!(difs,duhat,color = "white", label = "", xlabel = "Observed change", ylabel = "Predicted change")

        N = length(difs)      
        NRMSE = sqrt(sum((difs .- duhat).^2)/N)/std(difs)
    
        
        

        text_x = 0.5*(xmax-xmin)+xmin
        text_y = 0.1*(xmax-xmin)+xmin 

        Plots.annotate!(text_x, text_y, text("NRMSE = $(round(NRMSE, digits=3))", :left, 10))

        
        push!(plots, plt)
            
    end
        
    return plot(plots...)
end

function plot_predictions(UDE::BayesianUDE;ci=95)
 
    inits, obs, preds = predictions(UDE,summarize = true,ci=ci)
    
    plots = []
    for dim in 1:size(obs[1],1)
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1")
        scatter!(difs,reduce(hcat,preds[dim,:])[2,:][dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed median change Delta hatu_t", 
                                ylabel = "Predicted median change hatut - hatu_t")
        push!(plots, plt)
            
    end
        
    return plot(plots...)
end


"""
    plot_predictions(UDE::UDE, test_data::DataFrame)

Plots the correspondence between the observed state transitons and observed transitions in the test data. 
"""
function plot_predictions(UDE::UDE,test_data::DataFrame)
    check_test_data_names(UDE.data_frame, test_data)
    inits, obs, preds = predictions(UDE,test_data)
    
    plots = []
    for dim in 1:size(obs,1)
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1")
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change", 
                                ylabel = "Predicted change")
        push!(plots, plt)
            
    end
        
    return plot(plots...)
end

function plot_predictions(UDE::BayesianUDE,test_data::DataFrame;ci=95)
    check_test_data_names(UDE.data_frame, test_data)
    inits, obs, preds = predictions(UDE,test_data,summarize = true,ci=ci)
    
    plots = []
    for dim in 1:size(obs[1],1)
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1")
        scatter!(difs,reduce(hcat,preds[dim,:])[2,:][dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed median change Delta hatu_t", 
                                ylabel = "Predicted median change hatut - hatu_t")
        push!(plots, plt)
            
    end
        
    return plot(plots...)
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


"""
    forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})

predicitons from the trained model `UDE` starting at `u0` saving values at `times`. Assumes `u0` is the value at time `times[1]`
"""
function forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})
    
    uhats = UDE.parameters.uhat
    
    umax = mapslices(max_, uhats, dims = 2);umax=reshape(umax,length(umax))
    umin = mapslices(min_, uhats, dims = 2);umin=reshape(umin,length(umin))
    umean = mapslices(mean_, uhats, dims = 2);umean=reshape(umean,length(umean))
    
    
    #estimated_map = (x,dt) -> UDE.process_model.forecast(x,dt,UDE.parameters.process_model,umax,umin,umean)
    estimated_map = (x,t,dt) -> UDE.process_model.forecast(x,t,dt,UDE.parameters.process_model,umax,umin,umean)
    
    
    x = u0
    df = zeros(length(times),length(x)+1)
    df[1,:] = vcat([times[1]],x)
    
    for t in 2:length(times)
        dt = times[t]-times[t-1]
        x = estimated_map(x,times[t-1],dt)
        df[t,:] = vcat([times[t]],x)
    end 
    
    return df
end 


function forecast(UDE::BayesianUDE, u0::AbstractVector{}, times::AbstractVector{};summarize = true, ci = 95)
    dfs = zeros(length(UDE.parameters),length(times),length(x)+1)

    for i in 1:length(UDE.parameters)
        uhats = UDE.parameters[i].uhat
    
        umax = mapslices(max_, uhats, dims = 2);umax=reshape(umax,length(umax))
        umin = mapslices(min_, uhats, dims = 2);umin=reshape(umin,length(umin))
        umean = mapslices(mean_, uhats, dims = 2);umean=reshape(umean,length(umean))
    
    
        #estimated_map = (x,dt) -> UDE.process_model.forecast(x,dt,UDE.parameters.process_model,umax,umin,umean)
        estimated_map = (x,t,dt) -> UDE.process_model.forecast(x,t,dt,UDE.parameters[i].process_model,umax,umin,umean)
    
    
        x = u0
        df[i,1,:] = vcat([times[1]],x)
    
        for t in 2:length(times)
            dt = times[t]-times[t-1]
            x = estimated_map(x,times[t-1],dt)
            df[i,t,:] = vcat([times[t]],x)
        end 
    end


    if summarize
        dfs = [percentile(dfs[:,i,j],[(100-ci)/2,50,ci+(100-ci)/2]) for i in 1:10, j in 1:3] 
    end

    return dfs
end 

# """
#     forecast(UDE::UDE, u0::AbstractVector{}, t0::Real, times::AbstractVector{})

# predicitons from the trained model `UDE` starting at `u0` saving values at `times`. Assumes `u0` occurs at time `t0` and `times` are all larger than `t0`.
# """
function forecast(UDE::UDE, u0::AbstractVector{}, t0::Real, times::AbstractVector{})
    
    @assert all(times .> t0) "t0 is greater than the first time point in times"
    uhats = UDE.parameters.uhat
    
    umax = mapslices(max_, uhats, dims = 2);umax=reshape(umax,length(umax))
    umin = mapslices(min_, uhats, dims = 2);umin=reshape(umin,length(umin))
    umean = mapslices(mean_, uhats, dims = 2);umean=reshape(umean,length(umean))
    
    
    #estimated_map = (x,dt) -> UDE.process_model.forecast(x,dt,UDE.parameters.process_model,umax,umin,umean)
    estimated_map = (x,t,dt) -> UDE.process_model.forecast(x,t,dt,UDE.parameters.process_model,umax,umin,umean)
    
    
    x = u0
    df = zeros(length(times),length(x)+1)
    
    for t in eachindex(times)
        dt = times[t] - t0
        tinit = t0
        if t > 1
            dt = times[t]-times[t-1]
            tinit = times[t-1]
        end
        x = estimated_map(x,tinit,dt)
        df[t,:] = vcat([times[t]],x)
    end 
    
    return df
end 



function forecast(UDE::BayesianUDE, u0::AbstractVector{}, t0::Real, times::AbstractVector{};summarize = true, ci = 95)
    
    @assert all(times .> t0) "t0 is greater than the first time point in times"
    x = u0
    x = u0
    dfs = zeros(length(UDE.parameters),length(times),length(x)+1)

    for i in 1:length(UDE.parameters)
        uhats = UDE.parameters[i].uhat
    
        umax = mapslices(maximum, uhats, dims = 2);umax=reshape(umax,length(umax))
        umin = mapslices(minimum, uhats, dims = 2);umin=reshape(umin,length(umin))
        umean = mapslices(mean, uhats, dims = 2);umean=reshape(umean,length(umean))
    
    
        #estimated_map = (x,dt) -> UDE.process_model.forecast(x,dt,UDE.parameters.process_model,umax,umin,umean)
        estimated_map = (x,t,dt) -> UDE.process_model.forecast(x,t,dt,UDE.parameters[i].process_model,umax,umin,umean)
    
    
        x = u0

        for t in eachindex(times)
            dt = times[t] - t0
            tinit = t0
            if t > 1
                dt = times[t]-times[t-1]
                tinit = times[t-1]
            end
            x = estimated_map(x,tinit,dt)
            dfs[i,t,:] = vcat([times[t]],x)
        end 
    end
    
    if summarize
        dfs = [percentile(dfs[:,i,j],[(100-ci)/2,50,ci+(100-ci)/2]) for i in 1:length(times), j in 1:(length(x)+1)] 
    end

    return dfs
end 



"""
    plot_forecast(UDE::UDE, T::Int)

Plots the models forecast up to T time steps into the future from the last observation.  
"""
function plot_forecast(UDE::UDE, T::Int)
    u0 = UDE.parameters.uhat[:,end]
    dts = UDE.times[2:end] .- UDE.times[1:(end-1)]
    dt = sum(dts)/length(dts)
    times = UDE.times[end]:dt:(UDE.times[end] + T*dt )
    df = forecast(UDE, u0, times)
    plots = []
    for dim in 2:size(df,2)
        plt = plot(df[:,1],df[:,dim],color = "grey", linestyle=:dash, label = "forecast",
                    xlabel = "Time", ylabel = string("x", dim))
        plot!(UDE.times,UDE.data[dim-1,:],c=1, label = "data",
                    xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end 
    return plot(plots...), plots
end 

function plot_forecast(UDE::BayesianUDE, T::Int;ci = 95)
    u0 = reduce((x,y) -> cat(x,y,dims = 3),[UDE.parameters[i].uhat[:,end] for i in 1:length(UDE.parameters)])
    u0 = mean(u0,dims = 3)
    u0 = vec(u0[:,:,1])
    u0 = vec(u0[:,:,1])
    dts = UDE.times[2:end] .- UDE.times[1:(end-1)]
    dt = sum(dts)/length(dts)
    times = UDE.times[end]:dt:(UDE.times[end] + T*dt )
    df = forecast(UDE, u0, times,summarize = true, ci = ci)
    df = forecast(UDE, u0, times,summarize = true, ci = ci)
    meanForecast = [df[i,j][2] for i in 1:size(df,1), j in 1:size(df,2)]
    lowerForecast = [df[i,j][1] for i in 1:size(df,1), j in 1:size(df,2)]
    upperForecast = [df[i,j][3] for i in 1:size(df,1), j in 1:size(df,2)]
    plots = []
    for dim in 2:size(df,2)
        plt = plot(meanForecast[:,1],meanForecast[:,dim],ribbon = (meanForecast[:,dim]-lowerForecast[:,dim],upperForecast[:,dim]-meanForecast[:,dim]),
                    color = "grey", linestyle=:dash, label = "forecast",
                    xlabel = "Time", ylabel = string("x", dim))
        plot!(UDE.times,UDE.data[dim-1,:],c=1, label = "data",
                    xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end 
    return plot(plots...), plots
end 


"""
    plot_forecast(UDE::UDE, test_data::DataFrame)

Plots the models forecast over the range of the test_data along with the value of the test data.   
"""
function plot_forecast(UDE::UDE, test_data::DataFrame)
    check_test_data_names(UDE.data_frame, test_data)
    u0 = UDE.parameters.uhat[:,end]
    N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
    df = forecast(UDE, u0, UDE.times[end], times)
    plots = []
    for dim in 2:size(df)[2]
        plt = plot(df[:,1],df[:,dim],color = "grey", linestyle=:dash, label = "forecast", xlabel = "Time", ylabel = string("x", dim))
        scatter!(UDE.times,UDE.data[dim-1,:],c=1, label = "data", xlabel = "Time", ylabel = string("x", dim))
        scatter!(times,data[dim-1,:],c=2, label = "data", xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end 
    return plot(plots...), plots
end 

function plot_forecast(UDE::BayesianUDE, test_data::DataFrame;ci = 95)
    check_test_data_names(UDE.data_frame, test_data)
    u0 = reduce((x,y) -> cat(x,y,dims = 3),[UDE.parameters[i].uhat[:,end] for i in 1:length(UDE.parameters)])
    u0 = mean(u0,dims = 3)
    u0 = vec(u0[:,:,1])
    N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
    df = forecast(UDE, u0, UDE.times[end], times,summarize = true, ci = ci)
    u0 = vec(u0[:,:,1])
    N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
    df = forecast(UDE, u0, UDE.times[end], times,summarize = true, ci = ci)
    meanForecast = [df[i,j][2] for i in 1:size(df,1), j in 1:size(df,2)]
    lowerForecast = [df[i,j][1] for i in 1:size(df,1), j in 1:size(df,2)]
    upperForecast = [df[i,j][3] for i in 1:size(df,1), j in 1:size(df,2)]
    plots = []
    for dim in 2:size(df)[2]
        plt = plot(meanForecast[:,1],meanForecast[:,dim],ribbon = (meanForecast[:,dim]-lowerForecast[:,dim],upperForecast[:,dim]-meanForecast[:,dim]),
            color = "grey", linestyle=:dash, label = "forecast",
            xlabel = "Time", ylabel = string("x", dim))
        scatter!(UDE.times,UDE.data[dim-1,:],c=1, label = "data", xlabel = "Time", ylabel = string("x", dim))
        scatter!(times,data[dim-1,:],c=2, label = "data", xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end 
    return plot(plots...), plots
end 



function print_parameter_estimates(UDE)
    parnames = keys(UDE.parameters.process_model.known_dynamics)
    println("Estimated parameter values:")
    
    for par in parnames
        val = round(UDE.parameters.process_model.known_dynamics[par], digits = 3)
        println(par, ": ",val)
    end 
end 



function forecast_simulation_test(simulator,model,seed;train_fraction=0.9,step_size = 0.05, maxiter = 500)
    
    # generate data and split into training and testing sets 
    data = simulator(seed)
    N_train = floor(Int, train_fraction*size(data)[1])
    train_data = data[1:N_train,:]
    test_data = data[(N_train):end,:]
    
    # build model 
    model = model.constructor(train_data)
    gradient_descent!(model, step_size = step_size, maxiter = maxiter) 
    
    # forecast
    u0 =  model.parameters.uhat[:,end]
    times = test_data.t
    predicted_data = forecast(model, u0, times)
    predicted_data= DataFrame(predicted_data,names(test_data))
    
    # MSE
    SE = copy(predicted_data)
    SE[:,2:end] .= (predicted_data[:,2:end] .- test_data[:,2:end]).^2
    return train_data, test_data, predicted_data , SE
end 

function forecast_simulation_SE(simulator,model,seed;train_fraction=0.9,step_size = 0.05, maxiter = 500)
    
    # generate data and split into training and testing sets 
    data = simulator(seed)
    N_train = floor(Int, train_fraction*size(data)[1])
    train_data = data[1:N_train,:]
    test_data = data[(N_train):end,:]
    
    # build model 
    model = model.constructor(train_data)
    gradient_descent!(model, step_size = step_size, maxiter = maxiter) 
    
    # forecast
    u0 = model.parameters.uhat[:,end]
    times = test_data.t
    predicted_data = forecast(model, u0, times)
    

    return abs.(Matrix(predicted_data[:,2:end]) .- Matrix(test_data[:,2:end]))
end 


"""
    phase_plane(UDE::UDE; idx=[1,2], u1s=-5.0,0.25,5.0, u2s=-5:0.25:5,u3s = 0,T = 100)

Plots the trajectory of state variables as forecasted by the model. Runs a forecast for each permutation of u1 and u2 out to T timesteps.
Change the state variables which are plotted by changing idx such that it equals the indexes of the desired state variables as they appear in the data
"""
function phase_plane(UDE::UDE;idx = [1,2],u1s=-5:0.25:5, u2s=-5:0.25:5,u3s = 0,T = 100)
    
    # caclaute time to evaluate 
    lengths = size(UDE.data_frame,1)
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))
    
    # calcaulte u0s
    u0s = vcat([reduce(vcat,[u1,u2,u3s]) for u1 in u1s for u2 in u2s])
    permutation = unique([idx;collect(1:length(u0s[1]))])
    u0s = invpermute!.(u0s,Ref(permutation))

    plt = plot()
    for u0 in u0s
        data = forecast(UDE, u0, times) 
        Plots.plot!(plt,data[:,(idx[1]+1)],data[:,(idx[2]+1)], label = "",
                            line_z = log.(data[:,1]), c = :roma)
    end 
    
    return plt
    
end 


"""
    phase_plane(UDE::UDE, u0s::AbstractArray; idx=[1,2],T = 100)

Plots the trajectory of state variables as forecasted by the model. Runs a forecast for each provided initial condition out to T timesteps.
Change the state variables which are plotted by changing idx such that it equals the indexes of the desired state variables as they appear in the data
"""
function phase_plane(UDE::UDE, u0s::AbstractArray ;idx = [1,2],T = 100)
    
    # caclaute time to evaluate 
    lengths = size(UDE.data_frame,1)
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))
    
    plt = plot()
    for u0 in u0s
        data = forecast(UDE, u0, times) 
        Plots.plot!(plt,data[:,(idx[1]+1)],data[:,(idx[2]+1)], label = "",
                            line_z = log.(data[:,1]), c = :roma)
    end 
    
    return plt
    
end


"""
    phase_plane_3d(UDE::UDE; idx=[1,2,3], u1s=-5.0,0.25,5.0, u2s=-5:0.25:5,u3s=-5:0.25:5,T = 100)

The same as phase_plane(), but displays three dimensions/state variables instead of two
"""
function phase_plane_3d(UDE::UDE;idx = [1,2,3],u1s=-5:0.25:5, u2s=-5:0.25:5,u3s=-5:0.25:5,T = 100)
    
    # caclaute time to evaluate 
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))
    
    # calcaulte u0s  
    u0s = vcat([reduce(vcat,[u1,u2,u3]) for u1 in u1s for u2 in u2s for u3 in u3s])
    permutation = unique([idx;collect(1:length(u0s[1]))])
    u0s = invpermute!.(u0s,Ref(permutation))

    plt = plot3d()
    for u0 in u0s
        data = UniversalDiffEq.forecast(UDE, u0, 0.0, times, series) 
        Plots.plot3d!(plt,data[:,(idx[1]+2)],data[:,(idx[2]+2)],data[:,(idx[3]+2)], label = "",
                            line_z = log.(data[:,2]), c = :roma)
    end 

    return plt
    
end

function phase_plane_3d(UDE::UDE, u0s::AbstractArray;idx = [1,2,3],T = 100)
    
    # caclaute time to evaluate 
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    plt = plot3d()
    for u0 in u0s
        data = UniversalDiffEq.forecast(UDE, u0, 0.0, times, series) 
        Plots.plot3d!(plt,data[:,(idx[1]+2)],data[:,(idx[2]+2)],data[:,(idx[3]+2)], label = "",
                            line_z = log.(data[:,2]), c = :roma)
    end 

    return plt
    
end 