

function predictions(UDE::MultiUDE)
     

    N, T, dims, data, times,  dataframe, series_ls, inds, starts, lengths = process_multi_data(UDE.data_frame)

    series_ls =  unique(UDE.data_frame.series)

    inits = [zeros(dims,l-1) for l in lengths]
    obs = [zeros(dims,l-1) for l in lengths]
    preds = [zeros(dims,l-1) for l in lengths]
    
    for series in eachindex(starts)
        uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        for t in 1:(lengths[series]-1)
            u0 = uhat[:,t]; u1 = uhat[:,t+1];dt = time[t+1] - time[t]
            u1hat, epsilon = UDE.process_model.predict(u0,series_ls[series],UDE.times[t],dt,UDE.parameters.process_model) 
            u0 = UDE.observation_model.link(u0,UDE.parameters.observation_model)
            u1 = UDE.observation_model.link(u1,UDE.parameters.observation_model)
            u1hat = UDE.observation_model.link(u1hat,UDE.parameters.observation_model)
            inits[series][:,t] = u0;obs[series][:,t] = u1;preds[series][:,t] = u1hat
        end
    end 
    
    init = inits[1];ob = obs[1];pred = preds[1]
    for i in 2:length(starts)
        init = hcat(init,inits[i])
        ob = hcat(ob,obs[i])
        pred = hcat(pred,preds[i])
    end

    return init, ob, pred
end 

function predictions(UDE::MultiUDE,test_data::DataFrame)
     
    N, T, dims, data, times,  dataframe, series_ls, inds, starts, lengths = process_multi_data(test_data)
    series_ls =  unique(UDE.data_frame.series)
    inits = [zeros(dims,l-1) for l in lengths]
    obs = [zeros(dims,l-1) for l in lengths]
    preds = [zeros(dims,l-1) for l in lengths]
    for series in eachindex(starts)
        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        dat = mapslices(u -> UDE.observation_model.inv_link(u,UDE.parameters.observation_model), dat ,dims = 1) 
        for t in 1:(lengths[series]-1)
            u0 = dat[:,t];u1 = dat[:,t+1];dt = time[t+1] - time[t]
            u1hat, epsilon = UDE.process_model.predict(u0,series_ls[series],time[t],dt,UDE.parameters.process_model) 
            u0 = UDE.observation_model.link(u0,UDE.parameters.observation_model)
            u1 = UDE.observation_model.link(u1,UDE.parameters.observation_model)
            u1hat = UDE.observation_model.link(u1hat,UDE.parameters.observation_model)
            inits[series][:,t] = u0; obs[series][:,t] = u1; preds[series][:,t] = u1hat
        end
    end 
    
    init = inits[1];ob = obs[1];pred = preds[1]
    for i in 2:length(starts)
        init = hcat(init,inits[i])
        ob = hcat(ob,obs[i])
        pred = hcat(pred,preds[i])
    end

    return init, ob, pred
end 


function predict(UDE::MultiUDE,test_data::DataFrame)
     
    N, T, dims, data, times,  dataframe, series_ls, inds, starts, lengths = process_multi_data(test_data)
    series_ls =  unique(UDE.data_frame.series)
    dfs = [zeros(l-1,dims+2) for l in lengths]
    for series in eachindex(starts)
        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        dat = mapslices(u -> UDE.observation_model.inv_link(u,UDE.parameters.observation_model), dat ,dims = 1) 
        for t in 1:(lengths[series]-1)
            u0 = dat[:,t];u1 = dat[:,t+1];dt = time[t+1] - time[t]
            u1hat, epsilon = UDE.process_model.predict(u0,series_ls[series],time[t],dt,UDE.parameters.process_model) 
            u0 = UDE.observation_model.link(u0,UDE.parameters.observation_model)
            u1 = UDE.observation_model.link(u1,UDE.parameters.observation_model)
            u1hat = UDE.observation_model.link(u1hat,UDE.parameters.observation_model)
            dfs[series][t,:] = vcat([series,time[t+1]],u1hat)
        end
    end 
    
    df = dfs[1]
    for i in 2:length(starts)
        df = vcat(df,dfs[i])
    end
    names = vcat(["series","t"], [string("x",i) for i in 1:dims])
    return DataFrame(df,names)
end 

function plot_predictions(UDE::MultiUDE)
 
    inits, obs, preds = predictions(UDE)
    
    plots = []
    for dim in 1:size(obs)[1]
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "45 degree")
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change Delta hatu_t",ylabel = "Predicted change hatut - hatu_t")
        push!(plots, plt)
            
    end
        
    return plot(plots...)
end


function plot_predictions(UDE::MultiUDE,test_data::DataFrame)
 
    inits, obs, preds = predictions(UDE,test_data)
    plots = []
    for dim in 1:size(obs)[1]
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "45 degree")
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change", ylabel = "Predicted change")
        push!(plots, plt)
    end  
    return plot(plots...)
end


function plot_state_estimates(UDE::MultiUDE)
    

    plots = []

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(UDE.data_frame)

    for d in 1:dims
        plt = plot()
        for series in eachindex(starts)

            uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
            time = times[starts[series]:(starts[series]+lengths[series]-1)]
            dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]

            yhat = broadcast(t -> UDE.observation_model.link(uhat[:,t],UDE.parameters.observation_model)[d], eachindex(time))

            Plots.scatter!(time,dat[d,:], label = "", c = series, width = 0.5)
            Plots.plot!(time, yhat,  label= "",xlabel = "time", ylabel = string("x", dim), c = series, width =2, linestyle = :dash)
        
        end 
        push!(plots,plt)
    end
            
    return plot(plots...)       
end 



function forecast(UDE::MultiUDE, u0::AbstractVector{}, times::AbstractVector{}, series)
    estimated_map = (x,t,dt) -> UDE.process_model.predict(x,series,t,dt,UDE.parameters.process_model)[1]
    x = u0
    df = zeros(length(times),length(x)+1)
    df[1,:] = vcat([times[1]],UDE.observation_model.link(x,UDE.parameters.observation_model))
    for t in 2:length(times)
        dt = times[t]-times[t-1]
        x = estimated_map(x,times[t-1],dt)
        df[t,:] = vcat([times[t]],UDE.observation_model.link(x,UDE.parameters.observation_model))
    end 
    
    return df
end 


function forecast(UDE, u0::AbstractVector{}, t0::Real, times::AbstractVector{}, series)

    estimated_map = (x,t,dt) -> UDE.process_model.predict(x,series,t,dt,UDE.parameters.process_model)[1]
    x = u0

    df = zeros(length(times),length(x)+2)
    
    for t in eachindex(times)
        dt = times[t] - t0
        tinit = t0
        if t > 1
            dt = times[t]-times[t-1]
            tinit = times[t-1]
        end
        x = estimated_map(x,tinit,dt)
        df[t,:] = vcat([series,times[t]],UDE.observation_model.link(x,UDE.parameters.observation_model))
    end 
    
    return df
end 


function plot_forecast(UDE::MultiUDE, T::Int)
    
    N, T_, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(UDE.data_frame)
    series_ls =  unique(UDE.data_frame.series)
    plots = []
    for dim in 1:dims
        plt = plot();i=0
         for t0 in starts
            i+=1
            lower_ind = t0
            upper_ind = t0 + lengths[i]-1
            u0 = UDE.parameters.uhat[:,upper_ind]
            
            dts = UDE.times[(lower_ind+1):upper_ind] .- UDE.times[lower_ind:(upper_ind-1)]
            dt = sum(dts)/length(dts)
            times = UDE.times[upper_ind]:dt:(UDE.times[upper_ind] + T*dt )
            
            df = forecast(UDE, u0, times, series_ls[i])
            
            plt = plot!(df[:,1],df[:,dim+1],color = "grey", linestyle=:dash, label = "",xlabel = "Time", ylabel = string("x", dim))
            plot!(UDE.times[lower_ind:upper_ind],UDE.data[dim,lower_ind:upper_ind],c=1, label = "",xlabel = "Time", ylabel = string("x", dim))
     
        end 
       
        push!(plots, plt)
        
    end 

    return plot(plots...)
end 



function plot_forecast(UDE::MultiUDE, test_data::DataFrame)

    N, T, dims, test_data, test_times,  test_dataframe, test_series, inds, test_starts, test_lengths = process_multi_data(test_data)
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(UDE.data_frame)
    
    series_ls = unique(test_dataframe.series)
    plots = []
    for d in 1:dims
        plt = plot();i = 0
         for series in eachindex(series_ls)
            i += 1
            series = series_ls[series]
            time = times[starts[series]:(starts[series]+lengths[series]-1)]
            dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
            uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
            
            test_time = test_times[test_starts[series]:(test_starts[series]+test_lengths[series]-1)]
            test_dat = test_data[:,test_starts[series]:(test_starts[series]+test_lengths[series]-1)]

            df = forecast(UDE, uhat[:,end], test_dataframe.t[test_dataframe.series .== series], series)

            plt = plot!(df[:,1],df[:,d+1],c=i, linestyle=:dash, width = 2, label = "",xlabel = "Time", ylabel = string("x", dim))
            scatter!(time,dat[d,:],c=1, label = "",ylabel = string("x", dim))
            scatter!(test_time,test_dat[d,:],c=i, label = "")
     
        end 
       
        push!(plots, plt)
        
    end 

    return plot(plots...)
end 





function phase_plane(UDE;u1s=-5:0.25:5, u2s=-5:0.25:5,T = 100)
    
    # caclaute time to evaluate 
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))
    
    # calcaulte u0s
    u0s = vcat([[u1,u2] for u1 in u1s for u2 in u2s])
    plt = plot()
    for u0 in u0s
        data = forecast(UDE, u0, times) 
        Plots.plot!(plt,data[:,2],data[:,3], label = "",
                            line_z = log.(data[:,1]), c = :roma)
    end 
    
    return plt
    
end 