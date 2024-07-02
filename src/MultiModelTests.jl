
function get_final_state(UDE::MultiUDE)
    return UDE.parameters.uhat[:,end]
end 


function print_parameter_estimates(UDE::MultiUDE)
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


function get_parameters(UDE::MultiUDE)
    return UDE.parameters.process_model
end


function get_NN_parameters(UDE::MultiUDE)
    return UDE.parameters.process_model.NN
end


function get_right_hand_side(UDE::MultiUDE)
    pars = get_parameters(UDE)
    str_labs = UDE.series_labels.label
    int_labs = UDE.series_labels.index

    rhs = x -> 1.0
    if UDE.X == 0
        rhs = function (u,series,t)
            series = int_labs[str_labs .== series][1]
            return UDE.process_model.right_hand_side(u,series,t,pars)
        end
    else
        rhs = function (u,series,x,t)
            series = int_labs[str_labs .== series][1]
            return UDE.process_model.right_hand_side(u,series,x,t,pars)
        end
    end  
    return rhs
end 

function get_predict(UDE::MultiUDE)
    pars = get_parameters(UDE)
    (u,t,dt) -> UDE.process_model.predict(u,t,dt,pars)
end 


function predictions(UDE::MultiUDE)
    

    N, T, dims, data, times,  dataframe, series_ls, inds, starts, lengths, labels = process_multi_data(UDE.data_frame,UDE.time_column_name,UDE.series_column_name)

    series_ls =  unique(UDE.data_frame[:,"series"])

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
     
    N, T, dims, data, times,  dataframe, series_ls, inds, starts, lengths, labs = process_multi_data(test_data,UDE.time_column_name,UDE.series_column_name)
    series_ls =  unique(UDE.data_frame[:,"series"])
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



function plot_predictions(UDE::MultiUDE)
 
    inits, obs, preds = predictions(UDE)
    
    plots = []
    for dim in 1:size(obs)[1]
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1", title = UDE.varnames[dim])
        if dim > 1
            plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "", title = UDE.varnames[dim])
        end
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change", ylabel = "Predicted change")
        
        duhat = preds[dim,:].-inits[dim,:]
        nrmse = sqrt(sum((difs .- duhat).^2)/length(difs)) / std(difs)
        nrmse = round(nrmse,digits = 3)
        ylim = ylims(plt);ypos = (ylim[2]-ylim[1])*0.925 + ylim[1]
        xlim = xlims(plt);xpos = (xlim[2]-xlim[1])*0.250 + xlim[1]
        Plots.annotate!(plt,[xpos],[ypos],text(string("NRMSE: ", nrmse),9), legend_position = :bottomright)
        
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
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1", title = UDE.varnames[dim])
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change", ylabel = "Predicted change")
        
        duhat = preds[dim,:].-inits[dim,:]
        nrmse = sqrt(sum((difs .- duhat).^2)/length(difs)) / std(difs)
        nrmse = round(nrmse,digits = 3)
        ylim = ylims(plt);ypos = (ylim[2]-ylim[1])*0.925 + ylim[1]
        xlim = xlims(plt);xpos = (xlim[2]-xlim[1])*0.250 + xlim[1]
        Plots.annotate!(plt,[xpos],[ypos],text(string("NRMSE: ", nrmse),9), legend_position = :bottomright)

        push!(plots, plt)
    end  
    return plot(plots...)
end


function plot_state_estimates(UDE::MultiUDE; show_legend = true)
    

    plots = []

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, labs= process_multi_data(UDE.data_frame,UDE.time_column_name,UDE.series_column_name)

    for d in 1:dims
        plt = plot(); NRMSE = 0
        for series in eachindex(starts)

            uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
            time = times[starts[series]:(starts[series]+lengths[series]-1)]
            dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]

            yhat = broadcast(t -> UDE.observation_model.link(uhat[:,t],UDE.parameters.observation_model)[d], eachindex(time))
            series_lab = UDE.series_labels.label[ UDE.series_labels.index .== series][1]
            Plots.scatter!(time,dat[d,:], label = "", c = series, width = 0.5)
            if d == 1
                Plots.plot!(time, yhat,  label= string(series_lab),xlabel = "time", ylabel = UDE.varnames[d], c = series, width =2, linestyle = :dash)
            else
                Plots.plot!(time, yhat,  label= "",xlabel = "time", ylabel = UDE.varnames[d], c = series, width =2, linestyle = :dash)
            end
            NRMSE += sqrt(sum((dat[d,:] .- yhat).^2)/length(dat[d,:]))/std(dat[d,:])
        end 
        
        
        ylim = ylims(plt);ypos = (ylim[2]-ylim[1])*0.925 + ylim[1]
        xlim = xlims(plt);xpos = (xlim[2]-xlim[1])*0.750 + xlim[1]
        nrmse = round(NRMSE/length(series),digits=3)
        if show_legend
            Plots.annotate!(plt,[xpos],[ypos],text(string("Mean NRMSE: ", nrmse),9), legend_position = :outerbottomright)
        else
            Plots.annotate!(plt,[xpos],[ypos],text(string("Mean NRMSE: ", nrmse),9), legend_position = :none)        
        end
        push!(plots,plt)
    end
            
    return plot(plots...)       
end 




function forecast(UDE::MultiUDE, u0::AbstractVector{}, t0::Real, times::AbstractVector{}, series)
  
    ind = UDE.series_labels.index[UDE.series_labels.label .== series][1]
    
    estimated_map = (x,t,dt) -> UDE.process_model.predict(x,ind,t,dt,UDE.parameters.process_model)[1]
    x = u0

    df = zeros(length(times),length(x))
    series_dat = []
    times_dat = []
    for t in eachindex(times)
        dt = times[t] - t0
        tinit = t0
        if t > 1
            dt = times[t]-times[t-1]
            tinit = times[t-1]
        end
        x = estimated_map(x,tinit,dt)
        push!(series_dat,series)
        push!(times_dat,times[t])
        df[t,:] = UDE.observation_model.link(x,UDE.parameters.observation_model)
    end 
    df = DataFrame(df,UDE.varnames)
    df[:,UDE.series_column_name] .= series_dat
    df[:,UDE.time_column_name] .= times_dat
    df = df[:,vcat([UDE.series_column_name,UDE.time_column_name],UDE.varnames)]
    return df
end 




function plot_forecast(UDE::MultiUDE, test_data::DataFrame; show_legend = true)

    N, T, dims, test_data, test_times,  test_dataframe, test_series, inds, test_starts, test_lengths, labs = process_multi_data(test_data,UDE.time_column_name,UDE.series_column_name)
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, labs = process_multi_data(UDE.data_frame,UDE.time_column_name,UDE.series_column_name)
    
    series_ls = unique(test_dataframe[:,UDE.series_column_name])
    plots = []
    for d in 1:dims
        plt = plot();i = 0; NRMSE = 0
        for series in eachindex(series_ls)
            i += 1
      
            time = times[starts[series]:(starts[series]+lengths[series]-1)]
            dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
            uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
            
            test_time = test_times[test_starts[series]:(test_starts[series]+test_lengths[series]-1)]
            test_dat = test_data[:,test_starts[series]:(test_starts[series]+test_lengths[series]-1)]
            
            df = forecast(UDE, uhat[:,end], time[end], test_dataframe[test_dataframe[:,UDE.series_column_name] .== series_ls[series], UDE.time_column_name], series_ls[series])

            if d == 1
                plt = plot!(df[:,2],df[:,d+2],c=i, linestyle=:dash, width = 2, label = string(series_ls[series]),xlabel = "Time", ylabel = string("x", dim))
            
            else
                plt = plot!(df[:,2],df[:,d+2],c=i, linestyle=:dash, width = 2, label = "",xlabel = "Time", ylabel = string("x", dim))
            end

            scatter!(time,dat[d,:],c=i, label = "",ylabel = UDE.varnames[d], alpha = 0.5, markersize = 2.5)
            scatter!(test_time,test_dat[d,:],c=i, label = "", markersize = 2.5)
            NRMSE += sqrt(sum((test_dat[d,:] .- df[:,d+2]).^2)/length(test_dat[d,:]))/std(test_dat[d,:])
        end 
        ylim = ylims(plt)
        ypos = (ylim[2]-ylim[1])*0.925 + ylim[1]
        xlim = xlims(plt)
        xpos = (xlim[2]-xlim[1])*0.750 + xlim[1]
        nrmse = round(NRMSE/length(series),digits=3)
        if show_legend
            Plots.annotate!(plt,[xpos],[ypos],text(string("Mean NRMSE: ", nrmse),9), legend_position = :outerbottomright)
        else
            Plots.annotate!(plt,[xpos],[ypos],text(string("Mean NRMSE: ", nrmse),9), legend_position = :none)        
        end
       
        push!(plots, plt)
        
    end 

    return plot(plots...)
end 





function phase_plane(UDE::MultiUDE;u1s=-5:0.25:5, u2s=-5:0.25:5,T = 100)
    
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



