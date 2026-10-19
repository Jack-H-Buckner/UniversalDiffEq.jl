"""
    plot_state_estimates(UDE::UDE)

Plots the values of the state variables estimated by the UDE model.
"""
function UniversalDiffEq.plot_state_estimates(UDE::UniversalDiffEq.UDE)

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


function UniversalDiffEq.plot_state_estimates(UDE::UniversalDiffEq.CustomUDE)

    plots = []

    for dim in 1:size(UDE.data)[1]

        # Calculate NRMSE for the current dimension
        N = length(UDE.parameters.uhat[dim,:])
        NRMSE = sqrt(sum((UDE.data[dim,:] .-UDE.parameters.uhat[dim,:]).^2)/N)/std(UDE.data[dim,:])

        plt=Plots.scatter(UDE.times,UDE.data[dim,:], label = "observations")

        Plots.plot!(UDE.times,mapslices(UDE.state_variable_transform,UDE.parameters.uhat',dims = 2)[:,dim],
         color = "grey", label= "estimated states",
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
    plot_predictions(UDE::UDE)

Plots the correspondence between the observed state transitions and the predictions from the UDE model.

"""
function UniversalDiffEq.plot_predictions(UDE::UniversalDiffEq.UDE)

    inits, obs, preds = UniversalDiffEq.predictions(UDE)

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

function UniversalDiffEq.plot_predictions(UDE::UniversalDiffEq.CustomUDE)

    inits, obs, preds = UniversalDiffEq.predictions(UDE)

    plots = []
    for dim in 1:size(obs,1)

        difs = obs[dim,:].-inits[dim,:]

        xmax = difs[argmax(difs)]
        xmin = difs[argmin(difs)]

        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1", legend_position = :topleft)
        duhat = preds[dim,:].-inits[dim,:]
        scatter!(difs,duhat,color = "white", label = "", xlabel = "Observed change", ylabel = "Predicted change")

        push!(plots, plt)

    end

    return plot(plots...)
end


"""
    plot_predictions(UDE::BayesianUDE;ci=95)

Plots the correspondence between the observed state transitons and the predicitons for the model `UDE` with a confidence interval `ci`.

"""
function UniversalDiffEq.plot_predictions(UDE::UniversalDiffEq.BayesianUDE;ci=95)

    inits, obs, preds = UniversalDiffEq.predictions(UDE,summarize = true,ci=ci)

    plots = []
    for dim in 1:size(obs[1],1)
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1")
        scatter!(difs,reduce(hcat,preds[dim,:])[2,:][dim,:].-inits[dim,:], color = "white", label = "",
            xlabel = "Observed median change", ylabel = "Predicted median change")
        push!(plots, plt)

    end

    return plot(plots...)
end


"""
    plot_predictions(UDE::UDE, test_data::DataFrame)

Plots the correspondence between the observed state transitions in test data and the predictions from the UDE model.
"""
function UniversalDiffEq.plot_predictions(UDE::UniversalDiffEq.UDE,test_data::DataFrame)
    check_test_data_names(UDE.data_frame, test_data)
    inits, obs, preds = UniversalDiffEq.predictions(UDE,test_data)

    plots = []
    for dim in 1:size(obs,1)
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        duhat = preds[dim,:].-inits[dim,:]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1")
        scatter!(difs, duhat, color = "white", label = "", xlabel = "Observed change", ylabel = "Predicted change")

        N = length(difs)
        NRMSE = sqrt(sum((difs .- duhat).^2)/N)/std(difs)

        text_x = 0.5*(xmax-xmin)+xmin
        text_y = 0.1*(xmax-xmin)+xmin

        Plots.annotate!(text_x, text_y, text("NRMSE = $(round(NRMSE, digits=3))", :left, 10))
        push!(plots, plt)

    end

    return plot(plots...)
end

function UniversalDiffEq.plot_predictions(UDE::UniversalDiffEq.BayesianUDE,test_data::DataFrame;ci=95)
    check_test_data_names(UDE.data_frame, test_data)
    inits, obs, preds = UniversalDiffEq.predictions(UDE,test_data,summarize = true,ci=ci)

    plots = []
    for dim in 1:size(obs[1],1)
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "1:1")
        scatter!(difs,reduce(hcat,preds[dim,:])[2,:][dim,:].-inits[dim,:],color = "white", label = "",
            xlabel = "Observed median change", ylabel = "Predicted median change")
        push!(plots, plt)

    end

    return plot(plots...)
end


"""
    plot_forecast(UDE::UDE, T::Int)

Plots the model's forecast up to T time steps into the future from the last observation.
"""
function UniversalDiffEq.plot_forecast(UDE::UniversalDiffEq.UDE, T::Int)
    u0 = UDE.parameters.uhat[:,end]
    dts = UDE.times[2:end] .- UDE.times[1:(end-1)]
    dt = sum(dts)/length(dts)
    times = UDE.times[end]:dt:(UDE.times[end] + T*dt )
    df = UniversalDiffEq.forecast(UDE, u0, times)
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

"""
    plot_forecast(UDE::BayesianUDE, T::Int)

Plots the models forecast up to T time steps into the future from the last observation including the median prediction as well as the `ci`% lower and upper confidence intervals.
"""
function UniversalDiffEq.plot_forecast(UDE::UniversalDiffEq.BayesianUDE, T::Int;ci = 95)
    u0 = reduce((x,y) -> cat(x,y,dims = 3),[UDE.parameters[i].uhat[:,end] for i in 1:length(UDE.parameters)])
    u0 = mean(u0,dims = 3)
    u0 = vec(u0[:,:,1])
    u0 = vec(u0[:,:,1])
    dts = UDE.times[2:end] .- UDE.times[1:(end-1)]
    dt = sum(dts)/length(dts)
    times = UDE.times[end]:dt:(UDE.times[end] + T*dt )
    df = UniversalDiffEq.forecast(UDE, u0, times,summarize = true, ci = ci)
    df = UniversalDiffEq.forecast(UDE, u0, times,summarize = true, ci = ci)
    meanForecast = [df[i,j][2] for i in 1:size(df,1), j in 1:size(df,2)]
    lowerForecast = [df[i,j][1] for i in 1:size(df,1), j in 1:size(df,2)]
    upperForecast = [df[i,j][3] for i in 1:size(df,1), j in 1:size(df,2)]
    plots = []
    for dim in 2:size(df,2)
        plt = plot(meanForecast[:,1], meanForecast[:,dim],
            ribbon = (meanForecast[:,dim]-lowerForecast[:,dim],upperForecast[:,dim]-meanForecast[:,dim]),
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

Plots the model's forecast over the range of the test data along with the value of the test data.
"""
function UniversalDiffEq.plot_forecast(UDE::UniversalDiffEq.UDE, test_data::DataFrame)
    UniversalDiffEq.check_test_data_names(UDE.data_frame, test_data)
    u0 = UDE.parameters.uhat[:,end]
    N, dims, T, times, data, dataframe = UniversalDiffEq.process_data(test_data,UDE.time_column_name)
    df = UniversalDiffEq.forecast(UDE, u0, UDE.times[end], times)
    plots = []
    for dim in 2:size(df)[2]
        plt = plot(df[:,1], df[:,dim], color = "grey", linestyle=:dash, label = "forecast",
            xlabel = "Time", ylabel = string("x", dim))
        scatter!(UDE.times,UDE.data[dim-1,:],c=1, label = "data", xlabel = "Time", ylabel = string("x", dim))
        scatter!(times,data[dim-1,:],c=2, label = "data", xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end
    return plot(plots...), plots
end


"""
    plot_forecast(UDE::BayesianUDE, test_data::DataFrame)

Plots the model's forecast over the range of the `test_data` along with the value of the test data including the median prediction as well as the `ci`% lower and upper confidence intervals.
"""
function UniversalDiffEq.plot_forecast(UDE::UniversalDiffEq.BayesianUDE, test_data::DataFrame;ci = 95)
    UniversalDiffEq.check_test_data_names(UDE.data_frame, test_data)
    u0 = reduce((x,y) -> cat(x,y,dims = 3),[UDE.parameters[i].uhat[:,end] for i in 1:length(UDE.parameters)])
    u0 = mean(u0,dims = 3)
    u0 = vec(u0[:,:,1])
    N, dims, T, times, data, dataframe = UniversalDiffEq.process_data(test_data,UDE.time_column_name)
    df = UniversalDiffEq.forecast(UDE, u0, UDE.times[end], times,summarize = true, ci = ci)
    u0 = vec(u0[:,:,1])
    N, dims, T, times, data, dataframe = UniversalDiffEq.process_data(test_data,UDE.time_column_name)
    df = UniversalDiffEq.forecast(UDE, u0, UDE.times[end], times,summarize = true, ci = ci)
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


"""
    phase_plane(UDE::UDE; idx=[1,2], u1s=-5:0.25:5, u2s=-5:0.25:5, u3s = 0, T = 100)

Plots the trajectory of state variables as forecasted by the model. Runs a forecast for each permutation of u1 and u2 out to T timesteps.
Change the state variables that are plotted by changing `idx` such that it equals the indexes of the desired state variables as they appear in the data.
"""
function UniversalDiffEq.phase_plane(UDE::UniversalDiffEq.UDE;idx = [1,2],u1s=-5:0.25:5, u2s=-5:0.25:5,u3s = 0,T = 100)

    # calculate time to evaluate
    lengths = size(UDE.data_frame,1)
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    # calculate u0s
    u0s = vcat([reduce(vcat,[u1,u2,u3s]) for u1 in u1s for u2 in u2s])
    permutation = unique([idx;collect(1:length(u0s[1]))])
    u0s = invpermute!.(u0s,Ref(permutation))

    plt = plot()
    for u0 in u0s
        data = UniversalDiffEq.forecast(UDE, u0, times)
        Plots.plot!(plt,data[:,(idx[1]+1)],data[:,(idx[2]+1)], label = "",
                            line_z = log.(data[:,1]), c = :roma)
    end

    return plt

end


"""
    phase_plane(UDE::UDE, u0s::AbstractArray; idx=[1,2],T = 100)

Plots the trajectory of state variables as forecasted by the model. Runs a forecast for each provided initial condition out to `T` timesteps. Change the state variables that are plotted by changing `idx` such that it equals the indexes of the desired state variables as they appear in the data.
"""
function UniversalDiffEq.phase_plane(UDE::UniversalDiffEq.UDE, u0s::AbstractArray ;idx = [1,2],T = 100)

    # calculate time to evaluate
    lengths = size(UDE.data_frame,1)
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    plt = plot()
    for u0 in u0s
        data = UniversalDiffEq.forecast(UDE, u0, times)
        Plots.plot!(plt,data[:,(idx[1]+1)],data[:,(idx[2]+1)], label = "",
                            line_z = log.(data[:,1]), c = :roma)
    end

    return plt

end


"""
    phase_plane_3d(UDE::UDE; idx=[1,2,3], u1s=-5:0.25:5, u2s=-5:0.25:5, u3s=-5:0.25:5, T = 100)

The same as phase_plane(), but displays three dimensions/state variables instead of two.
"""
function UniversalDiffEq.phase_plane_3d(UDE::UniversalDiffEq.UDE;idx = [1,2,3],u1s=-5:0.25:5, u2s=-5:0.25:5,u3s=-5:0.25:5,T = 100)

    # calculate time to evaluate
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    # calculate u0s
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

function UniversalDiffEq.phase_plane_3d(UDE::UniversalDiffEq.UDE, u0s::AbstractArray;idx = [1,2,3],T = 100)

    # calculate time to evaluate
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


"""
    observation_error_correlations(UDE)

The first differnce plot of the observation errors ``\epsilon_t``. This allows the user to check for autocorrelation in the observation errors.   
"""
function UniversalDiffEq.observation_error_correlations(UDE;fig_size = (600,500))
    proc_resids, obs_resids = UniversalDiffEq.get_residuals(UDE)
    plts = []
    for i in 1:size(obs_resids)[1]
        for j in 1:size(obs_resids)[1]
            x = obs_resids[i,1:(end-1)]
            y = obs_resids[j,2:end]
            push!(plts,Plots.scatter(x,y, label = "", xlabel = string("y_",i), ylabel = string("y_",j),
                                    title = string("cor =", round(cor(x,y), digits = 2)),
                                    titlefontsize = 8, ticks = false, guidefontsize = 8))
        end 
    end
    plt = plot(plts...)
    plot!(plt,size = fig_size)
    return plt
end 