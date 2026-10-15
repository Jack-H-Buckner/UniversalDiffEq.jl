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
    check_test_data_names(UDE.data_frame, test_data)
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


function plot_forecast(UDE::MultiUDE, test_data::DataFrame; show_legend = true)
    check_test_data_names(UDE.data_frame, test_data)
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


function phase_plane(UDE::MultiUDE;idx = [1,2],u1s=-5:0.25:5, u2s=-5:0.25:5,u3s = 0,T = 100)

    # caclaute time to evaluate
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    # Generate subplot for each series
    plots = []
    for series in UDE.series_labels.label

        #Calculate u0s
        u0s = vcat([reduce(vcat,[u1,u2,u3s]) for u1 in u1s for u2 in u2s])
        permutation = unique([idx;collect(1:length(u0s[1]))])
        u0s = invpermute!.(u0s,Ref(permutation))

        plt = plot()
        for u0 in u0s
            data = forecast(UDE, u0, 0.0, times, series)
            Plots.plot!(plt,data[:,(idx[1]+2)],data[:,(idx[2]+2)], label = "",
                                line_z = log.(data[:,2]), c = :roma)
        end
        push!(plots, plt)
    end

    return plot(plots...)

end


function phase_plane(UDE::MultiUDE, u0s::AbstractArray;idx = [1,2],T = 100)

    # caclaute time to evaluate
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    # Generate subplot for each series
    plots = []
    for series in UDE.series_labels.label
        plt = plot()
        for u0 in u0s
            data = forecast(UDE, u0, 0.0, times, series)
            Plots.plot!(plt,data[:,(idx[1]+2)],data[:,(idx[2]+2)], label = "",
                                line_z = log.(data[:,2]), c = :roma)
        end
        push!(plots, plt)
    end

    return plot(plots...)

end

function phase_plane_3d(UDE::MultiUDE;idx = [1,2,3],u1s=0:0.1:1, u2s=-0:0.1:1,u3s = 0:0.1:1.0,T = 50)

    # caclaute time to evaluate
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    # Generate subplot for each series
    plots = []
    for series in UDE.series_labels.label

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
        push!(plots, plt)
    end

    return plot3d(plots...)

end

function phase_plane_3d(UDE::MultiUDE, u0s::AbstractArray;idx = [1,2,3],T = 100)

    # caclaute time to evaluate
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    dts = UDE.times[2:lengths[1]] .- UDE.times[1:(lengths[1]-1)]
    dt = sum(dts)/length(dts)
    times = collect(dt:dt:(T*dt))

    # Generate subplot for each series
    plots = []
    for series in UDE.series_labels.label

        plt = plot3d()
        for u0 in u0s
            data = UniversalDiffEq.forecast(UDE, u0, 0.0, times, series)
            Plots.plot3d!(plt,data[:,(idx[1]+2)],data[:,(idx[2]+2)],data[:,(idx[3]+2)], label = "",
                                line_z = log.(data[:,2]), c = :roma)
        end
        push!(plots, plt)
    end

    return plot3d(plots...)

end