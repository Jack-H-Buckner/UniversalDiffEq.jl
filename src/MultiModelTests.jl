
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
    check_test_data_names(UDE.data_frame, test_data)
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

function predict(UDE::MultiUDE,test_data::DataFrame)
    series_ls =  unique(test_data[:,UDE.series_column_name])
    check_test_data_names(UDE.data_frame, test_data)
    N, T, dims, data, times,  dataframe, series_, inds, starts,
    lengths = process_multi_data(test_data, UDE.time_column_name, UDE.series_column_name)
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
            dfs[series][t,:] = vcat([series_ls[series],time[t+1]],u1hat)
        end
    end

    df = dfs[1]
    for i in 2:length(starts)
        df = vcat(df,dfs[i])
    end
    names = vcat(["series","t"], [string("x",i) for i in 1:dims])
    return DataFrame(df,names)
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

function forecast(UDE::MultiUDE, test_data::DataFrame)

    series_ls =  unique(test_data[:,UDE.series_column_name])
    check_test_data_names(UDE.data_frame, test_data)
    N, T, dims, test_data, test_times,  test_dataframe, test_series, inds, test_starts, test_lengths, labs = process_multi_data(test_data,UDE.time_column_name,UDE.series_column_name)
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, labs = process_multi_data(UDE.data_frame,UDE.time_column_name,UDE.series_column_name)

    dfs = []
    i = 0
    for series in eachindex(series_ls)
        i += 1
        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
        times_i = test_dataframe[test_dataframe[:,UDE.series_column_name] .== series, UDE.time_column_name]
        df = forecast(UDE, uhat[:,end], time[end],times_i, series_ls[series])
        push!(dfs,df)

    end


    df  = dfs[1]
    for i in 2:length(dfs)
        df = vcat(df,dfs[i])
    end

    return df
end


