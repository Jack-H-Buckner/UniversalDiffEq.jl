
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

    N, T, dims, data, times, dataframe, series, inds, starts, lengths, labels = process_multi_data(UDE.data_frame,UDE.time_column_name,UDE.series_column_name)

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
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths, labs = process_multi_data(test_data,UDE.time_column_name,UDE.series_column_name)
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

function predict(UDE::MultiUDE, test_data::DataFrame)

    # Check for unseen series in test_data
    train_series = unique(UDE.data_frame[:, UDE.series_column_name])
    test_series  = unique(test_data[:, UDE.series_column_name])
    unseen = join(setdiff(test_series, train_series), ", ")
    if !isempty(unseen)
        @info "Test data contain series [$unseen] that were not in training data. " *
              "Using default model parameters for one-step-ahead predictions."
    end

    series_ls = unique(test_data[:, UDE.series_column_name])
    check_test_data_names(UDE.data_frame, test_data)

    N, T, dims, data, times, dataframe, series, inds, starts, lengths, varnames, labels_df = process_multi_data(test_data, UDE.time_column_name, UDE.series_column_name)

    dfs = DataFrame[]
    
    for series in eachindex(starts)
        train_idx = findfirst(==(series_ls[series]), train_series)
        series_i  = isnothing(train_idx) ? series : train_idx
        
        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:, starts[series]:(starts[series]+lengths[series]-1)]
        dat = mapslices(u -> UDE.observation_model.inv_link(u, UDE.parameters.observation_model), dat, dims = 1)
        
        # Numeric matrix for prediction outputs
        u1hat_mat = zeros(lengths[series] - 1, dims)
        times_out = zeros(lengths[series] - 1)

        for t in 1:(lengths[series] - 1)
            u0 = dat[:, t]; u1 = dat[:, t+1]; dt = time[t+1] - time[t]
            
            # Pass integer loop index `series` to ODE solver
            u1hat, epsilon = UDE.process_model.predict(u0, series_i, time[t], dt, UDE.parameters.process_model)
            u1hat = UDE.observation_model.link(u1hat, UDE.parameters.observation_model)
            
            u1hat_mat[t, :] = u1hat
            times_out[t] = time[t+1]
        end

        # Construct DataFrame for this series with string series labels
        df_s = DataFrame()
        df_s[!, Symbol(UDE.series_column_name)] = fill(series_ls[series], lengths[series] - 1)
        df_s[!, Symbol(UDE.time_column_name)] = times_out
        for i in 1:dims
            df_s[!, Symbol(varnames[i])] = u1hat_mat[:, i]
        end

        push!(dfs, df_s)
    end

    # Combine all series DataFrames
    return reduce(vcat, dfs)
end

function forecast(UDE::MultiUDE, u0::AbstractVector, t0::Real, times::AbstractVector, series)

    # Normalize labels to string for type-safe comparison
    labels_str = string.(UDE.series_labels.label)
    series_str = string(series)

    match_idx = findfirst(==(series_str), labels_str)

    # Safely resolve process model index `ind`
    local ind::Int
    if !isnothing(match_idx)
        ind = UDE.series_labels.index[match_idx]
    elseif series isa Integer && 1 <= series <= length(UDE.series_labels.index)
        # Fallback if `series` is already an integer index
        ind = series
    else
        # Fallback for unseen series in transfer learning
        ind = series isa Integer ? series : length(UDE.series_labels.index) + 1
    end

    estimated_map = (x, t, dt) -> UDE.process_model.predict(x, ind, t, dt, UDE.parameters.process_model)[1]
    x = u0

    df = zeros(length(times), length(x))
    series_dat = []
    times_dat  = []

    for t in eachindex(times)
        dt    = times[t] - t0
        tinit = t0
        if t > 1
            dt    = times[t] - times[t-1]
            tinit = times[t-1]
        end
        x = estimated_map(x, tinit, dt)
        push!(series_dat, series)
        push!(times_dat, times[t])
        df[t, :] = UDE.observation_model.link(x, UDE.parameters.observation_model)
    end

    df = DataFrame(df, UDE.varnames)
    df[:, Symbol(UDE.series_column_name)] = series_dat
    df[:, Symbol(UDE.time_column_name)]   = times_dat
    df = df[:, vcat([UDE.series_column_name, UDE.time_column_name], UDE.varnames)]

    return df
end

function forecast(UDE::MultiUDE, test_data::DataFrame)

    check_test_data_names(UDE.data_frame, test_data)

    # Process multi-data for both test and training datasets
    test_N, test_T, test_dims, test_data_mat, test_times, test_dataframe, test_series, test_inds, test_starts, test_lengths, test_varnames, test_labels_df = process_multi_data(
        test_data, UDE.time_column_name, UDE.series_column_name
    )
    N, T, dims, data, times, dataframe, series, inds, starts, lengths, varnames, train_labels_df = process_multi_data(
        UDE.data_frame, UDE.time_column_name, UDE.series_column_name
    )

    train_labels_str = string.(train_labels_df.label)

    # Inform user of any unseen series
    unseen = join(setdiff(test_labels_df.label, train_labels_df.label), ", ")
    if !isempty(unseen)
        @info "Test data contain unseen series [$unseen]. Using default model parameters for transfer learning forecast."
    end

    dfs = DataFrame[]

    for i in 1:length(test_starts)
        s_label_raw = test_labels_df.label[i]
        s_label_str = string(s_label_raw)

        # Map test label string to corresponding series index in training data
        train_idx = findfirst(==(s_label_str), train_labels_str)

        local u0, t0

        if !isnothing(train_idx)
            # Known series: initial condition u0 and t0 come from training uhat
            s_start    = starts[train_idx]
            s_length   = lengths[train_idx]
            time_train = times[s_start:(s_start + s_length - 1)]
            uhat_train = UDE.parameters.uhat[:, s_start:(s_start + s_length - 1)]

            u0 = uhat_train[:, end]
            t0 = time_train[end]
        else
            # Unseen series: initial condition from first test observation
            s_test_start = test_starts[i]
            t0           = test_times[s_test_start]
            u0_raw       = Vector{Float64}(test_data_mat[:, s_test_start])
            u0           = UDE.observation_model.inv_link(u0_raw, UDE.parameters.observation_model)
        end

        # Slice target test times for this series
        times_i = test_times[test_starts[i]:(test_starts[i] + test_lengths[i] - 1)]

        # Filter strictly AFTER t0 to avoid dt = 0.0 in single-series forecast
        times_eval = sort(unique(times_i[times_i .> t0]))

        if isempty(times_eval)
            @warn "Series '$s_label_raw' has no evaluation time points strictly after t0 = $t0. Skipping forecast."
            continue
        end

        # Pass s_label_raw directly to single-series forecast
        df = forecast(UDE, u0, t0, times_eval, s_label_raw)

        push!(dfs, df)
    end

    if isempty(dfs)
        @warn "No valid forecasts were generated. Ensure test time points occur after training time points."
        return DataFrame()
    end

    return reduce(vcat, dfs)
end