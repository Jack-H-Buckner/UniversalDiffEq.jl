
function process_long_format_data(data, time_column_name, variable_column_name, value_column_name)
    try
        unique(data[:,variable_column_name])
    catch e
        if isa(e, ArgumentError)
            error("variable_column_name was not found in provided data. Please set the kwarg: variable_column_name to the correct value")
        end
    end
    variables = unique(data[:,variable_column_name])
    times = []; values = []
    for var in variables
        inds = data[:,variable_column_name] .== var
        dat_i = data[inds,:]
        push!(times, dat_i[:,time_column_name])
        try
            push!(values, dat_i[:,value_column_name])
        catch e
            if isa(e, ArgumentError)
                error("value_column_name was not found in provided data. Please set the kwarg: value_column_name to the correct value")
            end
        end
    end
    return times, values, variables
end 

function interpolate_covariates(X::DataFrame, time_column_name, variable_column_name, value_column_name)
    times, values, variables = process_long_format_data(X,time_column_name, variable_column_name,value_column_name)
    interpolations = [linear_interpolation(times[i], values[i],extrapolation_bc = Interpolations.Flat()) for i in eachindex(times)]
    function covariates(t)
        return [interpolation(t) for interpolation in interpolations]
    end 
    return covariates, variables
end 


function process_long_format_data(data, time_column_name, series_column_name,  variable_column_name, value_column_name)
    try
        unique(data[:,variable_column_name])
    catch e
        if isa(e, ArgumentError)
            error("variable_column_name was not found in provided data. Please set the kwarg: variable_column_name to the correct value")
        end
    end
    variables = unique(data[:,variable_column_name])
    series = unique(data[:,series_column_name])
    times = []; values = []
    for series_i in series 
        times_var = []; values_var = []
        for var in variables
            inds = (data[:,variable_column_name] .== var) .& (data[:,series_column_name] .== series_i) 
            dat_i = data[inds,:]
            push!(times_var, dat_i[:,time_column_name])
            try
                push!(values_var, dat_i[:,value_column_name])
            catch e
                if isa(e, ArgumentError)
                    error("value_column_name was not found in provided data. Please set the kwarg: value_column_name to the correct value")
                end
            end
        end
        push!(times, times_var)
        push!(values, values_var)
    end
    return times, values, variables
end 


function interpolate_covariates(data, time_column_name, series_column_name,  variable_column_name, value_column_name)
    times, values, variables = process_long_format_data(data, time_column_name, series_column_name,  variable_column_name, value_column_name)
    interpolations = [[linear_interpolation(times[j][i], values[j][i],extrapolation_bc = Interpolations.Flat()) for i in eachindex(times[j])] for j in eachindex(times)]
    function covariates(t,series)
        return [interpolation(t) for interpolation in interpolations[round(Int,series)]]
    end 
    return covariates, variables
end 

