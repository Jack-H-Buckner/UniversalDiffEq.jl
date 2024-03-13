using Interpolations

function interpolate_covariates(X)
    N, dims, T, times, data, dataframe = process_data(X)
    interpolations = [linear_interpolation(times, data[i,:],extrapolation_bc = Interpolations.Flat()) for i in 1:dims]
    function covariates(t)
        return [interpolation(t) for interpolation in interpolations]
    end 
    return covariates
end 



function interpolate_covariates_multi(X)
    data_sets, times = process_multi_data2(X)
    dims = size(data_sets[1])[1]
    interpolations = [[linear_interpolation(times[j], data_sets[j][i,:],extrapolation_bc = Interpolations.Flat()) for i in 1:dims] for j in eachindex(times)]
    function covariates(t,j)
        return [interpolation(t) for interpolation in interpolations[round(Int,j)]]
    end 
    return covariates
end 


function plot_covariates(model)
    dims = length(model.process_model.covariates(0))
    plt = plot(model.times,broadcast(t -> model.process_model.covariates(t)[1],model.times), xlabel = "Time", ylabel = "X", label = "")
    for i in 2:dims
        plot!(plt,model.times,broadcast(t -> model.process_model.covariates(t)[i],model.times), label = "")
    end 
    return plt
end 
