

function get_final_state(UDE::UDE)
    return UDE.parameters.uhat[:,end]
end


"""
    print_parameter_estimates(UDE::UDE)

Prints the values of the known dynamics parameters estimated by the UDE model.
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

Returns the values of the weights and biases of the neural network.
"""
function get_NN_parameters(UDE::UDE)
    return UDE.parameters.process_model.NN
end



"""
    get_right_hand_side(UDE::UDE)

Returns the right-hand side of the differential equation (or difference equation) used to build the process model.

The function will take the state vector `u` and time `t` if the model does not include covariates. If covariates are included, then the arguments are the state vector `u` , covariates vector `x`, and time `t`.
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



function predict(UDE::UDE,test_data::DataFrame;df = true)
    check_test_data_names(UDE.data_frame, test_data)
    inits, obs, preds = predictions(UDE,test_data)
    if df
        N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
        names = vcat(["t"],[string("x",i) for i in 1:dims])
        return DataFrame(Array(vcat(times[2:end]',preds)'),names)
    else
        return preds
    end

end


"""
    forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})

Predictions from the trained UDE model starting at `u0` and saving values at `times`. Assumes `u0` is the value at initial time `times[1]`
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


 """
    forecast(UDE::UDE, u0::AbstractVector{}, t0::Real, times::AbstractVector{})

 predictions from the trained model `UDE` starting at `u0` saving values at `times`. Assumes `u0` occurs at time `t0` and `times` are all larger than `t0`.
 """
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



function forecast(UDE::UDE, test_data::DataFrame)
    check_test_data_names(UDE.data_frame, test_data)
    u0 = UDE.parameters.uhat[:,end]
    N, dims, T, times, data, dataframe = process_data(test_data,UDE.time_column_name)
    df = forecast(UDE, u0, UDE.times[end], times)
    return df
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
    test_data = data[(N_train+1):end,:]

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
    test_data = data[(N_train+1):end,:]

    # build model
    model = model.constructor(train_data)
    gradient_descent!(model, step_size = step_size, maxiter = maxiter)

    # forecast
    u0 = model.parameters.uhat[:,end]
    times = test_data.t
    predicted_data = forecast(model, u0, times)


    return abs.(Matrix(predicted_data[:,2:end]) .- Matrix(test_data[:,2:end]))
end

