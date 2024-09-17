function get_residuals(UDE)
    inits, obs, preds = predictions(UDE,UDE.data_frame)
    proc_resids = preds .- obs
    link = x -> UDE.observation_model.link(x,UDE.parameters.observation_model)
    obs_resids = mapslices(link,UDE.parameters.uhat,dims=1) .- UDE.data  
    return proc_resids, obs_resids
end 

function get_residual_sample_cov(UDE)
    
    proc_resids, obs_resids = get_residuals(UDE)
    d = size(proc_resids)[1]
    proc_cov = zeros(d,d)
    N = size(proc_resids)[2]
    for i in 1:size(proc_resids)[2]
        proc_cov .+= (proc_resids[:,i] * transpose(proc_resids[:,i] ))./(N-1)
    end

    d = size(obs_resids)[1]
    obs_cov = zeros(d,d)
    N = size(obs_resids)[2]
    for i in 1:size(obs_resids)[2]
        obs_cov .+= (obs_resids[:,i] * transpose(obs_resids[:,i] ))./(N-1)
    end
    return proc_cov, obs_cov
end 




# uses emperical variances to calcaulate the sum of process an dobservaiton errors 
# and then uses the ratios of observaiton and process errors used to train the model to 
# divide the total residual variance into process and observation loss. 
function particle_filter_states(UDE,u0,t0,data,N;nugget = 10^-2)

    # Get data range for extrapolation
    uhats = UDE.parameters.uhat
    umax = mapslices(max_, UDE.parameters.uhat, dims = 2);umax=reshape(umax,length(umax))
    umin = mapslices(min_, UDE.parameters.uhat, dims = 2);umin=reshape(umin,length(umin))
    umean = mapslices(mean, UDE.parameters.uhat, dims = 2);umean=reshape(umean,length(umean))

    # Get residuals
    estimated_map = (x,t,dt) -> UDE.process_model.forecast(x,t,dt,UDE.parameters.process_model,umax,umin,umean)

    # Estimate residual variances
    proc_cov, obs_cov = get_residual_sample_cov(UDE)

    d = size(proc_cov)[1]
    nugget = nugget * Matrix(I,d,d)

    reg_weight,proc_weight, obs_weight = UDE.weights

    total_loss_var = 1/obs_weight^2 + 1/proc_weight^2
    obs_loss_var = 1/obs_weight^2 / total_loss_var 
    proc_loss_var = 1/proc_weight^2 / total_loss_var 

    total_cov = proc_cov .+ obs_cov .+ nugget 

    obs_cov = obs_loss_var .* total_cov 
    proc_cov = proc_loss_var .* total_cov

    # observation error likeihood
    obs_dsn = Distributions.MvNormal(zeros(size(obs_cov)[1]), obs_cov)

    # get process_residuals as estimator for future process errors 
    proc_dsn = Distributions.MvNormal(zeros(size(obs_cov)[1]), proc_cov)

    # initialize sampels 
    d = size(proc_cov)[1]
    ut = transpose(repeat(transpose(u0),N))

    # initialize accumulators
    T = size(data)[1]
    particles = zeros(T,d,N)
    means = zeros(size(data)[1],d)

    # run first time step beforfore for loop
    dt = data[1,UDE.time_column_name] - t0
    t = t0
    u_obs = Vector(data[1,names(data) .!= UDE.time_column_name])

    # calcualte time step
    ut = mapslices(u -> estimated_map(u,t,dt),ut,dims=1)
    ut .+= rand(proc_dsn, N)

    # resample particles 
    
    weights = reshape(mapslices(u -> pdf(obs_dsn,u .- u_obs),ut,dims=1),N)
    inds = sample(1:size(ut)[2], Weights(weights), N, replace=true)
    ut = ut[:,inds]
    
    # update accumuatlors 
    means[1,:] = mapslices(mean,ut,dims=2)
    particles[1,:,:] = ut

    i = 0
    for t in eachindex(data[2:end,UDE.time_column_name])
        i +=1
        
        dt = data[t+1,UDE.time_column_name] - data[t,UDE.time_column_name]
        t_i = data[t,UDE.time_column_name]
        u_obs = Vector(data[i+1,names(data) .!= UDE.time_column_name])

        # calcualte time step
        ut = mapslices(u -> estimated_map(u,t_i,dt),ut,dims=1)
        ut .+= rand(proc_dsn, N)

        # resample particles 
        weights = reshape(mapslices(u -> pdf(obs_dsn,u .- u_obs),ut,dims=1),N)
        inds = sample(1:size(ut)[2], Weights(weights), N, replace=true)
        ut = ut[:,inds]

        # update accumuatlors 
        means[i+1,:] = mapslices(mean,ut,dims=2)
        particles[1+i,:,:] = ut

    end

    means = hcat(data[:,UDE.time_column_name],means)
    df = DataFrame(means, vcat([UDE.time_column_name],names(data)[names(data) .!= UDE.time_column_name]))

    return df, particles
end




function one_step_ahead(UDE,u0,t0, data;N=1000,nugget = 10^-1.5)
 
    umax = mapslices(max_, UDE.parameters.uhat, dims = 2);umax=reshape(umax,length(umax))
    umin = mapslices(min_, UDE.parameters.uhat, dims = 2);umin=reshape(umin,length(umin))
    umean = mapslices(mean, UDE.parameters.uhat, dims = 2);umean=reshape(umean,length(umean))

    df, particles = particle_filter_states(UDE,u0,t0,data,N,nugget=nugget)
    estimated_map = (x,t,dt) -> UDE.process_model.predict(x,t,dt,UDE.parameters.process_model)[1]#,umax,umin,umean)

    # accumulators
    predicted = deepcopy(data[1:end,:])
    observed = deepcopy(data[1:end,:])


    t = data[1,UDE.time_column_name]
    dt = t - t0

    pred = estimated_map(u0,t0,dt) # 
    predicted[1,names(predicted) .!= UDE.time_column_name] .= pred
    observed[1,names(observed) .!= UDE.time_column_name] .= Vector(data[1,names(observed) .!= UDE.time_column_name])

    for i in 2:size(data)[1]

        t0 = data[i-1,UDE.time_column_name]
        t = data[i,UDE.time_column_name]
        dt = t - t0

        u0 = Vector(df[i-1,names(df) .!= UDE.time_column_name])
        pred = estimated_map(u0,t0,dt) 
        predicted[i,names(predicted) .!= UDE.time_column_name] .= pred
        observed[i,names(observed) .!= UDE.time_column_name] .= Vector(data[i,names(observed) .!= UDE.time_column_name])
    
    end

    return predicted, observed, df
end

function one_step_ahead_mse(UDE,u0,t0,data;N=1000)

    preds,obs,df = one_step_ahead(UDE,u0,t0,data;N=N)
    preds = Matrix(preds[:,names(preds) .!= UDE.time_column_name])
    obs = Matrix(obs[:,names(obs) .!= UDE.time_column_name])

    mses = zeros(size(obs)[2])
    for i in 1:size(obs)[2]
        mses[i] = sum((obs[i] .- preds[i]).^2)/length(obs[i])
    end

    return mses
end


function rmse_(testing_data, predicted_data)

    k = length(testing_data)
    d = size(testing_data[1][:,2:end])[2]
    mses = zeros(d)
    for i in 1:k
        for j in 1:d
            test = Vector(testing_data[i][:,1+j])
            pred = Vector(predicted_data[i][:,1+j])
            mses[j] += 1/k * sum((test .- pred).^2)/length(pred)
        end
    end
    
    return mses
end 



"""
cross_validation_kfold(model::UDE; kwagrs...)

This funciton approximates model performance on out of sample data by leaving blocks of consequtive observaitons out of the training data. The model is trained on the remiaining observations and the and the one step ahead prediction accuracy is calcualted on the testing data set. This procedure is repeated k times.  
...
# Arguments
k = 10 - the number of testing data sets
BFGS = false - weather or not to train the models with the BFGS algorithm 
step_size = 0.05 - the step size for the first round of gradient descent optimization 
maxiter = 500 - the number of iterations for the first round of gradinet descent 
step_size2 = 0.05 - the step size for the second round of gradient descent 
maxiter2 = 500 - the number of iterations for the second round of gradient descent
N = 1000 - the number of particle to use in the particle filter algorithm that estiamtes the states in the out of sample data 
nugget = 10^-10 - a small number to added variance terms in the particle filter algorith to improve numerical stability
...
"""
function cross_validation_kfold(model::UDE; k = 10, BFGS = false, step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500, N=1000, nugget = 10^-10)
    
    # get final time
    dat = model.data_frame
    T = size(dat)[1]
    spacing = floor(Int, T/k)

    training_data = []
    testing_data = []

    starts = [ ]

    ind = spacing
    for i in 1:(k-1)
        push!(training_data,vcat(dat[1:ind,:],dat[(ind+spacing+1):end,:]))
        push!(testing_data, dat[(ind+1):(ind+spacing),:] )
        push!(starts, ind)
        ind += spacing
    end

    if ind < size(dat)[1]
        push!(training_data,dat[1:ind,:])
        push!(testing_data, dat[(ind+1):end,:] )
        push!(starts, ind)
    end

    t_skips = dat[starts,model.time_column_name]
    
    d = size(model.data_frame)[2]-1
    mses = zeros(k,d)

    estiamted_states = Array{Any}(nothing, length(training_data))
    predictions = Array{Any}(nothing, length(training_data))
    models = Array{Any}(nothing, length(training_data))
    testing_data_sets = Array{Any}(nothing, length(training_data))
    training_data_sets = Array{Any}(nothing, length(training_data))

    Threads.@threads for i in eachindex(training_data)
        println("Training data set ", i)
        t_skip = [t_skips[i]] 

        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, t_skip, maxiter = maxiter, step_size = step_size)   

        if BFGS
            try
                BFGS!(model_i, t_skip, verbose = false)
            catch
                println("BFGS failed running gradient_descent")
                gradient_descent!(model_i, t_skip, maxiter = maxiter2, step_size = step_size2)                 
            end   
        else
            gradient_descent!(model_i, t_skip, maxiter = maxiter2, step_size = step_size2)  
        end

        # forecast
        u0 = model_i.parameters.uhat[:,starts[i]]
        t0 = t_skip[1]
   
        preds_,obs,df = one_step_ahead(model_i,u0,t0,testing_data[i];N=N,nugget=nugget)
        preds = Matrix(preds_[:,names(preds_) .!= model_i.time_column_name])
        obs = Matrix(obs[:,names(obs) .!= model_i.time_column_name])

        mse = zeros(size(obs)[2])

        for i in 1:size(obs)[2]
            mse[i] = sum((obs[:,i] .- preds[:,i]).^2)/length(obs[:,i])
        end

        mses[i,:] .= mse

        estiamted_states[i] = df
        predictions[i] = preds_
        models[i] = model_i
        testing_data_sets[i] = testing_data[i]
        training_data_sets[i] = training_data[i]
           
    end 
    
    total_rmse = sqrt.(mean(mses))
    variable_rmse = sqrt.(mean(eachrow(mses)))
    diagnostics = (estiamted_states, predictions, models, testing_data_sets, training_data_sets)

    return total_rmse, variable_rmse, diagnostics 
    
end 



function kfold_diagnositcs_plot(diagnostics; range_45 = 0.25, time_series_plot_dims = (1000,1000), comparison_plot_dims = (500,500))


    estiamted_states, predictions, models, testing_data_sets, training_data_sets = diagnostics
    
    plts = []
    d = size(estiamted_states[1])[2]-1
    k = length(testing_data_sets)
    
    match_plts = [plot() for j in 1:d]
    colors = ["#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", "#60A1A5", "#3B80E4", "#9760A5",
                "#753BE4", "#DA3BE4", "#E43B6F", "#60A56D", "#99A560", "#A56060","#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", "#60A1A5", "#3B80E4", "#9760A5",
                "#753BE4", "#DA3BE4", "#E43B6F", "#60A56D", "#99A560", "#A56060","#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", "#60A1A5", "#3B80E4", "#9760A5",
                "#753BE4", "#DA3BE4", "#E43B6F", "#60A56D", "#99A560", "#A56060","#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", "#60A1A5", "#3B80E4", "#9760A5",
                "#753BE4", "#DA3BE4", "#E43B6F", "#60A56D", "#99A560", "#A56060","#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", "#60A1A5", "#3B80E4", "#9760A5",
                "#753BE4", "#DA3BE4", "#E43B6F", "#60A56D", "#99A560", "#A56060"]
    
    for i in 1:k

        plts_i = []
        
        for j in 1:d
            testing_esimates = estiamted_states[i]
            testing = testing_data_sets[i]
            training = training_data_sets[i]
            training_estiamtes = models[i].parameters.uhat
            preds = predictions[i]
            t0 = testing_esimates[1,1]
    
            if (i ==1) & (j == 1)
                plt = Plots.plot(testing_esimates[:,1], testing_esimates[:,1+j],label = "states est.", markersize = 2.0, color = "#FF8A2E")
                Plots.plot!(training[training[:,1] .< t0,1], training_estiamtes[j,training[:,1] .< t0],label = "state est", markersize = 2.0, color = "grey")
                Plots.plot!(training[training[:,1] .> t0,1], training_estiamtes[j,training[:,1] .> t0],label = "", markersize = 2.0, color = "grey")
                Plots.scatter!(training[:,1], training[:,1+j],label = "training data", markersize = 3.0, markerstrokewidth = 0.0, color = "#E9CB43")
                Plots.scatter!(testing[:,1], testing[:,1+j],label = "testing data", markersize = 3.0, markerstrokewidth = 0.0, color = "#2EE5FF")
                Plots.scatter!(preds[:,1], preds[:,1+j], label = "predictions", markersize = 3.0, markerstrokewidth = 0.0, color = "#30923B")
            else
                plt = Plots.plot(testing_esimates[:,1], testing_esimates[:,1+j],label = "", markersize = 2.0, color = "#FF8A2E")
                Plots.plot!(training[training[:,1] .< t0,1], training_estiamtes[j,training[:,1] .< t0],label = "", markersize = 2.0, color = "grey")
                Plots.plot!(training[training[:,1] .> t0,1], training_estiamtes[j,training[:,1] .> t0],label = "", markersize = 2.0, color = "grey")
                Plots.scatter!(training[:,1], training[:,1+j],label = "", markersize = 3.0, markerstrokewidth = 0.0, color = "#E9CB43")
                Plots.scatter!(testing[:,1], testing[:,1+j],label = "", markersize = 3.0, markerstrokewidth = 0.0, color = "#2EE5FF")
                Plots.scatter!(preds[:,1], preds[:,1+j], label = "", markersize = 3.0, markerstrokewidth = 0.0, color = "#30923B")
            end
    
            if i == 1
                Plots.plot!(match_plts[j],[-range_45,range_45],[-range_45,range_45], color = "grey", linestyle = :dash, label = "")
            end 
    
            if j ==1
                Plots.scatter!(match_plts[j], testing[2:end,1+j].-testing_esimates[1:(end-1),1+j],
                                         preds[2:end,1+j].-testing_esimates[1:(end-1),1+j], color = colors[i], 
                                        label = string("fold: ", i))
            else
                Plots.scatter!(match_plts[j], testing[2:end,1+j].-testing_esimates[1:(end-1),1+j], 
                            preds[2:end,1+j].-testing_esimates[1:(end-1),1+j], color = colors[i], label = "")
            end
            push!(plts_i, plt)
        end
        push!(plts, plts_i)
    end
    
    
    plts = [plot(plts_i ..., layout = (1,d)) for plts_i in plts]
    plt2 = plot(match_plts ..., size = comparison_plot_dims)
    plt1 = plot(plts ..., layout = (k,1), size = time_series_plot_dims)
    return plt1, plt2
end




function leave_future_out_cross_validation(model::UDE; forcast_horizon = 1, k = 10, skip = true, BFGS = false, step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500)
    
    if typeof(skip) != Int64
        skip = forcast_horizon
    end

    data = model.data_frame

    @assert skip*k+forcast_horizon < size(data)[1]

    training_data = []
    testing_data = []

    ind = size(data)[1] - forcast_horizon
    for i in 1:k
        push!(training_data,data[1:ind,:])
        push!(testing_data, data[(ind+1):(ind+forcast_horizon),:])
        ind += -skip
    end

    d = size(model.data_frame)[2]-1
    mses = zeros(k,d)

    predictions = Array{Any}(nothing, length(training_data))
    models = Array{Any}(nothing, length(training_data))
    testing_data_sets = Array{Any}(nothing, length(training_data))
    training_data_sets = Array{Any}(nothing, length(training_data))

    Threads.@threads for i in eachindex(training_data)
        println("Training data set ", i)

        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, maxiter = maxiter, step_size = step_size)   

        if BFGS
            try
                BFGS!(model_i,verbose = false)
            catch
                println("BFGS failed running gradient_descent")
                gradient_descent!(model_i,maxiter = maxiter2, step_size = step_size2)                 
            end   
        else
            gradient_descent!(model_i, maxiter = maxiter2, step_size = step_size2)  
        end

        # forecast
        # forecast
        u0 = model_i.parameters.uhat[:,end]
        t0 = model_i.times[end]
        times = testing_data[i][:,model.time_column_name]
        predicted_data = forecast(model_i, u0, times)
        predicted_data = DataFrame(predicted_data,names(testing_data[i]))
        testing = testing_data[i]

        preds = Matrix(predicted_data[:,names(predicted_data) .!= model_i.time_column_name])
        testing  = Matrix(testing[:,names(testing) .!= model_i.time_column_name])
        mse = zeros(size(testing)[2])
        for j in 1:size(testing)[2]
            mse[j] = sum((testing[:,j] .- preds[:,j]).^2)/length(testing[:,j])
        end

        mses[i,:] .= mse

        predictions[i] = predicted_data
        models[i] = model_i
        testing_data_sets[i] = testing_data[i]
        training_data_sets[i] = training_data[i]
           
    end 
    
    total_rmse = sqrt.(mean(mses))
    variable_rmse = sqrt.(mean(eachrow(mses)))
    diagnostics = (predictions, models, testing_data_sets, training_data_sets)

    return total_rmse, variable_rmse, diagnostics 
    
end 




function leave_future_out_diagnositcs_plot(diagnostics, range_45 = 0.25, time_series_plot_dims = (1000,1000), comparison_plot_dims = (500,500))

    predictions, models, testing_data_sets, training_data_sets = diagnostics
    
    plts = []
    d = size(training_data_sets[1])[2]-1
    k = length(testing_data_sets)
    
    match_plts = [plot() for j in 1:d]
    colors = ["#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4",  "#60A1A5", "#3B80E4", "#9760A5","#753BE4", "#DA3BE4", "#E43B6F",
            "#60A56D", "#99A560", "#A56060","#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", 
            "#60A1A5", "#3B80E4", "#9760A5","#753BE4", "#DA3BE4", "#E43B6F","#60A56D", "#99A560", "#A56060","#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", 
            "#60A1A5", "#3B80E4", "#9760A5","#753BE4", "#DA3BE4", "#E43B6F","#60A56D", "#99A560", "#A56060","#E4B73B", "#A56075","#9DE43B", "#3BE487", "#6067A5","#3BC5E4", 
            "#60A1A5", "#3B80E4", "#9760A5","#753BE4", "#DA3BE4", "#E43B6F","#60A56D", "#99A560", "#A56060"]

    tbegin = training_data_sets[1][1,1]
    tend = testing_data_sets[1][end,1]
    for i in 1:k

        plts_i = []
        
        for j in 1:d

            testing = testing_data_sets[i]
            training = training_data_sets[i]
            training_estiamtes = models[i].parameters.uhat
            preds = predictions[i]
            t0 = training[1,1]
    
            if (i ==1) & (j ==1)
                plt = Plots.plot(training[training[:,1] .< t0,1], training_estiamtes[j,training[:,1] .< t0],label = "state est", markersize = 2.0, color = "grey")
                Plots.scatter!(training[:,1], training[:,1+j],label = "training data", markersize = 3.0, markerstrokewidth = 0.0, color = "#E9CB43")
                Plots.scatter!(testing[:,1], testing[:,1+j],label = "testing data", markersize = 3.0, markerstrokewidth = 0.0, color = "#2EE5FF")
                Plots.scatter!(preds[:,1], preds[:,1+j], label = "predictions", markersize = 3.0, markerstrokewidth = 0.0, color = "#30923B")
                Plots.plot!(xlims = (tbegin,tend))
            else
                plt = Plots.plot(training[training[:,1] .< t0,1], training_estiamtes[j,training[:,1] .< t0],label = "", markersize = 2.0, color = "grey")
                Plots.scatter!(training[:,1], training[:,1+j],label = "", markersize = 3.0, markerstrokewidth = 0.0, color = "#E9CB43")
                Plots.scatter!(testing[:,1], testing[:,1+j],label = "", markersize = 3.0, markerstrokewidth = 0.0, color = "#2EE5FF")
                Plots.scatter!(preds[:,1], preds[:,1+j], label = "", markersize = 3.0, markerstrokewidth = 0.0, color = "#30923B")
                Plots.plot!(xlims = (tbegin,tend))
            end
    
            if i == 1
                Plots.plot!(match_plts[j],[-range_45,range_45],[-range_45,range_45], color = "grey", linestyle = :dash, label = "")
            end 
    
            if j ==1
                
                Plots.scatter!(match_plts[j], testing[2:end,1+j],preds[2:end,1+j], color = colors[i],label = string("fold: ", i))
            else
                Plots.scatter!(match_plts[j], testing[2:end,1+j],preds[2:end,1+j], color = colors[i], label = "")
            end
            push!(plts_i, plt)
        end
        push!(plts, plts_i)
    end
    
    
    plts = [plot(plts_i ..., layout = (1,d)) for plts_i in plts]
    plt2 = plot(match_plts ..., size = comparison_plot_dims)
    plt1 = plot(plts ..., layout = (k,1), size = time_series_plot_dims)

    return plt1, plt2
end































































## leave future out cross validation
function leave_future_out(model::UDE; forecast_length = 10,  forecast_number = 10, spacing = 1, using_BFGS=false, 
                                step_size = 0.05, maxiter = 500, step_size2 = 0.01, maxiter2 = 500)
    
    # get final time
    data = model.data_frame
    T = length(data[:,model.time_column_name])
    start1 = T - forecast_length - spacing*(forecast_number-1)
    starts = [start1 + spacing *i for i in 0:(forecast_number-1)]
    training_data = [data[1:t0,:] for t0 in starts]
    testing_data = [data[t0:(t0+forecast_length),:] for t0 in starts]
    
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]
    mses = zeros(forecast_number)

    Threads.@threads for i in 1:forecast_number

        println("Training data set ", i)
        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, step_size = step_size, maxiter = maxiter) 
           
        if using_BFGS
            try
                BFGS!(model_i,verbose = false)
            catch
                println("BFGS failed running gradient_descent")
                gradient_descent!(model_i, step_size = step_size2, maxiter = maxiter2)                
            end   
        else
            gradient_descent!(model_i, step_size = step_size2, maxiter = maxiter2) 
        end
                    
        # forecast
        u0 = model_i.parameters.uhat[:,end]
        t0 = model_i.times[end]
        times = testing_data[i][:,model.time_column_name]
        predicted_data = forecast(model_i, u0, times)
        predicted_data= DataFrame(predicted_data,names(testing_data[i]))
        
        mse = one_step_ahead_mse(model_i,u0,t0,testing_data[i];N=1000)
        mses[i] = mse

            
        SE = copy(predicted_data)
        SE[:,2:end] .= (predicted_data[:,2:end] .- testing_data[i][:,2:end]).^2
        
        push!(standard_errors[Threads.threadid()], SE)
        push!(predicted[Threads.threadid()], predicted_data)             
    end 
    
    standard_error = standard_errors[1]
    predicted_data = predicted[1]         
    for i in 2:Threads.nthreads()
                            
        standard_error = vcat(standard_error,standard_errors[i])
        predicted_data = vcat(predicted_data,predicted[i])
                            
    end
    
    return training_data, testing_data, standard_error, predicted_data, mses
    
end 




                                        
"""
    leave_future_out_cv(model::UDE; kwargs ...)
    
Runs K fold leave future out cross validation and returns the mean squared forecasting error and a plot to visualize the model fits.

...
# Arguments 
model - the UDE model to test
forecast_length = 5 - The number of data points to forecast
forecast_number = 10 - The number of trainin gnad testing data sets to create
spacing = 2 - The number of data points to skip between the end of each new triaining data set
BFGS=false - use BFGS algorithm to train the model
step_size = 0.05 - step size for first run of the ADAM algorithm 
maxiter = 500 - maximum iterations for first trial of ADAM
step_size2 = 0.01 - step size for second iteration of ADAM, only used of BFGS is false
maxiter2 = 500 - step size for second iteration of ADAM, only used of BFGS is false
...
"""
function leave_future_out_cv(model::UDE;forecast_length = 5,  forecast_number = 10, spacing = 2, BFGS=false, 
                                step_size = 0.05, maxiter = 500, step_size2 = 0.01, maxiter2 = 500)
    
    training_data, testing_data, standard_errors, predicted_data, mses = leave_future_out(model;forecast_length=forecast_length,forecast_number=forecast_number,using_BFGS=BFGS,spacing=spacing,step_size=step_size,maxiter=maxiter,step_size2 = step_size2, maxiter2 = maxiter2)
    RMSE = rmse_(testing_data, predicted_data)
    rmse = sqrt(sum(mses)/length(mses))

    return rmse, RMSE                            
end




# multiple time series

# find intervals of time spanned by at least one data set 
function t_in_intervals(t,tmins,tmaxs)
    @assert length(tmins) == length(tmaxs)
    for i in eachindex(tmins)
        if (t >= tmins[i]) & (t <= tmaxs[i])
            return true
        end  
    end
    return false
end 

function edge(tmins,tmaxs,dt)
    tmin = tmins[argmin(tmins)]
    tmax = tmaxs[argmax(tmaxs)]
    ts = tmin:dt:tmax
    test = broadcast(t -> t_in_intervals(t,tmins,tmaxs), ts)
    in_ = test[1]; edges = [tmin]
    for t in 2:length(ts)
        if in_ 
            if !test[t]
                push!(edges,ts[t])
            end
        else
            if test[t]
                push!(edges,ts[t-1])
            end
        end
        in_ = test[t]
    end
    return push!(edges,tmax)
end

function interval_lengths(edges)
    L = 0
    for i in 2:2:length(edges)
        L += edges[i] - edges[i-1]
    end
    return L
end

# K - number of intervals to leave out
# T - length of intervals left out 
function intervals(K,tmins,tmaxs,dt)
    edges = edge(tmins,tmaxs,dt)
    L = interval_lengths(edges)
    l = L/K
    lower = []
    for i in 2:2:length(edges)
        lower= vcat(lower,collect((edges[i-1]+dt):l:(edges[i]+dt)))
    end
    return lower[1:(end-1)]
end


max_value = x -> x[argmax(x)]



function multi_rmse(testing_data, predicted_data)
    k = length(testing_data)
    MSE = 0
    for i in 1:k
        test = Matrix(testing_data[i][:,3:end])
        pred = Matrix(predicted_data[i][:,3:end])
        MSE += 1/k * sum((test .- pred).^2)/length(pred)
    end
    
    return sqrt(RMSE)
end 

function kfold_cv(model::MultiUDE;k=10,leave_out=5,BFGS = false,step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500)
    
    # get final time
    df = model.data_frame
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(df,model.time_column_name,model.series_column_name)

    # split into training and testing data by leaving out
    tmins = [ times[t] for t in starts]
    tmaxs = [ times[starts[i] + lengths[i]-1] for i in eachindex(starts)]

    lower = intervals(k,tmins,tmaxs,10^-5)
    
    training_data = [df[(df[:,model.time_column_name] .<= t0) .| (df[:,model.time_column_name] .> (t0+leave_out)),:] for t0 in lower]
    testing_data = [df[(df[:,model.time_column_name] .> t0) .& (df[:,model.time_column_name] .<= (t0+leave_out)),:] for t0 in lower]
    
    # find time and series combos to skip  prediction step
    skips = [df[(df[:,model.time_column_name] .<= t0),:] for t0 in lower]
    skips = [combine(groupby(df,Symbol(model.series_column_name)), Symbol(model.time_column_name)  => max_value => Symbol(model.time_column_name)) for df in skips]
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]

    Threads.@threads for i in 1:k  
        print("k: ", i, "   ")
        skip = skips[i]
        
        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, skip, step_size = step_size, maxiter=maxiter, verbose = false)   

        if BFGS
            try
                BFGS!(model_i, skip, verbose = false)
            catch
                println("BFGS failed running gradient_decent")
                gradient_descent!(model_i, skip, step_size = 0.01)                 
            end  
        else
            gradient_descent!(model_i, skip, step_size = step_size2, maxiter=maxiter2)   
        end
        
        predictions =[]
        for s in skip[:,model.series_column_name]

            ind = model.series_labels[model.series_labels[:,"label"].==s,"index"]
            # get initial time point and state variables
            N_i, T_i, dims_i, data_i, times_i,  dataframe_i, series_i, inds_i, starts_i, lengths_i = process_multi_data(training_data[i],model.time_column_name,model.series_column_name)
            t0 = skip[skip[:,model.series_column_name] .== s,model.time_column_name][1]
            uhat_series = model_i.parameters.uhat[:,starts_i[ind][1]:(starts_i[ind][1]+lengths_i[ind][1]-1)]
            times_series = times_i[starts_i[ind][1]:(starts_i[ind][1]+lengths_i[ind][1]-1)]
            u0 = uhat_series[:,times_series .== t0]

            # get time from testing data
            times_test = testing_data[i][testing_data[i][:,model.series_column_name] .== s,model.time_column_name]
            predicted_data = forecast(model_i, u0[:,1], t0, times_test, s)

            predicted_data= DataFrame(predicted_data,names(testing_data[i])[1:(end-1)])

            # add time to predictions
            push!(predictions, predicted_data)
        end 
        
        # knit together predictions data frame
        predicted_data = predictions[1]
        for i in 2:length(predictions)
            predicted_data = vcat(predicted_data,predictions[i])
        end

        #SE = copy(predicted_data)
        #SE[:,3:end] .= (predicted_data[:,3:end] .- testing_data[i][:,3:end]).^2
        
        #push!(standard_errors[Threads.threadid()], SE)
        push!(predicted[Threads.threadid()], predicted_data)             
    end 
    
    #standard_error = standard_errors[1]
    predicted_data = predicted[1]         
    for i in 2:Threads.nthreads()
                            
        #standard_error = vcat(standard_error,standard_errors[i])
        predicted_data = vcat(predicted_data,predicted[i])
                            
    end
    # standard_error

    return  multi_rmse(testing_data, predicted_data)
    
end 



function kfold(model::UDE;k=10,leave_out=5,BFGS = false,step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500)
    
    # get final time
    data = model.data_frame
    T = size(data)[1]
    spacing = floor(Int, T/k)
    starts = [floor(Int, 1+ spacing *i) for i in 0:(k-1)]
    t_skips = data[starts,model.time_column_name] 
    training_data = [vcat(data[1:t0,:],data[(t0+leave_out+1):end,:]) for t0 in starts]
    testing_data = [data[(t0+1):(t0+leave_out),:] for t0 in starts]
    
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]
    

    d = size(model.data_frame)[2]-1
    mses = zeros(d)

    Threads.@threads for i in 1:k
        println("Training data set ", i)
        t_skip = [t_skips[i]] 

        model_i = 0
        if model.X == 0
            model_i = model.constructor(training_data[i])
        else
            model_i = model.constructor(training_data[i],model.X)
        end
                        
        gradient_descent!(model_i, t_skip, maxiter = maxiter, step_size = step_size)   

        if BFGS
            try
                BFGS!(model_i, t_skip, verbose = false)
            catch
                println("BFGS failed running gradient_descent")
                gradient_descent!(model_i, t_skip, maxiter = maxiter2, step_size = step_size2)                 
            end   
        else
            gradient_descent!(model_i, t_skip, maxiter = maxiter2, step_size = step_size2)  
        end
        # forecast
        u0 = model_i.parameters.uhat[:,starts[i]]
        t0 = t_skip[1]

        times = testing_data[i][:,model.time_column_name]
        predicted_data = forecast(model_i, u0, t0, times)
        predicted_data= DataFrame(predicted_data,names(testing_data[i]))
        
        mse = one_step_ahead_mse(model_i,u0,t0,testing_data[i];N=1000)
        mses .+= mse./k

        SE = copy(predicted_data)
        SE[:,2:end] .= (predicted_data[:,2:end] .- testing_data[i][:,2:end]).^2
        
        push!(standard_errors[Threads.threadid()], SE)
        push!(predicted[Threads.threadid()], predicted_data)             
    end 
    
    standard_error = standard_errors[1]
    predicted_data = predicted[1]   

    for i in 2:Threads.nthreads()
                            
        standard_error = vcat(standard_error,standard_errors[i])
        predicted_data = vcat(predicted_data,predicted[i])
                            
    end
    
    return testing_data, predicted_data, mses
    
end 


"""
    kfold_cv(model::UDE; kwargs ...)
    
Runs K fold leave future out cross validation and returns the mean squared forecasting error and a plot to visualize the model fits.

...
# Arguments 
model - a UDE model object
k=10 - the number of testing and taining data sets to build
leave_out=5 - the number of data points to leave out in each testingdata set
BFGS = false - use the BFGS algorithm to train the model
step_size = 0.05 - step size for first run of the ADAM algorithm 
maxiter = 500 - maximum iterations for first trial of ADAM
step_size2 = 0.01 - step size for second iteration of ADAM, only used of BFGS is false
maxiter2 = 500 - step size for second iteration of ADAM, only used of BFGS is false
...
"""
function kfold_cv(model::UDE;k=10,leave_out=5,BFGS = false,step_size = 0.05, maxiter = 500, step_size2 = 0.05, maxiter2 = 500)
  
    testing_data, predicted_data, one_step_mses = kfold(model;k=k,leave_out=leave_out,BFGS=BFGS,step_size=step_size,maxiter=maxiter,step_size2=step_size2,maxiter2=maxiter2)

    forecast_mses = rmse_(testing_data, predicted_data)

    one_step = sqrt.(one_step_mses)
    forecast = sqrt.(forecast_mses)

    one_step_all = sqrt.(sum(one_step_mses)/length(one_step_mses))
    forecast_all = sqrt.(sum(forecast_mses)/length(forecast_mses))

    return one_step_all, forecast_all, one_step, forecast                       
end
