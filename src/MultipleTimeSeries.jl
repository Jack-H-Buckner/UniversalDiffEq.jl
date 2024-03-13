include("MultiProcessModel.jl")

mutable struct MultiUDE
    times
    data
    X
    data_frame
    parameters
    loss_function
    process_model
    process_loss 
    observation_model
    observation_loss 
    process_regularization
    observation_regularization
    constructor
end

function init_single_loss(process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    function loss(parameters,times,data,series,starts,lengths)
        
        # observation loss
        L_obs = 0.0 
        uhat = parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]
        time = times[starts[series]:(starts[series]+lengths[series]-1)]
        dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
        for t in 1:(size(dat)[2])
            yt = dat[:,t]
            yhat = observation_model.link(uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end
    
        # dynamics loss 
        L_proc = 0
        for t in 2:(size(dat)[2])
            u0 = uhat[:,t-1]
            u1 = uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,series,times[t-1],dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model.NN,parameters.process_regularization)

        return L_obs + L_proc + L_reg
    end
    return loss
end

function init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    single_loss = init_single_loss(process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    function loss(parameters)
        L = 0
        for i in eachindex(starts)
            L+= single_loss(parameters,times,data,i,starts,lengths)
        end
        return L
    end   
    
    return loss
        
end
    



function MultiNODE(data;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)
    
    # convert data
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)

    process_model =  MultiNODE_process(dims,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(T,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    # parameters
    parameters = (uhat = zeros(size(data)), 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    constructor = (data) -> MultiNODE(data; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)
    
    return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)
    
end 


function MultiNODE(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)
    covariates = interpolate_covariates_multi(X)

    process_model = MultiNODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(T,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()
    
    # parameters
    parameters = (uhat = zeros(size(data)), 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = observation_regularization.reg_parameters)
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    constructor = (data) -> MultiNODE(data,X; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)
    
    return MultiUDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)
    
end 


function MultiNODESimplex(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)
    

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)
    covariates = interpolate_covariates_multi(X)

    process_model = MultiNODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = softmax()
    observation_loss = softmaxMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()

    parameters = (uhat = zeros(size(data)), 
            process_model = process_model.parameters,
            process_loss = process_loss.parameters,
            observation_model = observation_model.parameters,
            observation_loss = observation_loss.parameters,
            process_regularization = process_regularization.reg_parameters, 
            observation_regularization = observation_regularization.reg_parameters)

    parameters = ComponentArray(parameters)

    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = (data) -> MultiNODESimplex(data,X; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)

    return MultiUDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)

end 


function MultiNODESimplex(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    

    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(data)

    process_model = MultiNODE_process(dims,hidden_units,seed,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = softmax()
    observation_loss = softmaxMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(weight=reg_weight)
    elseif reg_type != "L2"
        print("Warning: Invalid choice of regularization: using L2 regularization")
    end 
    observation_regularization = no_reg()

    parameters = (uhat = zeros(size(data)), 
            process_model = process_model.parameters,
            process_loss = process_loss.parameters,
            observation_model = observation_model.parameters,
            observation_loss = observation_loss.parameters,
            process_regularization = process_regularization.reg_parameters, 
            observation_regularization = observation_regularization.reg_parameters)

    parameters = ComponentArray(parameters)

    loss_function = init_multi_loss_function(data,times,starts,lengths,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)

    constructor = (data) -> MultiNODE(data; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type, l=l,extrap_rho=extrap_rho)

    return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)

end 

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
            inits[series][:,t]  = u0;obs[series][:,t]  = u1;preds[series][:,t]  = u1hat
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
        for t in 1:(lengths[series]-1)
            u0 = dat[:,t];u1 = dat[:,t+1];dt = time[t+1] - time[t]
            u1hat, epsilon = UDE.process_model.predict(u0,series_ls[series],time[t],dt,UDE.parameters.process_model) 
            u0 = UDE.observation_model.link(u0,UDE.parameters.observation_model)
            u1 = UDE.observation_model.link(u1,UDE.parameters.observation_model)
            u1hat = UDE.observation_model.link(u1hat,UDE.parameters.observation_model)
            inits[series][:,t]  = u0;obs[series][:,t]  = u1;preds[series][:,t]  = u1hat
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


function plot_state_estiamtes(UDE::MultiUDE)
    

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
    
    N, T, dims, data, times,  dataframe, series, inds, starts, lengths = process_multi_data(UDE.data_frame)
    series_ls = unique(test_data.series)
    plots = []
    for d in 1:dims
        plt = plot()
         for series in eachindex(series_ls)
            series = series_ls[series]
            time = times[starts[series]:(starts[series]+lengths[series]-1)]
            dat = data[:,starts[series]:(starts[series]+lengths[series]-1)]
            uhat = UDE.parameters.uhat[:,starts[series]:(starts[series]+lengths[series]-1)]

            df = forecast(UDE, uhat[:,end], test_data.t[test_data.series .== series], series)
            plt = plot!(df[:,1],df[:,d+1],color = "grey", linestyle=:dash, label = "forecast",xlabel = "Time", ylabel = string("x", dim))
            scatter!(time,dat[d,:],c=1, label = "training data",ylabel = string("x", dim))
            scatter!(test_data.t,test_data[:,d+2],c=2, label = "test data")
     
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


# function MultiNODE(data,X;hidden_units=10,seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,reg_type="L2",l=0.5,extrap_rho=0.0)
    
#     # convert data
#     N, T, dims, data, dataframe, series, inds, starts, lengths, times = process_multi_data(data)
#     covariates = interpolate_covariates(X)

#     # initialize estiamted states
#     uhat = zeros(size(data))
    
#     process_model = NODE_process(dims,hidden_units,covariates,seed,l,extrap_rho)
#     process_loss = ProcessMSE(N,T,proc_weight)
#     observation_model = Identity()
#     observation_loss = ObservationMSE(N,obs_weight)
#     process_regularization = L2(weight=reg_weight)
#     if reg_type == "L1"
#         process_regularization = L1(weight=reg_weight)
#     elseif reg_type != "L2"
#         print("Warning: Invalid choice of regularization: using L2 regularization")
#     end 
#     observation_regularization = no_reg()
    
    
#     # parameters
#     parameters = (uhat = uhat, 
#                     process_model = process_model.parameters,
#                     process_loss = process_loss.parameters,
#                     observation_model = observation_model.parameters,
#                     observation_loss = observation_loss.parameters,
#                     process_regularization = process_regularization.reg_parameters, 
#                     observation_regularization = observation_regularization.reg_parameters)
    
#     parameters = ComponentArray(parameters)
    
#     loss_function = init_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
#     constructor = (data) -> MultiNODE(data; hidden_units=hidden_units,seed=seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,l=l,extrap_rho=extrap_rho)
    
#     return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization,constructor)
    
# end 


# function MultiNODESimplex(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
#     # convert data
#     dataframe = data
#     times = data.t # time in colum 1
#     data = transpose(Matrix(data[:,3:size(data)[2]]))
    
#     # initialize estiamted states
#     uhat = zeros(size(data))
    
#     process_model = ProcessModels.NODE_process(size(data)[1];hidden = hidden_units, seed = NN_seed)
#     process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
#     observation_model = ObservationModels.softmax()
#     observation_loss = LossFunctions.softmaxMSE(N = size(data)[2],weight = obs_weight)
#     process_regularization = Regularization.L2(weight=reg_weight)
#     observation_regularization = NamedTuple()
    
    
#     # parameters
#     parameters = (uhat = uhat, 
#                     process_model = process_model.parameters,
#                     process_loss = process_loss.parameters,
#                     observation_model = observation_model.parameters,
#                     observation_loss = observation_loss.parameters,
#                     process_regularization = process_regularization.reg_parameters, 
#                     observation_regularization = NamedTuple())
    
#     parameters = ComponentArray(parameters)
    
#     loss_function = init_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    
#     constructor = (data) -> MultiNODESimplex(data;
#                             hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
#     return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
#                 observation_loss,process_regularization,observation_regularization,constructor)
    
# end 

# function MultiCustomDerivatives(data,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
#     # convert data
#     N, T, dims, data, dataframe = process_multi_data(data)

#     # generate submodels 
#     process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
#     process_loss = ProcessMSE(N,T, proc_weight)
#     observation_model = Identity()
#     observation_loss = ObservationMSE(N,obs_weight)
#     process_regularization = L2(weight=reg_weight)
#     observation_regularization = no_reg()
    
#     # paramters vector
#     parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

#     # loss function 
#     loss_function = init_multi_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,
#                                                 process_regularization,observation_regularization)
#     # model constructor
#     constructor = (data) -> MultiCustomDerivatives(data,derivs!,initial_parameters;
#                     proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
#     return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
#                 observation_loss,process_regularization,observation_regularization,constructor)

# end


# function MultiCustomDerivatives(data,X,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
#     # convert data
#     N, T, dims, data, dataframe = process_multi_data(data)
#     covariates = interpolate_covariates(X)

#     # generate submodels 
#     process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),covariates,dims,l,extrap_rho)
#     process_loss = ProcessMSE(N,T, proc_weight)
#     observation_model = Identity()
#     observation_loss = ObservationMSE(N,obs_weight)
#     process_regularization = L2(weight=reg_weight)
#     observation_regularization = no_reg()
    
#     # paramters vector
#     parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

#     # loss function 
#     loss_function = init_multi_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,
#                                                 process_regularization,observation_regularization)
#     # model constructor
#     constructor = (data) -> MultiCustomDerivatives(data,X,derivs!,initial_parameters;
#                     proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
#     return MultiUDE(times,data,X,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
#                 observation_loss,process_regularization,observation_regularization,constructor)

# end



# function MultiCustomDifference(data,step,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
#     # convert data
#     N, T, dims, data, dataframe = process_multi_data(data)
    
#     # generate submodels 
#     process_model = DiscreteProcessModel(step,ComponentArray(initial_parameters),dims,l,extrap_rho)
#     process_loss = ProcessMSE(N,T,proc_weight)
#     observation_model = Identity()
#     observation_loss = ObservationMSE(N,obs_weight)
#     process_regularization = L2(weight=reg_weight)
#     observation_regularization = no_reg()
    
#     # paramters vector
#     parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

#     # loss function 
#     loss_function = init_multi_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,
#                                                 process_regularization,observation_regularization)
#     # model constructor
#     constructor = (data) -> MultiCustomDerivatives(data,step,initial_parameters;
#                     proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
#     return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
#                 observation_loss,process_regularization,observation_regularization,constructor)

# end




# function MultiNeuralNet(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
#     # convert data
#     dataframe = data
#     times = data.t # time in colum 1
#     data = transpose(Matrix(data[:,3:size(data)[2]]))
    
#     # initialize estiamted states
#     uhat = zeros(size(data))
    
#     process_model = ProcessModels.NeuralNetwork2(size(data)[1];hidden = hidden_units, seed = NN_seed)
#     process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
#     observation_model = ObservationModels.Identity()
#     observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
#     process_regularization = Regularization.L2(weight=reg_weight)
#     observation_regularization = NamedTuple()
    
#     # parameters
#     parameters = (uhat = uhat, 
#                     process_model = process_model.parameters,
#                     process_loss = process_loss.parameters,
#                     observation_model = observation_model.parameters,
#                     observation_loss = observation_loss.parameters,
#                     process_regularization = process_regularization.reg_parameters, 
#                     observation_regularization = NamedTuple())
    
#     parameters = ComponentArray(parameters)
    
#     # loss function 
#     loss_function = init_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
#     constructor = data -> NeuralNet(data;hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    
#     return MultiUDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
#                                 observation_loss,process_regularization,observation_regularization,constructor)
    
# end 
