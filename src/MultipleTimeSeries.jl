include("ObservationModels.jl")
include("ProcessModels.jl")
include("LossFunctions.jl")
include("Regularization.jl")
include("helpers.jl")

"""
UDE

A data structure that stores the informaiton required to define a universal differntial equation model

times - the points in time when observaions are made
data - the obervaitons with each colum corresponding to a time point and each row a dimension of the observaitons vector
parameters - the wights and biases of neural netowrks and any other parameters estimated by the model
loss_function - a function that defines model performance that is minimized by the optimization routine
process_model - a function that predicts the evolution of the state variables between time points
process_loss - a funtion that quantifies the acuracy of the process model
observation_model - a function that descirbes the relationship between the observations and state variables
observations_loss - a function that describes the accuracy of the observation model
process_regularization - a function that penealizes the process model for complexty of adds prior information
observation_regularization - a function that penealizes the observation model for complexty of adds prior information
"""
mutable struct MultiUDE
    times
    data
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


function init_multi_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    
    
    function loss(parameters)
        
        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end

        L_proc = 0
        i = 0
        for t0 in starts .-1
            i+=1
            for t in 2:lengths[i]
                dt = times[i][t] - times[i][t-1]
                u0 = parameters.uhat[:,t0+t-1]
                u1 = parameters.uhat[:,t0+t]
                u1hat, epsilon = process_model.predict(u0,dt,parameters.process_model) 
                L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)  
            end
        end 
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += process_regularization.loss(parameters.observation_model,parameters.process_regularization)
        
        return L_reg + L_obs + L_proc
    end   
    
    return loss
        
end
    
    
function MultiCustomDerivatives(data,derivs!,initial_parameters;proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6,extrap_rho = 0.1,l = 0.25)
    
    # convert data
    N, T, dims, data, dataframe = process_multi_data(data)
    
    # generate submodels 
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    process_loss = ProcessMSE(N,T,proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(weight=reg_weight)
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_multi_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,
                                                process_regularization,observation_regularization)
    # model constructor
    constructor = (data) -> MultiCustomDerivatives(data,derivs!,initial_parameters;
                    proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,extrap_rho=extrap_rho,l=l)
    
    return MultiUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end




function MultiNeuralNet(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
    # convert data
    dataframe = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,3:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.NeuralNetwork2(size(data)[1];hidden = hidden_units, seed = NN_seed)
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    # loss function 
    loss_function = init_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    constructor = data -> NeuralNet(data;hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    
    return MultiUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                                observation_loss,process_regularization,observation_regularization,constructor)
    
end 


function MultiNODE(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
    # convert data
    dataframe = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,3:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.NODE_process(size(data)[1];hidden = hidden_units, seed = NN_seed)
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.Identity()
    observation_loss = LossFunctions.MSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    
    constructor = (data) -> MultiNODE(data;
                            hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return MultiUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end 



function MultiNODESimplex(data;hidden_units=10,NN_seed = 1,proc_weight=1.0,obs_weight=1.0,reg_weight = 10^-6)
    
    # convert data
    dataframe = data
    times = data.t # time in colum 1
    data = transpose(Matrix(data[:,3:size(data)[2]]))
    
    # initialize estiamted states
    uhat = zeros(size(data))
    
    process_model = ProcessModels.NODE_process(size(data)[1];hidden = hidden_units, seed = NN_seed)
    process_loss = LossFunctions.MSE(N = size(data)[2]-1,weight = proc_weight)
    observation_model = ObservationModels.softmax()
    observation_loss = LossFunctions.softmaxMSE(N = size(data)[2],weight = obs_weight)
    process_regularization = Regularization.L2(weight=reg_weight)
    observation_regularization = NamedTuple()
    
    
    # parameters
    parameters = (uhat = uhat, 
                    process_model = process_model.parameters,
                    process_loss = process_loss.parameters,
                    observation_model = observation_model.parameters,
                    observation_loss = observation_loss.parameters,
                    process_regularization = process_regularization.reg_parameters, 
                    observation_regularization = NamedTuple())
    
    parameters = ComponentArray(parameters)
    
    loss_function = init_loss_function(dataframe,data,process_model,process_loss,observation_model,observation_loss,process_regularization,observation_regularization)
    
    
    constructor = (data) -> MultiNODESimplex(data;
                            hidden_units=hidden_units,NN_seed=NN_seed,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight)
    
    return MultiUDE(times,data,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)
    
end 



function predictions(UDE::MultiUDE)
     
    dims = size(UDE.data_frame)[2]-2
    inds = collect(1:nrow(UDE.data_frame))  
    starts = [inds[UDE.data_frame.series .== i][1] for i in unique(UDE.data_frame.series)]
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    
    inits = [zeros(dims,l-1) for l in lengths]
    obs = [zeros(dims,l-1) for l in lengths]
    preds = [zeros(dims,l-1) for l in lengths]
    
    
    i = 0
    for t0 in starts .-1
        i+=1
        for t in 1:(lengths[i]-1)
            u0 = UDE.parameters.uhat[:,t0+t]
            u1 = UDE.parameters.uhat[:,t0+t+1]
            dt = UDE.times[t0+t+1] - UDE.times[t0+t]
            u1hat, epsilon = UDE.process_model.predict(u0,dt,UDE.parameters.process_model) 
            inits[i][:,t]  = u0;obs[i][:,t]  = u1;preds[i][:,t]  = u1hat
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
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change Delta hatu_t", 
                                ylabel = "Predicted change hatut - hatu_t")
        push!(plots, plt)
            
    end
        
    return plot(plots...)
end


function plot_state_estiamtes(UDE::MultiUDE)
    
    dims = size(UDE.data_frame)[2]-2
    inds = collect(1:nrow(UDE.data_frame))  
    starts = [inds[UDE.data_frame.series .== i][1] for i in unique(UDE.data_frame.series)]
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    
    plots = []
    for dim in 1:size(UDE.data)[1]
        plt = plot();i=0
         for t0 in starts
            i+=1
            lower_ind = t0
            upper_ind = t0 + lengths[i]-1
            Plots.scatter!(UDE.times[lower_ind:upper_ind],UDE.data[dim,lower_ind:upper_ind], label = "", c = i, width = 0.5)
            xhat = mapslices( x -> UDE.observation_model.link(x,UDE.parameters.observation_model), UDE.parameters.uhat[:,lower_ind:upper_ind], dims = 1)
            Plots.plot!(UDE.times[lower_ind:upper_ind],xhat[dim,:], color = "grey", label= "",
                        xlabel = "time", ylabel = string("x", dim), c = i, width =2, linestyle = :dash)
     
        end 
       
        push!(plots, plt)
        
    end 
            
    return plot(plots...)       
end 



function forecast(UDE::MultiUDE, u0::AbstractVector{}, times::AbstractVector{})
    estimated_map = (x,dt) -> UDE.process_model.predict(x,dt,UDE.parameters.process_model)[1]
    x = u0
    df = zeros(length(times),length(x)+1)
    df[1,:] = vcat([times[1]],UDE.observation_model.link(x,UDE.parameters.observation_model))
    for t in 2:length(times)
        dt = times[t]-times[t-1]
        x = estimated_map(x,dt)
        df[t,:] = vcat([times[t]],UDE.observation_model.link(x,UDE.parameters.observation_model))
    end 
    
    return df
end 




function plot_forecast(UDE::MultiUDE, T)
    
    dims = size(UDE.data_frame)[2]-2
    inds = collect(1:nrow(UDE.data_frame))  
    starts = [inds[UDE.data_frame.series .== i][1] for i in unique(UDE.data_frame.series)]
    lengths = [sum(UDE.data_frame.series .== i) for i in unique(UDE.data_frame.series)]
    
    
    plots = []
    for dim in 1:size(UDE.data)[1]
        plt = plot();i=0
         for t0 in starts
            i+=1
            lower_ind = t0
            upper_ind = t0 + lengths[i]-1
            u0 = UDE.parameters.uhat[:,upper_ind]
            
            dts = UDE.times[(lower_ind+1):upper_ind] .- UDE.times[lower_ind:(upper_ind-1)]
            dt = sum(dts)/length(dts)
            times = UDE.times[upper_ind]:dt:(UDE.times[upper_ind] + T*dt )
            
            df = forecast(UDE, u0, times)
            
            plt = plot!(df[:,1],df[:,dim+1],color = "grey", linestyle=:dash, label = "forecast",
                    xlabel = "Time", ylabel = string("x", dim))
                plot!(UDE.times[lower_ind:upper_ind],UDE.data[dim,lower_ind:upper_ind],c=1, label = "data",
                        xlabel = "Time", ylabel = string("x", dim))
     
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