## functions to generate forecast from fitted UDE models 

using Optim, DataFrames

function get_residuals(UDE)
    inits, obs, preds = UniversalDiffEq.predictions(UDE,UDE.data_frame)
    proc_resids = preds .- obs
    link = x -> UDE.observation_model.link(x,UDE.parameters.observation_model)
    obs_resids = mapslices(link,UDE.parameters.uhat,dims=1) .- UDE.data  
    return proc_resids, obs_resids
end 

function inverse_link_target(x0,u0,link)
    sum((x0 .- link(u0)).^2)
end 

function invers_link(x0,link)
    target = u ->  inverse_link_target(x0,u,link)
    sol = optimize(target, [0.0, 0.0])
    sol.minimizer
end 

function sample_initial_state(x0,link,obs_resids)
    u0 = invers_link(x0,link)
    u0 .+ obs_resids[:,rand(1:size(obs_resids)[2])]
end


function max_(x)
    x[argmax(x)]
end

function min_(x)
    x[argmin(x)]
end

function mean_(x)
    sum(x)/length(x)
end


"""
    forecast_with_errors(UDE,x0,t0,T,N)

Makes N forcast length dt*T using the model UDE where dt is the average interval between observations in the training data. 

The forecast are made by making a deterministic forecasts of length dt. A noise term is added to the prediction by sampling from the process model residuals. The initial condition for each forecast is sampled by sampling a residual from the observaiton model and adding it to the initial dat point `x0`. 
"""
function forecast_with_errors(UDE,x0,t0,T,N)
    uhats = UDE.parameters.uhat
    umax = mapslices(max_, UDE.parameters.uhat, dims = 2);umax=reshape(umax,length(umax))
    umin = mapslices(min_, UDE.parameters.uhat, dims = 2);umin=reshape(umin,length(umin))
    umean = mapslices(mean_, UDE.parameters.uhat, dims = 2);umean=reshape(umean,length(umean))

    dt =sum(UDE.times[2:end] .- UDE.times[1:(end-1)])/length(UDE.times[1:(end-1)])
    estimated_map = (x,t) -> UDE.process_model.forecast(x,t,dt,UDE.parameters.process_model,umax,umin,umean)
    
    proc_resids, obs_resids = get_residuals(UDE)
    link = x -> UDE.observation_model.link(x,UDE.parameters.observation_model)

    # accumulator 
    data = DataFrame(t = zeros(N*T).+0.01,sample = zeros(N*T).+0.01)
    for d in 1:length(x0)
        data[:,string("X", d)] .= 0.0
    end 
    j =0
    for i in 1:N
        u = sample_initial_state(x0,link,obs_resids)
        for t in t0:dt:(dt*(T-1))
            j += 1
            data[j,"t"] = t; data[j,"sample"]=i; data[j,3:end] .= u
            u = estimated_map(u,t)
            u .+= proc_resids[:,rand(1:size(proc_resids)[2])]
        end
    end 
    return data
end 

