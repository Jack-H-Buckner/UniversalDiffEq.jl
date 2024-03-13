

function get_final_state(UDE::UDE)
    return UDE.parameters.uhat[:,end]
end 


"""
    plot_state_estiamtes(UDE::UDE)

Plots the value of the state variables estiamted by the UDE model. 
"""
function plot_state_estiamtes(UDE::UDE)
    
    plots = []
    for dim in 1:size(UDE.data)[1]
    
        plt=Plots.scatter(UDE.times,UDE.data[dim,:], label = "observations")
        
        Plots.plot!(UDE.times,UDE.parameters.uhat[dim,:], color = "grey", label= "estimated states",
                    xlabel = "time", ylabel = string("x", dim))
       
        push!(plots, plt)
    end 
            
    return plot(plots...)       
end 



"""
    print_parameter_estimates(UDE::UDE)

prints the value of the known dynamcis paramters. 
"""
function print_parameter_estimates(UDE::UDE)
    println("Estimated parameter values: ")
    i = 0
    for name in keys(UDE.parameters.process_model.known_dynamics)
        i += 1
        println(name, ": ", round(UDE.parameters.process_model.known_dynamics[i], digits = 3))
            
    end 
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
     
    N, dims, T, times, data, dataframe = process_data(test_data)
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





"""
    plot_predictions(UDE::UDE)

Plots the correspondence between the observed state transitons and the predicitons for the model `UDE`. 

"""
function plot_predictions(UDE::UDE)
 
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


"""
    plot_predictions(UDE::UDE, test_data::DataFrame)

Plots the correspondence between the observed state transitons and observed transitions in the test data. 
"""
function plot_predictions(UDE::UDE,test_data::DataFrame)
 
    inits, obs, preds = predictions(UDE,test_data)
    
    plots = []
    for dim in 1:size(obs)[1]
        difs = obs[dim,:].-inits[dim,:]
        xmin = difs[argmin(difs)]
        xmax = difs[argmax(difs)]
        plt = plot([xmin,xmax],[xmin,xmax],color = "grey", linestyle=:dash, label = "45 degree")
        scatter!(difs,preds[dim,:].-inits[dim,:],color = "white", label = "", xlabel = "Observed change", 
                                ylabel = "Predicted change")
        push!(plots, plt)
            
    end
        
    return plot(plots...)
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
    forecast(UDE::UDE, u0::AbstractVector{}, times::AbstractVector{})

predicitons from the trained model `UDE` starting at `u0` saving values at `times`.
"""
function forecast(UDE, u0::AbstractVector{}, times::AbstractVector{})
    
    uhats = UDE.parameters.uhat
    
    umax = mapslices(max_, UDE.parameters.uhat, dims = 2);umax=reshape(umax,length(umax))
    umin = mapslices(min_, UDE.parameters.uhat, dims = 2);umin=reshape(umin,length(umin))
    umean = mapslices(mean_, UDE.parameters.uhat, dims = 2);umean=reshape(umean,length(umean))
    
    
    #estimated_map = (x,dt) -> UDE.process_model.forecast(x,dt,UDE.parameters.process_model,umax,umin,umean)
    estimated_map = (x,t,dt) -> UDE.process_model.forecast(x,t,dt,UDE.parameters.process_model,umax,umin,umean)
    
    
    x = u0
    df = zeros(length(times),length(x)+1)
    df[1,:] = vcat([times[1]],x)
    
    for t in 2:length(times)
        dt = times[t]-times[t-1]
        x = estimated_map(x,times[t-1],dt)
        df[t,:] = vcat([times[t-1]],x)
    end 
    
    return df
end 


"""
    plot_forecast(UDE::UDE, T::Int)

Plots the models forecast up to T time steps into the future from the last observaiton.  

UDE - a UDE model object
T - the nuber of time steps to forecast
"""
function plot_forecast(UDE::UDE, T::Int)
    u0 = UDE.parameters.uhat[:,end]
    dts = UDE.times[2:end] .- UDE.times[1:(end-1)]
    dt = sum(dts)/length(dts)
    times = UDE.times[end]:dt:(UDE.times[end] + T*dt )
    df = forecast(UDE, u0, times)
    plots = []
    for dim in 2:size(df)[2]
        plt = plot(df[:,1],df[:,dim],color = "grey", linestyle=:dash, label = "forecast",
                    xlabel = "Time", ylabel = string("x", dim))
        plot!(UDE.times,UDE.data[dim-1,:],c=1, label = "data",
                    xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end 
    return plot(plots...), plots
end 


"""
    plot_forecast(UDE::UDE, test_data::DataFrame)

Plots the models forecast over the range of the test_data along with the value of the test data.   

UDE - a UDE model object
T - the nuber of time steps to forecast
"""
function plot_forecast(UDE::UDE, test_data::DataFrame)
    u0 = UDE.parameters.uhat[:,end]
    N, dims, T, times, data, dataframe = process_data(test_data)
    df = forecast(UDE, u0, times)
    plots = []
    for dim in 2:size(df)[2]
        plt = plot(df[:,1],df[:,dim],color = "grey", linestyle=:dash, label = "forecast", xlabel = "Time", ylabel = string("x", dim))
        scatter!(UDE.times,UDE.data[dim-1,:],c=1, label = "data", xlabel = "Time", ylabel = string("x", dim))
        scatter!(times,data[dim-1,:],c=2, label = "data", xlabel = "Time", ylabel = string("x", dim))
        push!(plots, plt)
    end 
    return plot(plots...), plots
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
    test_data = data[(N_train):end,:]
    
    # build model 
    model = model.constructor(train_data)
    gradient_decent!(model, step_size = step_size, maxiter = maxiter) 
    
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
    test_data = data[(N_train):end,:]
    
    # build model 
    model = model.constructor(train_data)
    gradient_decent!(model, step_size = step_size, maxiter = maxiter) 
    
    # forecast
    u0 = model.parameters.uhat[:,end]
    times = test_data.t
    predicted_data = forecast(model, u0, times)
    

    return abs.(Matrix(predicted_data[:,2:end]) .- Matrix(test_data[:,2:end]))
end 



function leave_future_out(model; forecast_length = 10,  forecast_number = 10, spacing = 1, step_size = 0.05, maxiter = 500,using_BFGS=false)
    
    # get final time
    data = model.data_frame
    T = length(data.t)
    start1 = T - forecast_length - spacing*(forecast_number-1)
    starts = [start1 + spacing *i for i in 0:(forecast_number-1)]
    training_data = [data[1:t0,:] for t0 in starts]
    testing_data = [data[t0:(t0+forecast_length),:] for t0 in starts]
    
    standard_errors = [[] for i in 1:Threads.nthreads()]
    predicted = [[] for i in 1:Threads.nthreads()]
    
    Threads.@threads for i in 1:forecast_number
        
        model_i = model.constructor(training_data[i])
                        
        gradient_decent!(model_i, step_size = step_size, maxiter = maxiter) 
           
        if using_BFGS
            try
                BFGS!(model_i,verbos = false)
            catch
                println("BFGS failed running gradient_decent")
                gradient_decent!(model_i, step_size = 0.25*step_size, maxiter = 2*maxiter)                 
            end   
        end
                    
        # forecast
        u0 = model_i.parameters.uhat[:,end]
        times = testing_data[i].t
        predicted_data = forecast(model_i, u0, times)
        predicted_data= DataFrame(predicted_data,names(testing_data[i]))
            
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
    
    return training_data, testing_data, standard_error, predicted_data
    
end 


function leave_future_out_mse(standard_errors)
    N = length(standard_errors)
    acc = zeros(size(Matrix(standard_errors[1])[:,2:end]))
        
    for i in 1:N
                                    
        acc .+= standard_errors[i][:,2:end] ./ N
                                    
    end 
    
    MSE = DataFrame(hcat(collect(1:size(acc)[1]), acc), names(standard_errors[1]))
    
    return MSE
end 

                                        
function plot_leave_future_out_cv(data,testing_data, standard_errors , predicted_data)
    plts1 = []
    plts2 = []
    for i in 2:(size(data)[2])
        p1=Plots.scatter(data.t,data[:,i],label = "")
        p2 = Plots.plot([0,data.t[end]],[0.0,0.0], color = "black", linestyle = :dash,label = "")
        for j in 1:length(testing_data)
            Plots.plot!(p1,predicted_data[j].t,predicted_data[j][:,i], linestyle = :dash, width= 2,label = "") 
            Plots.plot!(p2,standard_errors[j].t,standard_errors[j][:,i], width= 2,label = "")
        end 
        push!(plts1,p1)
        push!(plts2,p2)
    end
    p1 = plot(plts1...)
    p2 = plot(plts2...)
    return p1,p2
end 

"""
    leave_future_out_cv(model; forecast_length = 10,  K = 10, spacing = 1, step_size = 0.05, maxiter = 500)
    
Runs K fold leave future out cross validation and returns the mean squared forecasting error and a plot to visulaize the model fits.

...
# Arguments 
model - the UDE model to test
forecast_length - the number of steps to calcualte the forecast performance (default 10).
K - the number of forecast tests to run (default 10).
spacing - the number of data points to skip between testing sets (default 1).
step_size - step size parameter for the gradient decent algorithm (default 0.05).
maxiter - number of iterations for gradent decent (default 500).. 
...
"""
function leave_future_out_cv(model; forecast_length = 10,  K = 10, spacing = 1, step_size = 0.05, maxiter = 500)
    training_data, testing_data, standard_errors, predicted_data = leave_future_out(model;forecast_length=forecast_length,forecast_number=K,spacing=spacing,step_size=step_size,maxiter=maxiter)
    MSE = leave_future_out_mse(standard_errors)
    plt = plot_leave_future_out_cv(model.data_frame,testing_data, standard_errors , predicted_data)
    return MSE, plt                             
end


function forecast_simulation_tests(N,simulator,model;train_fraction=0.9,step_size = 0.05, maxiter = 500)
    
    # get test data size and set accumulator
    sizeSE = forecast_simulation_SE(simulator,model,1;
                    train_fraction=train_fraction,step_size = step_size, maxiter = 1)
    
    MSE_acc = [zeros(size(sizeSE)) for i in 1:Threads.nthreads()]
    
    # run simulation tests with multithreading
    Threads.@threads for seed in 1:N
            
        MSE_acc[Threads.threadid()] .+= forecast_simulation_SE(simulator,model,seed;train_fraction=train_fraction,step_size=step_size,maxiter=maxiter)./N
            
    end 
        
    MSE = MSE_acc[1]
    for i in 2:Threads.nthreads()
        MSE .+= MSE_acc[i]
    end 
                
    T = size(MSE)[1]
    MSE = DataFrame(MSE ,:auto)
    MSE.t = 1:T
                
    return MSE
  
end 