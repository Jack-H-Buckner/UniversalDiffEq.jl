

mutable struct UQ_UDE
    times
    data
    X
    data_frame
    X_data_frame
    parameters
    initial_parameters
    loss_function
    permuted_loss_function
    permuted_loss_function_it
    loss_function_process
    loss_function_process_it
    process_model
    observation_model
    σ
    regularization
    constructor
    time_column_name
    variable_column_name
    value_column_name
end


function init_losses(data,times,observation_model,process_model,regularization,ρ, k, β)
    
    # loss function 
    function loss_function(observation_parameters, process_parameters,σ, uhat)

        # observation loss_function_process
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(uhat[:,t],observation_parameters)
            L_obs += sum((yt .- yhat).^2 ./ σ.^2 )
            L_obs += 2*sum(log.(abs.(σ .^2)))
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = uhat[:,t-1]
            u1 = uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,times[t-1],dt,process_parameters) 
            L_proc += sum( (u1.-u1hat).^2 ./ (ρ .* σ) .^2 )
            L_proc += 2*sum(log.(abs.((ρ .* σ) .^2)))
        end
        
        # regularization
        L_reg = regularization.loss(process_parameters,0)
        L_reg += sum(-(k-1)*log.(σ.^2) .+ β*σ.^2)

        return L_obs + L_proc + L_reg
    end

    function permuted_loss_function(process_parameters,uncertainty_parameters, uhat,B,ϵ)

        # observation loss
        L_obs = 0.0 
        data_ = data .+ ϵ*B
        for t in 1:(size(data_)[2])
            yt = data_[:,t]
            yhat = observation_model.link(uhat[:,t],process_parameters.observation_model)
            L_obs += sum((yt .- yhat).^2 ./ σ.^2 )
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data_)[2])
            u0 = uhat[:,t-1]
            u1 = uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,times[t-1],dt,process_parameters.process_model) 
            L_proc += sum( (u1.-u1hat).^2 ./ uncertainty_parameters.τ .^2 )
            L_proc += 2*sum(log.(abs.(uncertainty_parameters.τ)))
        end
        
        # regularization
        L_reg = regularization.loss(process_parameters.process_model, process_parameters.process_regularization)
        L_reg += sum(-(k-1)*log.(uncertainty_parameters.τ.^2) .+ β*uncertainty_parameters.τ.^2)

        return L_obs + L_proc + L_reg
    end

    function permuted_loss_function_it(i,t,parameters,B,ϵ)

        # observation loss
        data_ = data .+ ϵ*B
        yt = data_[:,t]
        yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
        L_obs = ((yt .- yhat).^2 ./ σ.^2 )[i]

        return L_obs
    end

    function loss_function_process(parameters, uhat, B, ϵ)

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = uhat[:,t-1]
            u1 = uhat[:,t] .+ ϵ*B
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model) 
            L_proc += sum( (u1.-u1hat).^2 ./ (parameters.τ).^2 )
            L_proc += 2*sum(log.(abs.(parameters.τ)))
            

        end
        
        # regularization
        L_reg = regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += sum(-(k-1)*log.(parameters.τ.^2) .+ β*parameters.τ.^2)

        return L_proc + L_reg
    end

    function loss_function_process_it(i,t,parameters,B,ϵ)

        data_ = data .+ ϵ*B
        yt = data_[:,t]

        u0 = parameters.uhat[:,t-1]
        dt = times[t]-times[t-1]
        uhat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model) 
        yhat = observation_model.link(uhat, parameters.observation_model)
        L = ((yt.-yhat).^2 ./ (parameters.σ).^2 )[i]

        return L
    end

    return loss_function, permuted_loss_function, permuted_loss_function_it, loss_function_process, loss_function_process_it
end 


function UQ_CustomDerivatives(data::DataFrame,derivs!::Function,initial_parameters,ρ;time_column_name = "time",proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,k=10,β=20.0,extrap_rho=0.1,l=0.25,reg_type = "L2")
    time_column_name = check_column_names(data, time_column_name = time_column_name)[1]
    # convert data
    N, dims, T, times, data, dataframe = process_data(data,time_column_name )

    # generate submodels
    process_model = ContinuousProcessModel(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho)
    observation_model = Identity()
    regularization = L2(initial_parameters,weight=reg_weight)

    # parameters vector
    parameters = ComponentArray((uhat = zeros(size(data)), process_model = initial_parameters,observation_model = NamedTuple(),σ = zeros(dims) .+ 1.0, process_regularization = 0))

    # loss function
    loss_function, permuted_loss_function, permuted_loss_function_it, loss_function_process, loss_function_process_it = init_losses(data,times,observation_model,process_model,regularization,ρ, k, β)

    # model constructor
    constructor = data -> CustomDerivatives(data,derivs!,initial_parameters,priors;time_column_name = time_column_name ,proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)

    return UQ_UDE(times,data,0,dataframe,0,parameters,parameters, loss_function, permuted_loss_function,
                     permuted_loss_function_it, loss_function_process, loss_function_process_it,
                     process_model,observation_model,σ,regularization,
                     constructor,time_column_name,nothing, nothing)

end

function optimize_all!(UDE, verbose, step_size, maxiter)

    # set optimization problem 
    target = (x,p) -> UDE.loss_function(x.observation_model,x.process_model,x.σ, x.uhat)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
    if verbose
        callback = function (p, l; doplot = false)
            print(round(l,digits = 3), " ")
            return false
        end
    else
        callback = function (p, l; doplot = false)
            return false
        end 
    end

    # run optimizer
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter )
    
    # assign parameters to model 
    UDE.parameters = sol.u
    
    return nothing
end

function optimize_process!(UDE, verbose, step_size, maxiter)

    # set optimization problem 
    target = (x,p) -> UDE.loss_function(x.observation_model,x.process_model,UDE.parameters.σ, x.uhat)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
    if verbose
        callback = function (p, l; doplot = false)
            print(round(l,digits = 3), " ")
            return false
        end
    else
        callback = function (p, l; doplot = false)
            return false
        end 
    end

    # run optimizer
    sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter )
    
    # assign parameters to model 
    UDE.parameters = sol.u
    
    return nothing
end

function optimize_uncertianty!(UDE, verbose)

    # set optimization problem 
    target = (x,p) -> UDE.loss_function(UDE.parameters.observation_model,UDE.parameters.process_model,x.σ, x.uhat)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(target, adtype)
    optprob = Optimization.OptimizationProblem(optf, UDE.parameters)
    
    # print value of loss function at each time step 
    if verbose
        callback = function (p, l; doplot = false)
            print(round(l,digits = 3), " ")
            return false
        end
    else
        callback = function (p, l; doplot = false)
            return false
        end 
    end

    # run optimizer
    sol = Optimization.solve(optprob, Optim.BFGS(; initial_stepnorm = 0.005); callback, allow_f_increases = false)
    # sol = Optimization.solve(optprob, OptimizationOptimisers.Adam(step_size), callback = callback, maxiters = maxiter )
    
    # assign parameters to model 
    UDE.parameters = sol.u
    
    return nothing

end 

function optimize!(model::UQ_UDE; verbose = true, step_size = 0.05, maxiter_gd = 250, iters = 2)
    optimize_all!(model,verbose, step_size,maxiter_gd );println(" ")
    optimize_uncertianty!(model, verbose)
    for i in 1:iters 
        println(i)
        optimize_process!(model, verbose, step_size, maxiter_gd);println(" ")
        optimize_uncertianty!(model, verbose)
    end 
end 