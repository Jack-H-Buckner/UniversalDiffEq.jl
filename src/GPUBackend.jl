include("GPUProcessModel.jl")

function init_loss_GPU(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization, gpu_device)
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            #=
            print("data: ")
            println(typeof(data))
            print("u0: ")
            println(typeof(u0))
            print("u1: ")
            println(typeof(u1))
            print("dt: ")
            println(typeof(dt))
            =#
            u1hat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model)

            print("u1hat: ")
            println(typeof(u1hat))
            println()
            L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += observation_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end

    # skips the prediction steps for intervals starting at a time in t_skip
    function loss_function(parameters,t_skip)
        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            yhat = observation_model.link(parameters.uhat[:,t],parameters.observation_model)
            L_obs += observation_loss.loss(yt, yhat,parameters.observation_model)
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            if !(times[t-1] in t_skip)
                u1hat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model) 
                L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
            end
        end
        
        # regularization
        L_reg = process_regularization.loss(parameters.process_model,parameters.process_regularization)
        L_reg += observation_regularization.loss(parameters.process_model,parameters.process_regularization)
        
        return L_obs + L_proc + L_reg
    end
    return loss_function
end 


function CustomDerivativesGPU(data,derivs!,initial_parameters, gpu_device;proc_weight=1.0,obs_weight=1.0,reg_weight=10^-6,extrap_rho=0.1,l=0.25,reg_type = "L2")
    
    # convert data
    N, dims, T, times, data, dataframe = process_data(data)
    # generate submodels 
    process_model = ContinuousProcessModel_GPU(derivs!,ComponentArray(initial_parameters),dims,l,extrap_rho, gpu_device)
    process_loss = ProcessMSE(N,T, proc_weight)
    observation_model = Identity()
    observation_loss = ObservationMSE(N,obs_weight)
    process_regularization = L2(initial_parameters,weight=reg_weight)
    if reg_type == "L1"
        process_regularization = L1(initial_parameters,weight=reg_weight)
    elseif reg_type != "L2"
        println("Invalid regularization type - defaulting to L2")
    end
    observation_regularization = no_reg()
    
    # paramters vector
    parameters = init_parameters(data,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization)

    # loss function 
    loss_function = init_loss_GPU(data,times,observation_model,observation_loss,process_model,process_loss,process_regularization,observation_regularization, gpu_device)
    

    # model constructor
    constructor = data -> CustomDerivativesGPU(data,derivs!,initial_parameters, gpu_device;proc_weight=proc_weight,obs_weight=obs_weight,reg_weight=reg_weight,reg_type=reg_type)
    
    return UDE(times,data,0,dataframe,parameters,loss_function,process_model,process_loss,observation_model,
                observation_loss,process_regularization,observation_regularization,constructor)

end