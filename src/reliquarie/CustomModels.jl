
mutable struct CustomModel
    data
    process_model
    observation_model
    priors
    parameters
    loss
end 


function init_loss(data, process_model, observation_model, priors)
    # loss function 
    function loss_function(parameters)

        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            L_obs += observation_model(yt, parameters.uhat[:,t],parameters)
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            L_proc += process_model(u0,u1,times[t-1],dt,parameters)
        end
        
        # regularization
        L_prior = priors(parameters)

        return L_obs + L_proc + L_prior
    end

    # skips the prediction steps for intervals starting at a time in t_skip
    function loss_function(parameters,t_skip)
        # observation loss
        L_obs = 0.0 
        
        for t in 1:(size(data)[2])
            yt = data[:,t]
            L_obs = observation_model(yt, parameters.uhat[:,t],parameters)
        end

        # dynamics loss 
        L_proc = 0
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            if !(times[t-1] in t_skip)
                L_proc += process_model(u0,times[t-1],dt,parameters)
            end
        end
        
        # regularization
        L_priors = priors(parameters)
        
        return L_obs + L_proc + L_priors
    end
    return loss_function
end 


