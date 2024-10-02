


function init_loss_ARD(data,times,observation_model,observation_loss,process_model,process_loss,outputs,σ_r,λ)
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
        L_proc = 0; L_reg = 0 
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            u1hat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model) 
            L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
            L_reg += sum(epsilon)
        end
        
        # regularization
        L_reg += ARD_regularization(parameters.process_model,outputs,σ_r,λ)

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
        L_proc = 0; L_reg = 0 
        for t in 2:(size(data)[2])
            u0 = parameters.uhat[:,t-1]
            u1 = parameters.uhat[:,t]
            dt = times[t]-times[t-1]
            if !(times[t-1] in t_skip)
                u1hat, epsilon = process_model.predict(u0,times[t-1],dt,parameters.process_model) 
                L_proc += process_loss.loss(u1,u1hat,dt,parameters.process_loss)
                L_reg += sum(epsilon)
            end
        end
        
        # regularization
        L_reg += ARD_regularization(parameters.process_model,outputs,σ_r,λ)
        
        return L_obs + L_proc + L_reg
    end

    return loss_function
end 


function ARD_regularization(parameters,outputs,σ_r,λ)
    L = 0
    for d in 1:outputs
        L += sum((parameters.NN[Symbol(string("NN_",d))].layer_1.weight).^2 ./(σ_r^2 ))
        L += sum((parameters.NN[Symbol(string("NN_",d))].layer_2.weight).^2 ./(σ_r^2 ))
    end 
    L += λ*sum(abs.(parameters.scale))
    L += sum(log.(abs.(parameters.α)))
end