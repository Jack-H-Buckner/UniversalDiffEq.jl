
function distance_matrices(inducing_points)
    N = size(inducing_points)[1]; d = size(inducing_points)[2]
    distances = zeros(N,N,d)
    for i in 1:d
        distances[:,:,i] .= (inducing_points[:,i]' .- inducing_points[:,i])
    end
    return d, N, distances, Matrix(I,N,N)
end 

function ARD_covariance(p, distances,DiagMat)
    nugget = 10^-6
    Sigma = p.alpha^2 * exp.(-1 * sum(p.l.^2 .* distances.^2 , dims=3)[:,:,1]) .+ nugget .* DiagMat
end


function cov_function(x1,x2,p) 
    p.alpha^2 * exp(-sum(p.l[1,1,:].^2 .* (x1.- x2).^2))
end 


struct GaussianProcess
    N
    d
    inducing_points
    distances
    diagonal
    psi
    mu 
    mu_values
end 

function init_parameters(GP)
    N = GP.N; d = GP.d
    l = zeros(1,1,d) .+ 1.0
    inducing_values = zeros(N)
    alpha = 0.1
    return ComponentArray(inducing_values = inducing_values, l = l, alpha = alpha)
end

function GaussianProcess(inducing_points)
    d, N, distances, diagonal = distance_matrices(inducing_points)
    psi = 3.14159/2 
    mu = x -> 0
    mu_values = mapslices(mu, inducing_points, dims = 2)[:,1]
    GP = GaussianProcess(N,d,inducing_points,distances,diagonal,psi,mu, mu_values)
    parameters = init_parameters(GP)
    return GP, parameters
end 


function GaussianProcess(inducing_points,psi)
    d, N, distances, diagonal = distance_matrices(inducing_points)
    mu = x -> 0
    mu_values = mapslices(mu, inducing_points, dims = 2)[:,1]
    GP = GaussianProcess(N,d,inducing_points,distances,diagonal,psi, mu, mu_values)
    parameters = init_parameters(GP)
    return GP, parameters
end

function GaussianProcess(inducing_points,psi,mu)
    d, N, distances, diagonal = distance_matrices(inducing_points)
    mu_values = mapslices(mu, inducing_points, dims = 2)[:,1]
    GP = GaussianProcess(N,d,inducing_points,distances,diagonal,psi, mu, mu_values)
    parameters = init_parameters(GP)
    return GP, parameters
end

function likelihood(parameters,GP)
    Sigma = ARD_covariance(parameters, GP.distances, GP.diagonal)
    0.5*( GP.d*log(2*3.14159) + log(det(Sigma)) + (parameters.inducing_values .- GP.mu_values)'*inv(Sigma)*(parameters.inducing_values .- GP.mu_values))
end


function (GP::GaussianProcess)(x,p)
    weights =  broadcast(i->cov_function(GP.inducing_points[i,:],x,p),1:GP.N)
    Sigma = ARD_covariance(p, GP.distances, GP.diagonal)
    Sigma_inv = inv(Sigma)
    y =  GP.mu(x) .+ weights' * Sigma_inv * (p.inducing_values .- GP.mu_values)
    return y
end


struct MvGaussianProcess
    d
    GPs
end

function initMvGaussianProcess(outputs, inducing_points)

    GP_1, parameters_1 = GaussianProcess(inducing_points)
    GPs = (GP_1 = GP_1, ); parameters = (GP_1 = parameters_1, )

    for i in 2:outputs

        GP_i, parameters_i = GaussianProcess(inducing_points)
  
        pars_dict = Dict(string("GP_", i) => parameters_i,)
        pars_tuple = (; (Symbol(k) => v for (k,v) in pars_dict)...)
        parameters  = merge(parameters , pars_tuple)

        NN_dict = Dict(string("GP_", i) => GP_i,)
        NN_tuple = (; (Symbol(k) => v for (k,v) in NN_dict)...)
        GPs  = merge(GPs , NN_tuple)
    end

    return MvGaussianProcess(outputs,GPs), parameters
end 

function (model::MvGaussianProcess)(x,parameters)
    f = d -> model.GPs[Symbol(string("GP_",d))](x,parameters[Symbol(string("GP_",d))])
    broadcast(f, 1:model.d)
end 

function Mvlikelihood(parameters,model)
    L = d -> likelihood(parameters[Symbol(string("GP_",d))],model.GPs[Symbol(string("GP_",d))])
    sum(broadcast(L,1:model.d))
end

function init_loss_GP(data,times,observation_model,observation_loss,process_model,process_loss,α,β)
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
        L_reg += Mvlikelihood(parameters.process_model,process_model.GP)
        L_reg += sum( -(α-1)*2*parameters.process_loss.log_σ.+β.*exp.(2*parameters.process_loss.log_σ) )

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
        L_reg += Mvlikelihood(parameters.process_model,process_model.GP)
        L_reg += sum( -(α-1)*2*parameters.process_loss.log_σ.+β.*exp.(2*parameters.process_loss.log_σ) )

        return L_obs + L_proc + L_reg
    end

    return loss_function
end 
