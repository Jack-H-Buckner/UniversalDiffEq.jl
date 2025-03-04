
"""
    LossFunction

A Julia mutable struct that stores the loss function and parameters.
...
# Elements
- parameters: ComponentArray
- loss: Function 
...
"""
mutable struct LossFunction
    parameters
    loss
end

function ObservationMSE(N, weight)
    parameters = NamedTuple()
    loss = (u,uhat,parameters) -> weight*sum((u.-uhat).^2)/N
    return LossFunction(parameters,loss)
end


function ProcessMSE(N,T, weight)
    parameters = NamedTuple()
    loss = (u,uhat,dt,parameters) -> T/dt*weight*sum((u.-uhat).^2)/N^2
    return LossFunction(parameters,loss)
end


function inv_softmax(p)
    c = (1 - sum(log.(p)) - log(1-sum(p)))/(length(p)+1)
    return log.(p) .+ c
end 

function softmaxMSE(N,weight)
    parameters = NamedTuple()
    loss = (u,uhat,parameters) -> weight*sum((inv_softmax(u).-inv_softmax(uhat)).^2)/N
    return LossFunction(parameters,loss)
end



function MSE_and_errors_sd(dims;N = 1, MSE_weight = 1.0, errors_weight = 1.0)
    parameters = (mean=zeros(dims),)
    function loss(u,uhat,epsilon,parameter)
        errors = errors_weight*sum((epsilon.-parameters.mean).^2)/N
        MSE = MSE_weight*sum((u.-uhat).^2)/N
        return errors + MSE
    end 

    return LossFunction(parameters,loss)
end

function FixedMvNoraml(Σ)

    parameters = NamedTuple()
    invΣ = inv(Σ)
    function loss(u,uhat,parameter)
        devs = uhat .- u
        L = 0.5*devs'*invΣ*devs 
        return L
    end 

    return LossFunction(parameters,loss)
end


function DiagonalNoraml(dims;σ0 = 1.0)

    parameters = (log_σ = repeat([log(σ0)],dims),)

    function loss(u,uhat,dt,parameter)
        L = 0
        for d in 1:dims
            L += parameter.log_σ[d] + 0.5 * ((u[d]-uhat[d])/(dt*exp(parameter.log_σ[d])))^2
        end 
        return L
    end 

    return LossFunction(parameters,loss)
end


function dirichlet(dims;σ0 = 1.0)

    parameters = NamedTuple()

    function loss(u,uhat,dt,parameters)
        -pdf(Distributions.Dirichlet((w/dt).*u),uhat)
    end 

    return LossFunction(parameters,loss)
end
