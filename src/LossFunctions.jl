module LossFunctions

mutable struct LossFunction
    parameters
    loss
end

function MSE(;N = 1, weight = 1.0)
    parameters = NamedTuple()
    loss = (u,uhat,parameters) -> weight*sum((u.-uhat).^2)/N
    return LossFunction(parameters,loss)
end


function inv_softmax(p)
    c = (1 - sum(log.(p)) - log(1-sum(p)))/(length(p)+1)
    return log.(p) .+ c
end 

function softmaxMSE(;N = 1, weight = 1.0)
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


end # module