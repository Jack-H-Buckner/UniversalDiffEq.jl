
"""
    Regularization 

A Julia mutable struct that stores the loss function and parameters.
...
# Elements
- reg_parameters: ComponentArray
- loss: Function 
...
"""
mutable struct Regularization  
    reg_parameters 
    loss::Function
end 

function L2(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum(parameters.layer_1.weight.^2) + sum(parameters.layer_2.weight.^2))
    
    return Regularization(reg_parameters,loss)
    
end 

function find_NN_weights(tree, parameters)
    inds = []
    try 
       inds =  keys(parameters)
    catch 
        return nothing
    end
   
    if typeof(inds[1]) != Symbol
        return nothing
    end 

    for ind in inds 
        if ind == :weight
            push!(tree,parameters[ind])
        else
            find_NN_weights(tree, parameters[ind])
        end    
    end 
    return tree

end

function find_NN_weights(parameters)
    tree = []
    find_NN_weights(tree, parameters)
end 

function L2(parameters;weight=1.0)
    
    reg_parameters = NamedTuple()

    function  loss(parameters,reg_parameters)  
        weights = find_NN_weights(parameters)
        L=0
        for w in weights
            L+=weights * sum(w.^2)
        end 
        return L
    end
    
    return Regularization(reg_parameters,loss)
    
end 


function L2UDE(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum(parameters.NN.layer_1.weight.^2) + sum(parameters.NN.layer_2.weight.^2))
    
    return Regularization(reg_parameters,loss)
    
end 


function L1(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum(abs.(parameters.layer_1.weight))+ sum(abs.(parameters.layer_2.weight)))
    
    return Regularization(reg_parameters,loss)
    
end 


function L2_all(;weight=1.0)
    
    reg_parameters = NamedTuple()

    loss = (parameters,reg_parameters) -> weight * (sum(parameters.^2))
    
    return Regularization(reg_parameters,loss)
    
end 


function no_reg()
    return Regularization(NamedTuple(),(p,r) -> 0.0)
end 


