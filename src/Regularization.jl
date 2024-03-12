
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


