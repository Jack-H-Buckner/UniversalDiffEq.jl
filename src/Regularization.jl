"""
The module defines the regularization classes. 

We declare an abstact type AbstractRegularization of which all the regularization model classes are subtypes. Instances of AbstractRegularization subtypes are expected to have a field named parameters that store paramter values for hierarchical models and a function called loss that takes a component array of model paramters. This component array (the argument of the loss function) is a composite of all the parameters required to define a state space UDE, it has sub fields for the observaiton and process models and regularization. 
"""
module Regularization

mutable struct L2 
    reg_parameters 
    loss::Function
end 

"""
Initalizes and instance of the L2 AbstractRegularization class. The loss function only act on the first layer of the nerual network. This regularization model does not require parameters. 
"""
function L2(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum(parameters.layer_1.weight.^2) + sum(parameters.layer_2.weight.^2))
    
    return L2(reg_parameters,loss)
    
end 

function L2UDE(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum(parameters.NN.layer_1.weight.^2) + sum(parameters.NN.layer_2.weight.^2))
    
    return L2(reg_parameters,loss)
    
end 


function L1(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum(abs.(parameters.layer_1.weight))+ sum(abs.(parameters.layer_2.weight)))
    
    return L2(reg_parameters,loss)
    
end 


function L2_all(;weight=1.0)
    
    reg_parameters = NamedTuple()

    loss = (parameters,reg_parameters) -> weight * (sum(parameters.^2))
    
    return L2(reg_parameters,loss)
    
end 




end