
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



function L2(L,reg_weight,parameters)

    inds =  keys(parameters)
   
    if typeof(inds[1]) != Symbol
        return 0
    end 
    val = []
    for ind in inds 
        if ind == :weight
            return  reg_weight * sum(parameters[ind].^2)
        else
            L+=L2(L,reg_weight,parameters[ind])
        end    
    end 
    return L

end



function L1(L,reg_weight,parameters)

    inds =  keys(parameters)
   
    if typeof(inds[1]) != Symbol
        return 0
    end 
    val = []
    for ind in inds 
        if ind == :weight
            return  reg_weight * sum(abs(parameters[ind]))
        else
            L+=L1(L,reg_weight,parameters[ind])
        end    
    end 
    return L

end


function L2(parameters;weight=1.0)
    
    reg_parameters = NamedTuple()

    function  loss(parameters,reg_parameters)  
        return L2(0,weight,parameters)
    end
    
    return Regularization(reg_parameters,loss)
    
end 


function L1(parameters;weight=1.0)
    
    reg_parameters = NamedTuple()

    function  loss(parameters,reg_parameters)  
        return L1(0,weight,parameters)
    end
    
    return Regularization(reg_parameters,loss)
    
end 


function L1(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum(abs.(parameters.NN.layer_1.weight))+ sum(abs.(parameters.NN.layer_2.weight)))
    
    return Regularization(reg_parameters,loss)
    
end 


function L2(;weight=1.0)
    
    reg_parameters = NamedTuple()
    
    loss = (parameters,reg_parameters) -> weight * (sum((parameters.NN.layer_1.weight).^2)+ sum((parameters.NN.layer_2.weight).^2))
    
    return Regularization(reg_parameters,loss)
    
end 


function no_reg()
    return Regularization(NamedTuple(),(p,r) -> 0.0)
end 


