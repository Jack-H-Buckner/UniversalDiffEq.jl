"""
The module defines the observation model classes. 

We declare an abstact type AbstractObservationModel of which all the observation model classes are subtypes. Instances of AbstractObservationModel subtypes are expected to have a field named parameters that stores the model parameter, link, that transforms the state variables u_t and loss a function that takes the observaiton x_t, the transformed state and the model paramters and calcualtes the performanceof the prediction. The parameters object will be assumed to be a Component Array from `ComponentArrays.jl` with field sub fields `parameters.link` and `parameters.loss`

Each AbstractObservationModel subtype can then have additional fields that store data specific to that model.  
"""
module ObservationModels

mutable struct LinkFunction
    parameters
    link
end

function Identity()
        
    # parameters
    parameters = NamedTuple()
    vals = NamedTuple()
    
    # link 
    link = (u,parameters) -> u
    
    return LinkFunction(parameters,link)
end 


function softmax()
        
    # parameters
    parameters = NamedTuple()
    vals = NamedTuple()
    
    # link 
    function link(u, parameters)
        xe = 1-sum(u)
        sm = sum(exp.(u)) + exp(xe) 
        return exp.(u)./sm
    end 

    
    return LinkFunction(parameters,link)
end 


end