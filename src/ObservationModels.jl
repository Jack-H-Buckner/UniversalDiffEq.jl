
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

