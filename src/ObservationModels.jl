
mutable struct LinkFunction
    parameters
    link
    inv_link
end

function Identity()
        
    # parameters
    parameters = NamedTuple()
    vals = NamedTuple()
    
    # link 
    link = (u,parameters) -> u
    inv_link = (u,parameters) -> u
    return LinkFunction(parameters,link,inv_link)
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

    function inv_link(u, parameters)
        k = length(u)+1
        x = log.(u)
        C = (1-sum(log.(u))-log(1-sum(u)))/k
        return x .+ C
    end
    
    return LinkFunction(parameters,link,inv_link)
end 

