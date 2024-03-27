# old function names 

function gradient_decent!(UDE; step_size = 0.05, maxiter = 500, verbos = false)
    print("Depricated due to spelling error, please use key word `verbose` ")
    gradient_decent!(UDE; step_size = 0.05, maxiter = 500, verbose = false)
end 

function BFGS!(UDE; verbos = false, initial_step_norm = 0.01)
    print("Depricated due to spelling error, please use key word `verbose` ")
    BFGS!(UDE; verbose = false, initial_step_norm = 0.01)
end 