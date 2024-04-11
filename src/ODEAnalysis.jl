function meshgrid(n,lower,upper)
    xs = ones(n) .* (1:n)'
    ys = xs'
    xys = permutedims(cat(xs, ys; dims = 3), [3, 1, 2])
    scale = upper .- lower 
    return scale .* reshape(xys, 2, n^2) ./ n .+ lower .- 0.5*scale ./n
end

function vectorfield2d(UDE,X; t = 0, xlabel = "u1", ylabel = "u2", title = "UDE vector field", n = 15, lower = [0.0,0.0], upper = [1.0,1.0], arrowlength=0.1, arrow_color = "grey")

    points = meshgrid(n,lower,upper)

    RHS = get_right_hand_side(UDE)
    field = (x,y) -> RHS([x,y],X,t)

    vectors = similar(points)
    for i in 1:size(points)[2]
        vectors[:, i] .= collect(field(points[:, i]...))
    end

    vectors .*= arrowlength
    quiver(points[1, :],points[2, :],quiver=(vectors[1, :], vectors[2, :]),xlabel = xlabel, ylabel = ylabel, title = title, color = arrow_color)
end

function vectorfield2d(UDE; t = 0, xlabel = "u1", ylabel = "u2", title = "UDE vector field", n = 15, lower = [0.0,0.0], upper = [1.0,1.0], arrowlength=0.1, arrow_color = "grey")
    
    points = meshgrid(n,lower,upper)

    RHS = get_right_hand_side(UDE)
    field = (x,y) -> RHS([x,y],t)

    vectors = similar(points)
    for i in 1:size(points)[2]
        vectors[:, i] .= collect(field(points[:, i]...))
    end

    vectors .*= arrowlength
    quiver(points[1, :],points[2, :],quiver=(vectors[1, :], vectors[2, :]),xlabel = xlabel, ylabel = ylabel, title = title, color = arrow_color )
end

function nullclines2d(UDE; t = 0, upper = [0.0,0.0], lower = [1.0,1.0])
    RHS = get_right_hand_side(UDE)
    
    function nullclineU21(u2;t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU12(u1;t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 


    function nullclineU22(u2;t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU11(u1;t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 

    return nullclineU21, nullclineU12, nullclineU22, nullclineU11
end 

function nullclines2d(UDE,X; t = 0, upper = [0.0,0.0], lower = [1.0,1.0])
    RHS = get_right_hand_side(UDE)
    
    function nullclineU21(u2;X = X,t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],X,t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU12(u1;X=X,t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],X,t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 


    function nullclineU22(u2;X = X,t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],X,t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU11(u1;X=X,t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],X,t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 

    return nullclineU21, nullclineU12, nullclineU22, nullclineU11
end 


function vectorfield_and_nullclines(UDE;t = 0, n = 15, lower = [0.0,0.0], upper = [1.0,1.0], arrowlength = 0.2, arrow_color = "grey", xlabel = "u1", ylabel = "u2", title = "Vector Field", color_u1 = "red", color_u2 = "black",legend = :outerright)
    N21,N12,N22,N11 = UniversalDiffEq.nullclines2d(UDE,lower = lower,upper = upper,t = t)
    dx2 = (upper[2] .- lower[2]) / (5*n)
    dx1 = (upper[1] .- lower[1]) / (5*n)
    x2 = lower[2]:dx2:upper[2]; n21 = N21.(x2); n22 = N22.(x2)
    x1 = lower[1]:dx1:upper[1]; n12 = N12.(x1); n11 = N11.(x1)

    p1 = UniversalDiffEq.vectorfield2d(UDE, t = t, title = title , n = n, lower = lower , upper =  upper , arrowlength=arrowlength,arrow_color = arrow_color)
    Plots.plot!(p1,n21,x2,  label = L"\frac{dx}{dt} = 0", color= color_u1 ,width = 2,xlabel=xlabel,ylabel=ylabel)
    Plots.plot!(p1,n22,x2, label = L"\frac{dy}{dt} = 0", color= color_u2,width = 2,legend = legend)
    Plots.plot!(p1,x1,n12,  label = "", color= color_u2,width = 2)
    Plots.plot!(p1,x1,n11,  label ="", color=  color_u1,width = 2)
    return p1
end


function vectorfield_and_nullclines(UDE,X;t = 0, n = 15, lower = [0.0,0.0], upper = [1.0,1.0], arrowlength = 0.2, arrow_color = "grey", xlabel = "u1", ylabel = "u2", title = "Vector Field", color_u1 = "red", color_u2 = "black",legend = :outerright)
    N21,N12,N22,N11 = UniversalDiffEq.nullclines2d(UDE,X,lower = lower,upper = upper,t = t)
    dx2 = (upper[2] .- lower[2]) / (5*n)
    dx1 = (upper[1] .- lower[1]) / (5*n)
    x2 = lower[2]:dx2:upper[2]; n21 = N21.(x2); n22 = N22.(x2)
    x1 = lower[1]:dx1:upper[1]; n12 = N12.(x1); n11 = N11.(x1)

    p1 = UniversalDiffEq.vectorfield2d(UDE,X, t = t, title = title , n = n, lower = lower , upper =  upper , arrowlength=arrowlength,arrow_color = arrow_color)
    Plots.plot!(p1,n21,x2,  label = L"\frac{dx}{dt} = 0", color= color_u1 ,width = 2,xlabel=xlabel,ylabel=ylabel)
    Plots.plot!(p1,n22,x2, label = L"\frac{dy}{dt} = 0", color= color_u2,width = 2,legend = legend)
    Plots.plot!(p1,x1,n12,  label = "", color= color_u2,width = 2)
    Plots.plot!(p1,x1,n11,  label ="", color=  color_u1,width = 2)
    return p1
end


function root(UDE,lower,upper;t=0)
    RHS = get_right_hand_side(UDE)
    f = x -> RHS(x,t)
    rt = nlsolve(f, (upper .- lower) .* rand(length(lower)) .+ lower)
    return rt.zero
end 


function roots_(UDE,lower,upper,Ntrials;t=0,tol=10^-3)
    roots = [root(UDE,lower,upper;t=0)]
    for i in 1:Ntrials
        rt = root(UDE,lower,upper;t=t)
        new = true
        for root in roots
            if sum((root .- rt).^2) < tol
                new = false
            end
            
            if any(isnan.(rt))
                new = false
            end
        end
        if new & !(any(rt .<lower)|any(rt .> upper))
            push!(roots,rt)
        end
    end
    return roots
end

function jacobian(UDE,x,t)
    RHS = get_right_hand_side(UDE)
    f = x -> RHS(x,t)
    return FiniteDiff.finite_difference_jacobian(f,x)
end

function eigen_values(UDE,x,t)
   J = jacobian(UDE,x,t)
   magnitudes = real.(eigvals(J))
   return magnitudes[argmax(magnitudes)]
end

"""
    equilibrium_and_stability(UDE,lower,upper;t=0,Ntrials=100,tol=10^-3)

Attempts to find all the equilibirum points for the UDE model between the upper and lower bound and return the real component of the leading eigen value to analyze stability. 

...
# kwargs
- t = 0: The point in time where the UDE model is evaluated, only relevant for time aware UDEs.
- Ntrials = 100: the number of initializations of the root finding algorithm. 
- tol = 10^-3: The threshold euclidean distance between point beyond which a new equilbirum is sufficently differnt to be retained. 
...
"""
function equilibrium_and_stability(UDE,lower,upper;t=0,Ntrials=100,tol=10^-3)
    rts = roots_(UDE,lower,upper,Ntrials;t=t,tol = tol)
    srs = []
    for rt in rts
        push!(srs,eigen_values(UDE,rt,t))
    end
    return rts,srs
end 



function root(UDE,X,lower,upper;t=0)
    RHS = get_right_hand_side(UDE)
    f = x -> RHS(x,X,t)
    rt = nlsolve(f, (upper .- lower) .* rand(length(lower)) .+ lower)
    return rt.zero
end 



function roots_(UDE,X,lower,upper,Ntrials;t=0,tol=10^-3)
    roots = [root(UDE,X,lower,upper;t=0)]
    for i in 1:Ntrials
        rt = root(UDE,X,lower,upper;t=t)
        new = true
        for root in roots
            if sum((root .- rt).^2) < tol
                new = false
            end
            if any(isnan.(rt))
                new = false
            end
        end
        if new & !(any(rt .<lower)|any(rt .> upper))
            push!(roots,rt)
        end
    end
    return roots
end

function jacobian(UDE,x,X,t)
    RHS = get_right_hand_side(UDE)
    f = x -> RHS(x,X,t)
    return FiniteDiff.finite_difference_jacobian(f,x)
end

function eigen_values(UDE,x,X,t)
   J = jacobian(UDE,x,X,t)
   magnitudes = real.(eigvals(J))
   return magnitudes[argmax(magnitudes)]
end

"""
    equilibrium_and_stability(UDE,X,lower,upper;t=0,Ntrials=100,tol=10^-3)

Attempts to find all the equilibirum points for the UDE model between the upper and lower bound and return the real component of the leading eigen value to analyze stability. 

...
# kwargs
- t = 0: The point in time where the UDE model is evaluated, only relevant for time aware UDEs.
- Ntrials = 100: the number of initializations of the root finding algorithm. 
- tol = 10^-3: The threshold euclidean distance between point beyond which a new equilbirum is sufficently differnt to be retained. 
...
"""
function equilibrium_and_stability(UDE,X,lower,upper;t=0,Ntrials=100,tol=10^-3)
    rts = roots_(UDE,X,lower,upper,Ntrials;t=t,tol = tol)
    srs = []
    for rt in rts
        push!(srs,eigen_values(UDE,rt,X,t))
    end
    return rts,srs
end 