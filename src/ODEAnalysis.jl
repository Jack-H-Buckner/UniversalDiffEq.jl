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
    
    function nullclineU1(u2;t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU2(u1;t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 


    function nullclineU12(u2;t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU21(u1;t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 

    return nullclineU1, nullclineU2, nullclineU12, nullclineU12
end 

function nullclines2d(UDE,X; t = 0, upper = [0.0,0.0], lower = [1.0,1.0])
    RHS = get_right_hand_side(UDE)
    
    function nullclineU1(u2;X = X,t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],X,t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU2(u1;X=X,t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],X,t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 


    function nullclineU12(u2;X = X,t=t,upper = upper[1], lower = lower[1])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([x, u2],X,t)[2], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 

    end 
    function nullclineU21(u1;X=X,t = t,upper = upper[2], lower = lower[2])
        val = 0
        try
            val = Roots.find_zero(x -> RHS([u1,x],X,t)[1], (lower,upper), Roots.Bisection())
        catch 
            #print("No root found in interval returning lower bound")
            val =  NaN
        end
        return val 
    end 

    return nullclineU1, nullclineU2, nullclineU12, nullclineU12
end 


function vectorfield_and_nullclines(UDE;t = 0, n = 15, lower = [0.0,0.0], upper = [1.0,1.0], arrowlength = 0.2, arrow_color = "grey", xlabel = "u1", ylabel = "u2", title = "Vector Field", color_u1 = "red", color_u2 = "black",legend = :outerright)
    N1,N2,N12,N21 = UniversalDiffEq.nullclines2d(UDE,lower = lower,upper = upper,t = t)
    x2 = -0.05:0.05:4.0; n21 = N1.(x2); n22 = N12.(x2)
    x1 = -0.0025:0.0025:0.3; n12 = N2.(x1); n11 = N21.(x1)

    p1 = UniversalDiffEq.vectorfield2d(UDE, t = t, title = title , n = n, lower = lower , upper =  upper , arrowlength=arrowlength,arrow_color = arrow_color)
    Plots.plot!(p1,n21,x2,  label = L"\frac{dx}{dt} = 0", color= color_u1 ,width = 2,xlabel=xlabel,ylabel=ylabel)
    Plots.plot!(p1,n22,x2, label = L"\frac{dy}{dt} = 0", color= color_u2,width = 2,legend = legend)
    Plots.plot!(p1,x1,n12,  label = "", color= color_u1,width = 2)
    Plots.plot!(p1,x1,n11,  label ="", color=  color_u2,width = 2)
    return p1
end


function vectorfield_and_nullclines(UDE,X;t = 0, n = 15, lower = [0.0,0.0], upper = [1.0,1.0], arrowlength = 0.2, arrow_color = "grey", xlabel = "u1", ylabel = "u2", title = "Vector Field", color_u1 = "red", color_u2 = "black",legend = :outerright)
    N1,N2,N12,N21 = UniversalDiffEq.nullclines2d(UDE,X,lower = lower,upper = upper,t = t)
    dx2 = (upper[2] .- lower[2]) / (5*n)
    dx1 = (upper[1] .- lower[1]) / (5*n)
    x2 = lower[2]:dx2:upper[2]; n21 = N1.(x2); n22 = N12.(x2)
    x1 = lower[1]:dx1:upper[1]; n12 = N2.(x1); n11 = N21.(x1)

    p1 = UniversalDiffEq.vectorfield2d(UDE,X, t = t, title = title , n = n, lower = lower , upper =  upper , arrowlength=arrowlength,arrow_color = arrow_color)
    Plots.plot!(p1,n21,x2,  label = L"\frac{dx}{dt} = 0", color= color_u1 ,width = 2,xlabel=xlabel,ylabel=ylabel)
    Plots.plot!(p1,n22,x2, label = L"\frac{dy}{dt} = 0", color= color_u2,width = 2,legend = legend)
    Plots.plot!(p1,x1,n12,  label = "", color= color_u1,width = 2)
    Plots.plot!(p1,x1,n11,  label ="", color=  color_u2,width = 2)
    return p1
end

