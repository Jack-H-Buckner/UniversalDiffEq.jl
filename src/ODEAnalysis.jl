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

"""
vectorfield_and_nullclines(UDE; kwargs)

Calculate the vector field and nullclines of the 2D `UDE` model and returns their plot.

# kwargs
-`t`: Time step `t` at which the vector field and nullclines are calculated. Default is `0`.
-`n`: Number of elements per axes to evaluate vector field at. Default is `15`.
-`lower`: Lower limits of vector field and nullclines. Default is `[0.0,0.0]`.
-`upper`: Upper limits of vector field and nullclines. Default is `[1.0,1.0]`.
-`arrowlength`: Arrow size of vector field plot. Default is `2`.
-`arrow_color`: Arrow color of vector field plot. Default is `grey`.
-`xlabel`: X-label of vector field plot. Default is `u1`.
-`ylabel`: Y-label of vector field plot. Default is `u2`.
-`title`: Plot title. Default is `Vector field`.
-`color_u1`: Color of nullcline in x-axis. Default is `"red"`.
-`color_u2`: Color of nullcline in y-axis. Default is `"black"`.
-`legend`: Position of legends of nullcines in plot. Default is `:outerright`.
"""
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
- tol = 10^-3: The threshold euclidean distance between point beyond which a new equilbirum is sufficently different to be retained. 
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
    return rt.zero, sum(abs.(f(rt.zero)))
end 



function roots_(UDE,X,lower,upper,Ntrials;t=0,tol=10^-3, tol2 = 10^-8)
    roots = [] #root(UDE,X,lower,upper;t=0)[1]
    for i in 1:Ntrials
        rt, norm = root(UDE,X,lower,upper;t=t)
        new = true
        for root in roots
            if sum((root .- rt).^2) < tol
                new = false
            end
            if any(isnan.(rt))
                new = false
            end
        end
        if (new & !(any(rt .<lower)|any(rt .> upper))) & (norm < tol2)
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
- tol = 10^-3: The threshold euclidean distance between point beyond which a new equilbirum is sufficently different to be retained. 
...
"""
function equilibrium_and_stability(UDE,X,lower,upper;t=0,Ntrials=100,tol=10^-3,tol2 = 10^-6)
    rts = roots_(UDE,X,lower,upper,Ntrials;t=t,tol = tol, tol2=tol2)
    srs = []
    for rt in rts
        push!(srs,eigen_values(UDE,rt,X,t))
    end
    return rts,srs
end 


### mutiple time series ODE analysis 

using NLsolve, FiniteDiff, LinearAlgebra
function root(UDE::MultiUDE,site,X,lower,upper;t=0)
    RHS = get_right_hand_side(UDE)
    f = x -> RHS(x,site,X,t)
    rt = nlsolve(f, (upper .- lower) .* rand(length(lower)) .+ lower)
    return rt.zero, sum(abs.(f(rt.zero)))
end 



function roots_(UDE::MultiUDE,site,X,lower,upper,Ntrials;t=0,tol=10^-3, tol2 = 10^-8)
    roots = [] #root(UDE,X,lower,upper;t=0)[1]
    for i in 1:Ntrials
        rt, norm = root(UDE,site,X,lower,upper;t=t)
        new = true
        for root in roots
            if sum((root .- rt).^2) < tol
                new = false
            end
            if any(isnan.(rt))
                new = false
            end
        end
        if (new & !(any(rt .<lower)|any(rt .> upper))) & (norm < tol2)
            push!(roots,rt)
        end
    end
    return roots
end


function jacobian(UDE::MultiUDE,x,site,X,t)
    RHS = get_right_hand_side(UDE)
    f = x -> RHS(x,site,X,t)
    return FiniteDiff.finite_difference_jacobian(f,x)
end

function eigen_values(UDE::MultiUDE,x,site,X,t)
   J = jacobian(UDE,x,site,X,t)
   magnitudes = real.(eigvals(J))
   return magnitudes[argmax(magnitudes)]
end

"""
    equilibrium_and_stability(UDE::MultiUDE,site,X,lower,upper;t=0,Ntrials=100,tol=10^-3)

Attempts to find all the equilibirum points for the UDE model between the upper and lower bound and return the real component of the leading eigen value to analyze stability. 

...
# kwargs
- t = 0: The point in time where the UDE model is evaluated, only relevant for time aware UDEs.
- Ntrials = 100: the number of initializations of the root finding algorithm. 
- tol = 10^-3: The threshold euclidean distance between point beyond which a new equilbirum is sufficently different to be retained. 
...
"""
function equilibrium_and_stability(UDE::MultiUDE,site,X,lower,upper;t=0,Ntrials=20,tol=10^-3,tol2 = 10^-6)
    rts = roots_(UDE,site,X,lower,upper,Ntrials;t=t,tol = tol, tol2=tol2)
    srs = []
    for rt in rts
        push!(srs,eigen_values(UDE,rt,site,X,t))
    end
    return rts,srs
end 



function arguments(UDE)
    if (string(typeof(UDE)) .== "MultiUDE") | (string(typeof(UDE)) .== "UniversalDiffEq.MultiUDE")
        if UDE.process_model.covariates == 0
            println("Right hand side: f(u::Vector,site::Int,t:Float)")
            println("process_model.predict: f(u::Vector,i::Int,t::Float,dt::Float,parameters::ComponentArray)")
        else
            println("Right hand side: f(u::Vector,site::Int,X::Vector,t:Float)")
            println("process_model.predict: f(u::Vector,i::Int,t::Float,dt::Float,parameters::ComponentArray)")       
        end
    elseif (string(typeof(UDE)) .== "UDE") | (string(typeof(UDE)) .== "UniversalDiffEq.UDE")
        if UDE.process_model.covariates == 0
            println("Right hand side: f(u::Vector,t:Float)")
            println("process_model.predict: f(u::Vector,t::Float,dt::Float,parameters::ComponentArray)")
        else
            println("Right hand side: f(u::Vector,X::Vector,t:Float)")
            println("process_model.predict: f(u::Vector,t::Float,dt::Float,parameters::ComponentArray)")        
        end
    end
    print("Not applicable the arguemnt is not a UDE model ")
    return
end


# Bifrucation diagrams

function get_variable_names(model::UDE)
    # state variable names 
    nms = names(model.data_frame)
    nms = nms[nms .!= model.time_column_name]

    # Covariates names 
    xnms = []
    if typeof(model.variable_column_name) == Nothing
        xnms = names(model.X_data_frame)
        xnms = xnms[xnms .!= model.time_column_name]
    else
        xnms = unique(model.X_data_frame[:,model.variable_column_name])
    end 
    return nms, xnms
end 

"""
    bifructaion_data(model::UDE;N=25)

Calcualtes the equilibrium values of the state variabels ``y_t`` as a function of the covariates `X_t` and return the value in a data frame. The funciton calcualtes the equilibrium values on a grid of ``N`` evenly spaced point for each covariate. 
"""
function bifructaion_data(model::UDE;N=25)

    X = model.process_model.covariates.(model.times)
    dims = length(X[1])
    values = zeros(N,dims)
    for i in 1:dims
        x_min = minimum(broadcast(t -> X[t][i], 1:length(X)))
        x_max = maximum(broadcast(t -> X[t][i], 1:length(X)))
        range = x_max - x_min
        values[:,i] = collect(x_min:(range/N):x_max)[1:N]
    end

    u_dims = size(model.data)[1]
    umin = zeros(u_dims)
    umax = zeros(u_dims)
    for i in 1:u_dims
        x_min = minimum(model.data[i,:])
        x_max = maximum(model.data[i,:])
        range = x_max - x_min
        x_mean = (x_max - x_min)/2
        x_max = 3*(x_max - x_mean) + x_mean
        x_min = 3*(x_min - x_mean) + x_mean
        umin[i] = x_min
        umax[i] = x_max
    end

    vals = values[:,1]
    for d in 2:dims
        vals = vcat.(vals,values[:,d]')
        vals = reshape(vals,length(vals))
    end 

    data = zeros(length(vals)*6, length(vals[1])+size(model.data)[1]+1)
    n = 0; i = 0
    for x in vals
        rts, stb = UniversalDiffEq.equilibrium_and_stability(model,x,umin,umax;t=0,Ntrials=20,tol=10^-3,tol2 = 10^-6)
        i = 0
        for rt in rts
            n += 1; i +=1
            data[n,:] = vcat(vcat(x,rt), [stb[i]])
        end
    end 

    # filter out extra zeros
    data = data[data[:,1] .!= 0.0,:]

    nms, xnms = get_variable_names(model)
    df = DataFrame(data, vcat(xnms, vcat(nms, ["eigen"])))
    df = df[df[:,"eigen"] .!== 0,:]
    
    return df

end


"""
    plot_bifrucation_diagram(model::UDE, xvariable; N = 25, color_variable= nothing, conditional_variable = nothing, size= (600, 400))
    
This function returns a plot of the equilibrium values of the state varaibles ``y_t`` as a funciton of the covariates ``X_t``. The arguemnt `xvariable` determines the covariate plotted on the x-axis. Additional variables can be visualized in sperate panel by specifying the `conditional_variable` key word argument or visualized by the color scheme using the `color_variable` argument. 

The key word arguent `size` controls the dimensions of the final plot. 
"""
function plot_bifrucation_diagram(model::UDE, xvariable; N = 25, color_variable= nothing, conditional_variable = nothing, size= (600, 400))
    # compute equilibriums
    data = bifructaion_data(model;N=N)
    
    # transform data frame
    nms,xnms = get_variable_names(model)
    data = melt(data, id_vars = vcat(xnms,["eigen"]))

    conditional_levels = [0]
    conditional_values = [0]
    plts = []
    for yvariable in unique(data.variable)
        dat = data[data.variable .== yvariable,:]

        if typeof(conditional_variable) == Nothing

            if typeof(color_variable) == Nothing
                plt = Plots.scatter( dat[:,xvariable], dat.value, markersize = 3.5 .- 1.5 *(dat.eigen .> 0.0),
                                    label = "", ylabel = string("Eg. ", yvariable),
                                    xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, color= "black")
                push!(plts,plt)
            else
                plt = Plots.scatter( dat[:,xvariable], dat.value, markersize = 3.5 .- 1.5 *(dat.eigen .> 0.0),
                                    label = "", ylabel = string("Eg. ", yvariable),
                                    xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, zcolor = dat[:,color_variable])
                push!(plts,plt)
            end
        else 
            conditional_values = sort(unique(data[:,conditional_variable]))
            conditional_levels = conditional_values[[1,round(Int,length(conditional_values)/2),end]]

            for X in conditional_levels

                if typeof(color_variable) == Nothing
                    dat_i = dat[broadcast(x -> x == X, dat[:,conditional_variable]),:]
                    plt = Plots.scatter( dat_i[:,xvariable], dat_i.value, markersize = 3.5 .- 1.5 *(dat_i.eigen .> 0.0),
                                        title = string( conditional_variable, " = ", round(X, digits = 2) ), 
                                        titlefontsize = 9, label = "", ylabel = string("Eg. ", yvariable),
                                        xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, color= "black")
                    push!(plts,plt)
                else
                    dat_i = dat[broadcast(x -> x == X, dat[:,conditional_variable]),:]
                    plt = Plots.scatter( dat_i[:,xvariable], dat_i.value, markersize = 3.5 .- 1.5 *(dat_i.eigen .> 0.0),
                                        title = string( conditional_variable, " = ", round(X, digits = 2) ), 
                                        titlefontsize = 9, label = "", ylabel = string("Eg. ", yvariable),
                                        xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, zcolor = dat_i[:,color_variable])
                    push!(plts,plt)
                end
            end 
        end
    end

    return plot(plts...,layout =(length(unique(data.variable)),length(conditional_levels)), size = size), data
end


# Mutiple time series bifrucation diagrams 



function get_variable_names(model::MultiUDE)
    # state variable names 
    nms = names(model.data_frame)
    nms = nms[broadcast(nm -> !(nm in [model.time_column_name,model.series_column_name]),nms)]

    # Covariates names 
    xnms = []
    if typeof(model.variable_column_name) == Nothing
        xnms = names(model.X_data_frame)
        xnms = xnms[broadcast(nm -> !(nm in [model.time_column_name,model.series_column_name]),xnms)]
    else
        xnms = unique(model.X_data_frame[:,model.variable_column_name])
    end 
    return nms, string.(xnms)
end 

"""
    bifructaion_data(model::MultiUDE;N=25)

Calcualtes the equilibrium values of the state variabels ``y_t`` as a function of the covariates `X_t` and return the value in a data frame. The funciton calcualtes the equilibrium values on a grid of ``N`` evenly spaced point for each covariate. The calcualtion are repeated for each time series ``i`` included in the training data set.  
"""
function bifructaion_data(model::MultiUDE; N=25)

    X = model.process_model.covariates.(model.times,1)
    dims = length(X[1])
    series = unique(model.data_frame[:,model.series_column_name])
    values = zeros(N,length(series)*dims)
    x_min = nothing
    x_max = nothing
    for s in series
        X = model.process_model.covariates.(model.times,s)
        for i in 1:dims
            if typeof(x_min) == Nothing
                x_min = minimum(broadcast(t -> X[t][i], 1:length(X)))
                x_max = maximum(broadcast(t -> X[t][i], 1:length(X)))
            elseif x_min > minimum(broadcast(t -> X[t][i], 1:length(X)))
                x_min = minimum(broadcast(t -> X[t][i], 1:length(X)))
            elseif x_max < maximum(broadcast(t -> X[t][i], 1:length(X)))
                x_max = maximum(broadcast(t -> X[t][i], 1:length(X)))
            end
            
            range = x_max - x_min
            values[:,i] = collect(x_min:(range/N):x_max)[1:N]
        end
    end

    u_dims = size(model.data)[1]
    umin = zeros(u_dims)
    umax = zeros(u_dims)
    for i in 1:u_dims
        x_min = minimum(model.data[i,:])
        x_max = maximum(model.data[i,:])
        range = x_max - x_min
        x_mean = (x_max - x_min)/2
        x_max = 3*(x_max - x_mean) + x_mean
        x_min = 3*(x_min - x_mean) + x_mean
        umin[i] = x_min
        umax[i] = x_max
    end

    vals = values[:,1]
    for d in 2:dims
        vals = vcat.(vals,values[:,d]')
        vals = reshape(vals,length(vals))
    end 

    
    data = zeros(length(vals)*6*length(series), length(vals[1])+size(model.data)[1]+2)
    n = 0; i = 0

    for s in series
        for x in vals
            rts, stb = UniversalDiffEq.equilibrium_and_stability(model,s,x,umin,umax;t=0,Ntrials=20,tol=10^-3,tol2 = 10^-6)
            i = 0
            for rt in rts
                n += 1; i +=1
                data[n,:] = vcat(vcat(x,rt), [s,stb[i]])
            end
        end 
    end
    # filter out extra zeros
    data = data[data[:,1] .!= 0.0,:]

    nms, xnms = get_variable_names(model)
    df = DataFrame(data, vcat(xnms, vcat(nms, [model.series_column_name, "eigen"])))
    df = df[df[:,"eigen"] .!== 0,:]
    
    return df

end

"""
    plot_bifrucation_diagram(model::UDE, xvariable; N = 25, color_variable= nothing, conditional_variable = nothing, size= (600, 400))
    
This function returns a plot of the equilibrium values of the state varaibles ``y_t`` as a funciton of the covariates ``X_t``. The arguemnt `xvariable` determines the covariate plotted on the x-axis. Additional variables can be visualized in sperate panel by specifying the `conditional_variable` key word argument or visualized by the color scheme using the `color_variable` argument. 

The time sereis are treated as an additional covariate that can be visualized by setting the `color_variable` or `conditional_variable` equal to "series" or the series column name in the training data. 

The key word arguent `size` controls the dimensions of the final plot. 
"""
function plot_bifrucation_diagram(model::MultiUDE, xvariable; N=25, color_variable=nothing, conditional_variable=nothing, size= (600, 400))
    
    series = nothing
    if (conditional_variable == model.series_column_name) | (conditional_variable == "series")
        series = "Panels"
        conditional_variable=nothing
    elseif (color_variable == model.series_column_name) | (color_variable == "series")
        series = "Color"
        color_variable=nothing
    end 

    # compute equilibriums
    data = bifructaion_data(model;N=N)

    # transform data frame
    nms,xnms = get_variable_names(model)
    data = melt(data, id_vars = vcat(xnms,[model.series_column_name,"eigen"]))

    conditional_levels = [0]
    conditional_values = [0]
    plts = []
    for yvariable in unique(data[:,model.variable_column_name])
        dat = data[data.variable .== yvariable,:]

        if series == "Panels"
            series_vals = unique(model.data_frame[:,model.series_column_name])
            for s in series_vals
                if typeof(color_variable) == Nothing
                    dat_s = dat[dat[:,model.series_column_name] .== s,:]
                    plt = Plots.scatter( dat_s[:,xvariable], dat_s.value, markersize = 3.5 .- 1.5 *(dat_s.eigen .> 0.0),
                                        label = "", ylabel = string("Eg. ", yvariable), 
                                        title = string(model.series_column_name," = ",s),titlefontsize = 9,
                                        xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, color= "black")
                    push!(plts,plt)
                else
                    dat_s = dat[dat[:,model.series_column_name] .== s,:]
                    plt = Plots.scatter( dat_s[:,xvariable], dat_s.value, markersize = 3.5 .- 1.5 *(dat_s.eigen .> 0.0),
                                        label = "", ylabel = string("Eg. ", yvariable), titlefontsize = 9,
                                        xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, 
                                        title = string(model.series_column_name," = ",s),
                                        zcolor = dat_s[:,color_variable])
                    push!(plts,plt)
                end
            end
        elseif series == "Color"

            if typeof(conditional_variable) == Nothing

                plt = Plots.scatter( dat[:,xvariable], dat.value, markersize = 3.5 .- 1.5 *(dat.eigen .> 0.0),
                                    label = "", ylabel = string("Eg. ", yvariable),
                                    xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, 
                                    zcolor = dat[:,model.series_column_name])
                push!(plts,plt)

            else 
                conditional_values = sort(unique(data[:,conditional_variable]))
                conditional_levels = conditional_values[[1,round(Int,length(conditional_values)/2),end]]

                for X in conditional_levels
                    dat_i = dat[broadcast(x -> x == X, dat[:,conditional_variable]),:]
                    plt = Plots.scatter( dat_i[:,xvariable], dat_i.value, markersize = 3.5 .- 1.5 *(dat_i.eigen .> 0.0),
                                        title = string( conditional_variable, " = ", round(X, digits = 2) ), 
                                        titlefontsize = 9, label = "", ylabel = string("Eg. ", yvariable),
                                        xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, 
                                        zcolor = dat_i[:,model.series_column_name])
                    push!(plts,plt)
                end 
            end

        else
            if typeof(conditional_variable) == Nothing

                if typeof(color_variable) == Nothing
                    plt = Plots.scatter( dat[:,xvariable], dat.value, markersize = 3.5 .- 1.5 *(dat.eigen .> 0.0),
                                        label = "", ylabel = string("Eg. ", yvariable),
                                        xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, color= "black")
                    push!(plts,plt)
                else
                    plt = Plots.scatter( dat[:,xvariable], dat.value, markersize = 3.5 .- 1.5 *(dat.eigen .> 0.0),
                                        label = "", ylabel = string("Eg. ", yvariable),
                                        xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, 
                                        zcolor = dat[:,color_variable])
                    push!(plts,plt)
                end
            else 
                conditional_values = sort(unique(data[:,conditional_variable]))
                conditional_levels = conditional_values[[1,round(Int,length(conditional_values)/2),end]]

                for X in conditional_levels

                    if typeof(color_variable) == Nothing
                        dat_i = dat[broadcast(x -> x == X, dat[:,conditional_variable]),:]
                        plt = Plots.scatter( dat_i[:,xvariable], dat_i.value, markersize = 3.5 .- 1.5 *(dat_i.eigen .> 0.0),
                                            title = string( conditional_variable, " = ", round(X, digits = 2) ), 
                                            titlefontsize = 9, label = "", ylabel = string("Eg. ", yvariable),
                                            xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, color= "black")
                        push!(plts,plt)
                    else
                        dat_i = dat[broadcast(x -> x == X, dat[:,conditional_variable]),:]
                        plt = Plots.scatter( dat_i[:,xvariable], dat_i.value, markersize = 3.5 .- 1.5 *(dat_i.eigen .> 0.0),
                                            title = string( conditional_variable, " = ", round(X, digits = 2) ), 
                                            titlefontsize = 9, label = "", ylabel = string("Eg. ", yvariable),
                                            xlabel= xvariable, labelfontsize = 9, tickfontsize = 7, 
                                            zcolor = dat_i[:,color_variable])
                        push!(plts,plt)
                    end
                end 
            end
        end 

    end

    plt = 0
    if series == "Panels"
        n = length(unique(data[:,model.variable_column_name]))
        m = length(unique(data[:,model.series_column_name]))
        plt = plot(plts...,layout =(n,m), size = size)
    elseif series == "Color"
        n = length(unique(data[:,model.variable_column_name]))
        m = length(conditional_levels)
        plt = plot(plts...,layout =(n,m), size = size)
    else
        n = length(unique(data[:,model.variable_column_name]))
        m = length(conditional_levels)
        print(n,m)
        plt = plot(plts...,layout =(n,m), size = size)
    end

    return plt
end