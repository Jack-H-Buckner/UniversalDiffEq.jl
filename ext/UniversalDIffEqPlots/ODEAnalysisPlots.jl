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


"""
    plot_bifurcation_diagram(model::UDE, xvariable; N = 25, color_variable= nothing, conditional_variable = nothing, size= (600, 400))

This function returns a plot of the equilibrium values of the state varaibles ``y_t`` as a funciton of the covariates ``X_t``. The arguemnt `xvariable` determines the covariate plotted on the x-axis. Additional variables can be visualized in sperate panel by specifying the `conditional_variable` key word argument or visualized by the color scheme using the `color_variable` argument.

The key word arguent `size` controls the dimensions of the final plot.
"""
function plot_bifurcation_diagram(model::UDE, xvariable; N = 25, color_variable= nothing, conditional_variable = nothing, size= (600, 400))
    # compute equilibriums
    data = bifurcation_data(model;N=N)

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


"""
    plot_bifurcation_diagram(model::UDE, xvariable; N = 25, color_variable= nothing, conditional_variable = nothing, size= (600, 400))

This function returns a plot of the equilibrium values of the state varaibles ``y_t`` as a funciton of the covariates ``X_t``. The arguemnt `xvariable` determines the covariate plotted on the x-axis. Additional variables can be visualized in sperate panel by specifying the `conditional_variable` key word argument or visualized by the color scheme using the `color_variable` argument.

The time sereis are treated as an additional covariate that can be visualized by setting the `color_variable` or `conditional_variable` equal to "series" or the series column name in the training data.

The key word arguent `size` controls the dimensions of the final plot.
"""
function plot_bifurcation_diagram(model::MultiUDE, xvariable; N=25, color_variable=nothing, conditional_variable=nothing, size= (600, 400))

    series = nothing
    if (conditional_variable == model.series_column_name) | (conditional_variable == "series")
        series = "Panels"
        conditional_variable=nothing
    elseif (color_variable == model.series_column_name) | (color_variable == "series")
        series = "Color"
        color_variable=nothing
    end

    # compute equilibriums
    data = bifurcation_data(model;N=N)

    # transform data frame
    nms,xnms = get_variable_names(model)
    data = melt(data, id_vars = vcat(xnms,[model.series_column_name,"eigen"]))

    conditional_levels = [0]
    conditional_values = [0]
    plts = []
    for yvariable in nms
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
        n = length(unique(data[:,"variable"]))
        m = length(unique(data[:,model.series_column_name]))
        plt = plot(plts...,layout =(n,m), size = size)
    elseif series == "Color"
        n = length(unique(data[:,"variable"]))
        m = length(conditional_levels)
        plt = plot(plts...,layout =(n,m), size = size)
    else
        n = length(unique(data[:,"variable"]))
        m = length(conditional_levels)
        print(n,m)
        plt = plot(plts...,layout =(n,m), size = size)
    end

    return plt
end